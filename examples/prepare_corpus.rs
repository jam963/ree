//! Offline, deterministic corpus selection. No model code or downloads.
#[path = "support/quality.rs"]
mod quality;

use anyhow::{Context, Result, ensure};
use clap::Parser;
use quality::{Corpus, Document, Query};
use ree::util::{hash, read_limited};
use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::{BufRead, BufReader, Read, Write},
    path::{Path, PathBuf},
};

const VERSION: &str = "ree-subset-v1";
const MAX_LINE: u64 = 1024 * 1024;
const MAX_SMALL_FILE: u64 = 128 * 1024 * 1024;
const MAX_TEXT: usize = 96 * 1024 * 1024;

#[derive(Parser)]
struct Args {
    /// Hash-locked local inputs and selection recipe; see docs/evaluation-protocol.md.
    #[arg(long)]
    manifest: PathBuf,
    /// New file only. Published atomically after all inputs validate.
    #[arg(long)]
    output: PathBuf,
    /// Disk-backed ID/qrel validation workspace (removed on success or failure).
    #[arg(long)]
    scratch_dir: PathBuf,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Input {
    path: PathBuf,
    sha256: String,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    version: String,
    dataset: String,
    source_url: String,
    revision: String,
    split: String,
    domain: String,
    license: String,
    license_reference: String,
    /// Exact upstream-to-local conversion/miner recipe; never executed here.
    preparation: String,
    seed: String,
    query_count: usize,
    document_count: usize,
    /// Audited claim about the upstream corpus, not inferred from local file size.
    source_is_complete: bool,
    exclude_query_id: bool,
    /// Minimum pinned ranked distractors per query when not using the full corpus.
    hard_negatives_per_query: usize,
    documents: Input,
    queries: Input,
    qrels: Input,
    negatives: Option<Input>,
}
#[derive(Deserialize)]
struct SourceDocument {
    #[serde(rename = "_id")]
    id: String,
    #[serde(default)]
    title: String,
    text: String,
}
#[derive(Deserialize)]
struct SourceQuery {
    #[serde(rename = "_id")]
    id: String,
    text: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Negatives {
    query_id: String,
    /// Ranked best-first; positives are skipped, not relabeled as negatives.
    document_ids: Vec<String>,
}

fn valid_hash(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
impl Manifest {
    fn validate(&self) -> Result<()> {
        ensure!(self.version == VERSION, "unsupported subset version");
        for value in [
            &self.dataset,
            &self.source_url,
            &self.revision,
            &self.split,
            &self.domain,
            &self.license,
            &self.license_reference,
            &self.preparation,
            &self.seed,
        ] {
            ensure!(
                !value.trim().is_empty(),
                "manifest provenance/seed must not be empty"
            );
        }
        ensure!(
            !["main", "master", "latest"].contains(&self.revision.as_str()),
            "dataset revision must be immutable, not a moving branch"
        );
        ensure!(
            (1..=1000).contains(&self.query_count)
                && (1..=10000).contains(&self.document_count)
                && (1..=100).contains(&self.hard_negatives_per_query),
            "invalid subset limits"
        );
        for input in [&self.documents, &self.queries, &self.qrels]
            .into_iter()
            .chain(self.negatives.iter())
        {
            ensure!(
                valid_hash(&input.sha256),
                "input requires lowercase SHA-256"
            );
        }
        Ok(())
    }
}

/// Hash exactly the bytes parsed, including newlines. Bounded lines and total bytes;
/// no hash-then-reopen race and no read_to_string of a multi-million-document corpus.
fn lines(
    input: &Input,
    base: &Path,
    max_bytes: u64,
    mut visit: impl FnMut(usize, &str) -> Result<()>,
) -> Result<()> {
    let path = base.join(&input.path);
    let mut reader =
        BufReader::new(File::open(&path).with_context(|| format!("open {}", path.display()))?);
    let mut digest = Sha256::new();
    let mut total = 0u64;
    let mut line = Vec::new();
    let mut number = 0;
    loop {
        line.clear();
        let n = reader
            .by_ref()
            .take(MAX_LINE + 1)
            .read_until(b'\n', &mut line)?;
        if n == 0 {
            break;
        }
        number += 1;
        total += n as u64;
        ensure!(
            n as u64 <= MAX_LINE && total <= max_bytes,
            "{} exceeds input limits",
            path.display()
        );
        digest.update(&line);
        let text = std::str::from_utf8(&line)?.trim_end_matches(['\r', '\n']);
        ensure!(
            !text.is_empty(),
            "{}:{number}: blank record",
            path.display()
        );
        visit(number, text).with_context(|| format!("{}:{number}", path.display()))?;
    }
    ensure!(
        format!("{:x}", digest.finalize()) == input.sha256,
        "checksum mismatch: {}",
        path.display()
    );
    Ok(())
}

// Length-prefixed components avoid ambiguous concatenation; domain separation
// keeps query selection and document fill independent. Lexical ID breaks hash ties.
fn rank(seed: &str, kind: &str, id: &str) -> String {
    let mut digest = Sha256::new();
    for component in [VERSION, seed, kind, id] {
        digest.update((component.len() as u64).to_be_bytes());
        digest.update(component.as_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn prepare(
    manifest: &Manifest,
    base: &Path,
    scratch: &Path,
    manifest_hash: &str,
) -> Result<Corpus> {
    manifest.validate()?;
    // ID and qrel indexes live on disk, not one large in-memory map of the corpus.
    let workspace = tempfile::tempdir_in(scratch)?;
    let mut db = Connection::open(workspace.path().join("selection.db"))?;
    db.execute_batch(
        "PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;
        PRAGMA temp_store=FILE; PRAGMA cache_size=-8192;
        CREATE TABLE documents (id TEXT PRIMARY KEY) WITHOUT ROWID;
        CREATE TABLE qrels (query TEXT, doc TEXT, grade INTEGER NOT NULL,
            PRIMARY KEY(query, doc)) WITHOUT ROWID;",
    )?;
    let mut queries = BTreeMap::new();
    lines(&manifest.queries, base, MAX_SMALL_FILE, |_, line| {
        let query: SourceQuery = serde_json::from_str(line)?;
        ensure!(
            !query.id.is_empty() && !query.text.trim().is_empty(),
            "empty query ID/text"
        );
        ensure!(
            queries.insert(query.id, query.text).is_none(),
            "duplicate query ID"
        );
        ensure!(queries.len() <= 100000, "more than 100000 input queries");
        Ok(())
    })?;
    let mut eligible = BTreeSet::new();
    {
        let tx = db.transaction()?;
        let mut insert = tx.prepare("INSERT INTO qrels VALUES (?1, ?2, ?3)")?;
        lines(&manifest.qrels, base, MAX_SMALL_FILE, |number, line| {
            if number == 1 {
                ensure!(
                    line == "query-id\tcorpus-id\tscore",
                    "expected BEIR qrels header"
                );
                return Ok(());
            }
            let fields: Vec<_> = line.split('\t').collect();
            ensure!(
                fields.len() == 3 && !fields[1].is_empty(),
                "invalid qrel row"
            );
            ensure!(
                queries.contains_key(fields[0]),
                "qrel references missing query {}",
                fields[0]
            );
            let grade: u32 = fields[2]
                .parse()
                .context("qrel grades must be nonnegative integers")?;
            insert
                .execute(params![fields[0], fields[1], grade])
                .context("duplicate qrel pair")?;
            if grade > 0 {
                eligible.insert(fields[0].to_owned());
            }
            Ok(())
        })?;
        drop(insert);
        tx.commit()?;
    }
    ensure!(
        eligible.len() >= manifest.query_count,
        "not enough queries with positive qrels"
    );
    let eligible_count = eligible.len();
    let mut selected: Vec<_> = eligible.into_iter().collect();
    selected.sort_by_cached_key(|id| (rank(&manifest.seed, "query", id), id.clone()));
    selected.truncate(manifest.query_count);
    selected.sort();
    let mut output_queries = Vec::new();
    let mut required = BTreeSet::new();
    let mut retained_zero_qrels = 0usize;
    let mut positive_pairs = 0usize;
    for id in &selected {
        let mut relevance = BTreeMap::new();
        let mut stmt = db.prepare("SELECT doc, grade FROM qrels WHERE query=?1 ORDER BY doc")?;
        for row in stmt.query_map([id], |r| Ok((r.get::<_, String>(0)?, r.get::<_, u32>(1)?)))? {
            let (doc, grade) = row?;
            // All judged documents, including grade-zero negatives, are mandatory.
            required.insert(doc.clone());
            if grade > 0 {
                relevance.insert(doc, grade);
                positive_pairs += 1;
            } else {
                retained_zero_qrels += 1;
            }
        }
        output_queries.push(Query {
            id: id.clone(),
            text: queries.remove(id).unwrap(),
            relevant: relevance.keys().cloned().collect(),
            relevance,
            domain: manifest.domain.clone(),
        });
    }
    let selected_set: BTreeSet<_> = selected.iter().map(String::as_str).collect();
    let mut negative_counts = BTreeMap::new();
    if let Some(input) = &manifest.negatives {
        let mut seen = BTreeSet::new();
        lines(input, base, MAX_SMALL_FILE, |_, line| {
            let negatives: Negatives = serde_json::from_str(line)?;
            ensure!(
                queries.contains_key(&negatives.query_id)
                    || selected_set.contains(negatives.query_id.as_str()),
                "negative list references missing query"
            );
            ensure!(
                seen.insert(negatives.query_id.clone()),
                "duplicate negative query ID"
            );
            let unique: BTreeSet<_> = negatives.document_ids.iter().collect();
            ensure!(
                unique.len() == negatives.document_ids.len()
                    && unique.iter().all(|id| !id.is_empty()),
                "empty/duplicate negative document ID"
            );
            if let Some(query) = output_queries.iter().find(|q| q.id == negatives.query_id) {
                let ids: Vec<_> = negatives
                    .document_ids
                    .iter()
                    .filter(|id| {
                        !query.relevance.contains_key(*id)
                            && (!manifest.exclude_query_id || *id != &query.id)
                    })
                    .take(manifest.hard_negatives_per_query)
                    .collect();
                negative_counts.insert(query.id.clone(), ids.len());
                required.extend(ids.into_iter().cloned());
            }
            Ok(())
        })?;
    }
    ensure!(
        required.len() <= manifest.document_count,
        "mandatory judgments/negatives exceed document budget; do not drop positives or skip queries"
    );
    let fill_count = manifest.document_count - required.len();
    let mut mandatory = BTreeMap::new();
    let mut fill: BTreeMap<(String, String), Document> = BTreeMap::new();
    let mut text_bytes = 0usize;
    let mut total_documents = 0usize;
    {
        let tx = db.transaction()?;
        let mut insert = tx.prepare("INSERT INTO documents VALUES (?1)")?;
        lines(
            &manifest.documents,
            base,
            8 * 1024 * 1024 * 1024,
            |_, line| {
                let doc: SourceDocument = serde_json::from_str(line)?;
                ensure!(!doc.id.is_empty(), "empty document ID");
                insert.execute([&doc.id]).context("duplicate document ID")?;
                total_documents += 1;
                let text = if doc.title.is_empty() {
                    doc.text
                } else {
                    format!("{}\n{}", doc.title, doc.text)
                };
                ensure!(!text.trim().is_empty(), "empty document text: {}", doc.id);
                if required.contains(&doc.id) {
                    text_bytes += text.len();
                    mandatory.insert(doc.id.clone(), Document { id: doc.id, text });
                } else if fill_count > 0 {
                    let key = (rank(&manifest.seed, "document", &doc.id), doc.id.clone());
                    if fill.len() < fill_count
                        || fill.last_key_value().is_some_and(|(last, _)| key < *last)
                    {
                        if fill.len() == fill_count {
                            text_bytes -= fill.pop_last().unwrap().1.text.len();
                        }
                        text_bytes += text.len();
                        fill.insert(key, Document { id: doc.id, text });
                    }
                }
                ensure!(text_bytes <= MAX_TEXT, "retained text exceeds 96 MiB limit");
                Ok(())
            },
        )?;
        drop(insert);
        tx.commit()?;
    }
    ensure!(
        mandatory.len() == required.len(),
        "mandatory qrel/negative document missing from source corpus"
    );
    let missing_qrel: bool = db.query_row(
        "SELECT EXISTS(SELECT 1 FROM qrels q LEFT JOIN documents d ON d.id=q.doc WHERE d.id IS NULL)",
        [], |r| r.get(0))?;
    ensure!(
        !missing_qrel,
        "qrel references missing source document (including unselected queries)"
    );
    let full_corpus = manifest.source_is_complete && total_documents <= manifest.document_count;
    if !full_corpus {
        ensure!(
            selected
                .iter()
                .all(|id| negative_counts.get(id) == Some(&manifest.hard_negatives_per_query)),
            "subsets require the pinned hard-negative quota for every selected query"
        );
    }
    let mut documents: Vec<_> = mandatory.into_values().chain(fill.into_values()).collect();
    documents.sort_by(|a, b| a.id.cmp(&b.id));
    let corpus = Corpus {
        documents,
        queries: output_queries,
        exclude_query_id: manifest.exclude_query_id,
        provenance: json!({"preparation_version":VERSION,"manifest_sha256":manifest_hash,
            "manifest":manifest,"eligible_queries":eligible_count,"source_documents":total_documents,
            "full_corpus":full_corpus,"positive_qrel_pairs":positive_pairs,
            "retained_zero_qrel_pairs":retained_zero_qrels,"hard_negative_counts":negative_counts,
            "selection":"all judgments + pinned ranked negatives + SHA-256 fill",
            "judgment_policy":"unjudged is not known irrelevant; positive integer grades preserved"}),
    };
    corpus.validate()?;
    Ok(corpus)
}

fn publish(corpus: &Corpus, output: &Path) -> Result<String> {
    let mut bytes = serde_json::to_vec(corpus)?;
    bytes.push(b'\n');
    ensure!(
        bytes.len() <= MAX_SMALL_FILE as usize,
        "serialized corpus exceeds 128 MiB"
    );
    let parent = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let mut temp = tempfile::NamedTempFile::new_in(parent)?;
    temp.write_all(&bytes)?;
    temp.as_file().sync_all()?;
    temp.persist_noclobber(output)
        .context("publish corpus (output must not exist)")?;
    File::open(parent)?.sync_all()?;
    Ok(hash(&bytes))
}
fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(!args.output.try_exists()?, "output must not exist");
    let bytes = read_limited(File::open(&args.manifest)?, 1024 * 1024)?;
    let manifest: Manifest = serde_json::from_slice(&bytes)?;
    let base = args.manifest.parent().unwrap_or(Path::new("."));
    let corpus = prepare(&manifest, base, &args.scratch_dir, &hash(&bytes))?;
    let sha256 = publish(&corpus, &args.output)?;
    println!(
        "{}",
        json!({"type":"corpus_prepared","path":args.output,"sha256":sha256,
        "documents":corpus.documents.len(),"queries":corpus.queries.len()})
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn input(dir: &Path, name: &str, contents: &str) -> Input {
        std::fs::write(dir.join(name), contents).unwrap();
        Input {
            path: name.into(),
            sha256: hash(contents.as_bytes()),
        }
    }
    fn fixture(dir: &Path) -> Manifest {
        Manifest {
            version: VERSION.into(),
            dataset: "fixture".into(),
            source_url: "local fixture".into(),
            revision: "fixture-v1".into(),
            split: "test".into(),
            domain: "test".into(),
            license: "MIT OR Apache-2.0".into(),
            license_reference: "repository licenses".into(),
            preparation: "hand-authored unit-test data; not qualification".into(),
            seed: "20260909".into(),
            query_count: 1,
            document_count: 4,
            source_is_complete: true,
            exclude_query_id: false,
            hard_negatives_per_query: 1,
            documents: input(
                dir,
                "docs.jsonl",
                concat!(
                    "{\"_id\":\"a\",\"title\":\"Title\",\"text\":\"alpha\"}\n",
                    "{\"_id\":\"b\",\"text\":\"beta\"}\n",
                    "{\"_id\":\"c\",\"text\":\"judged negative\"}\n",
                    "{\"_id\":\"d\",\"text\":\"hard distractor\"}\n",
                    "{\"_id\":\"e\",\"text\":\"random distractor\"}\n"
                ),
            ),
            queries: input(
                dir,
                "queries.jsonl",
                "{\"_id\":\"q\",\"text\":\"alpha?\"}\n",
            ),
            qrels: input(
                dir,
                "qrels.tsv",
                "query-id\tcorpus-id\tscore\nq\ta\t2\nq\tb\t1\nq\tc\t0\n",
            ),
            negatives: Some(input(
                dir,
                "negatives.jsonl",
                "{\"query_id\":\"q\",\"document_ids\":[\"a\",\"d\",\"e\"]}\n",
            )),
        }
    }
    #[test]
    fn preserves_all_judgments_and_ranked_negatives() {
        let dir = tempfile::tempdir().unwrap();
        let manifest = fixture(dir.path());
        let corpus = prepare(&manifest, dir.path(), dir.path(), "fixture").unwrap();
        assert_eq!(
            corpus
                .documents
                .iter()
                .map(|d| d.id.as_str())
                .collect::<Vec<_>>(),
            ["a", "b", "c", "d"]
        );
        assert_eq!(corpus.documents[0].text, "Title\nalpha");
        assert_eq!(corpus.queries[0].relevance["a"], 2);
        assert_eq!(corpus.queries[0].relevant, ["a", "b"]);
        assert_eq!(corpus.provenance["retained_zero_qrel_pairs"], 1);
        let again = prepare(&manifest, dir.path(), dir.path(), "fixture").unwrap();
        assert_eq!(
            serde_json::to_vec(&corpus).unwrap(),
            serde_json::to_vec(&again).unwrap()
        );
        let output = dir.path().join("corpus.json");
        let digest = publish(&corpus, &output).unwrap();
        assert_eq!(ree::util::hash_file(&output).unwrap(), digest);
        assert!(publish(&again, &output).is_err());
        assert_eq!(ree::util::hash_file(&output).unwrap(), digest);
    }
    #[test]
    fn refuses_easy_subsets_and_positive_loss() {
        let dir = tempfile::tempdir().unwrap();
        let mut manifest = fixture(dir.path());
        manifest.document_count = 2;
        assert!(prepare(&manifest, dir.path(), dir.path(), "fixture").is_err());
        manifest.document_count = 4;
        manifest.negatives = None;
        assert!(prepare(&manifest, dir.path(), dir.path(), "fixture").is_err());
        manifest.document_count = 5;
        manifest.source_is_complete = false;
        assert!(prepare(&manifest, dir.path(), dir.path(), "fixture").is_err());
        manifest.source_is_complete = true;
        assert!(
            prepare(&manifest, dir.path(), dir.path(), "fixture")
                .unwrap()
                .provenance["full_corpus"]
                .as_bool()
                .unwrap()
        );
    }
    #[test]
    fn rejects_corruption_duplicates_and_dangling_references() {
        let dir = tempfile::tempdir().unwrap();
        let mut manifest = fixture(dir.path());
        manifest.documents.sha256 = "0".repeat(64);
        assert!(prepare(&manifest, dir.path(), dir.path(), "fixture").is_err());
        let mut manifest = fixture(dir.path());
        manifest.qrels = input(
            dir.path(),
            "qrels.tsv",
            "query-id\tcorpus-id\tscore\nq\ta\t1\nq\ta\t1\n",
        );
        assert!(prepare(&manifest, dir.path(), dir.path(), "fixture").is_err());
        manifest.qrels = input(
            dir.path(),
            "qrels.tsv",
            "query-id\tcorpus-id\tscore\nq\tmissing\t1\n",
        );
        assert!(prepare(&manifest, dir.path(), dir.path(), "fixture").is_err());
        let mut manifest = fixture(dir.path());
        manifest.documents = input(
            dir.path(),
            "docs.jsonl",
            "{\"_id\":\"a\",\"text\":\"x\"}\n{\"_id\":\"a\",\"text\":\"x\"}\n",
        );
        assert!(prepare(&manifest, dir.path(), dir.path(), "fixture").is_err());
    }
    #[test]
    fn input_order_does_not_change_selection_or_text() {
        let dir = tempfile::tempdir().unwrap();
        let mut manifest = fixture(dir.path());
        manifest.document_count = 4;
        manifest.qrels = input(
            dir.path(),
            "qrels.tsv",
            "query-id\tcorpus-id\tscore\nq\ta\t1\n",
        );
        let before = prepare(&manifest, dir.path(), dir.path(), "fixture").unwrap();
        let text = std::fs::read_to_string(dir.path().join("docs.jsonl")).unwrap();
        let reversed = text.lines().rev().collect::<Vec<_>>().join("\n") + "\n";
        manifest.documents = input(dir.path(), "docs.jsonl", &reversed);
        let after = prepare(&manifest, dir.path(), dir.path(), "fixture").unwrap();
        assert_eq!(
            serde_json::to_value(before.documents).unwrap(),
            serde_json::to_value(after.documents).unwrap()
        );
    }
    #[test]
    fn query_selection_has_a_stable_hash_and_never_skips_for_budget() {
        assert_eq!(
            rank("20260909", "query", "q"),
            "cbfeabe8af465e5180e4e1df954e42dc27030077e86b5f5f3d2d685db2353258"
        );
        assert_eq!(
            rank("20260909", "document", "a"),
            "7c327a6e0d5929c30504421e31e1f7567b8bcfee251f3dedf7820ef665324273"
        );
        let dir = tempfile::tempdir().unwrap();
        let mut manifest = fixture(dir.path());
        manifest.queries = input(
            dir.path(),
            "queries.jsonl",
            "{\"_id\":\"q2\",\"text\":\"second?\"}\n{\"_id\":\"q\",\"text\":\"first?\"}\n",
        );
        manifest.qrels = input(
            dir.path(),
            "qrels.tsv",
            "query-id\tcorpus-id\tscore\nq2\te\t1\nq\ta\t2\nq\tb\t1\nq\tc\t0\n",
        );
        let corpus = prepare(&manifest, dir.path(), dir.path(), "fixture").unwrap();
        assert_eq!(corpus.queries[0].id, "q");
        assert_eq!(corpus.provenance["eligible_queries"], 2);
        manifest.document_count = 1;
        assert!(prepare(&manifest, dir.path(), dir.path(), "fixture").is_err());
        manifest.document_count = 5;
        manifest.query_count = 3;
        assert!(prepare(&manifest, dir.path(), dir.path(), "fixture").is_err());
    }
    #[test]
    fn negative_lists_require_unique_real_nonpositive_candidates() {
        let dir = tempfile::tempdir().unwrap();
        for ids in ["[\"a\",\"b\"]", "[\"d\",\"d\"]", "[\"missing\"]", "[]"] {
            let mut manifest = fixture(dir.path());
            manifest.negatives = Some(input(
                dir.path(),
                "negatives.jsonl",
                &format!("{{\"query_id\":\"q\",\"document_ids\":{ids}}}\n"),
            ));
            assert!(prepare(&manifest, dir.path(), dir.path(), "fixture").is_err());
        }
        let mut manifest = fixture(dir.path());
        manifest.exclude_query_id = true;
        manifest.negatives = Some(input(
            dir.path(),
            "negatives.jsonl",
            "{\"query_id\":\"q\",\"document_ids\":[\"q\",\"d\"]}\n",
        ));
        let corpus = prepare(&manifest, dir.path(), dir.path(), "fixture").unwrap();
        assert!(!corpus.documents.iter().any(|d| d.id == "q"));
        manifest.queries = input(
            dir.path(),
            "queries.jsonl",
            "{\"_id\":\"q\",\"text\":\"first\"}\n{\"_id\":\"q\",\"text\":\"duplicate\"}\n",
        );
        assert!(prepare(&manifest, dir.path(), dir.path(), "fixture").is_err());
    }
    #[test]
    fn line_and_manifest_limits_fail_closed() {
        let dir = tempfile::tempdir().unwrap();
        let mut manifest = fixture(dir.path());
        manifest.revision = "main".into();
        assert!(manifest.validate().is_err());
        let data = "x".repeat(MAX_LINE as usize + 1);
        let source = input(dir.path(), "long", &data);
        assert!(lines(&source, dir.path(), MAX_SMALL_FILE, |_, _| Ok(())).is_err());
        let source = input(dir.path(), "short", "a\nb\n");
        assert!(lines(&source, dir.path(), 3, |_, _| Ok(())).is_err());
    }
}
