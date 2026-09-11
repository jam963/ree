//! Ordinary subprocess tests: empty real-schema DBs, no tokenizer/model/GPU.
use ree::storage::Database;
use serde_json::{Value, json};
use std::{
    io::{BufRead, BufReader, Write},
    os::unix::{fs::PermissionsExt, net::UnixStream},
    path::PathBuf,
    process::{Child, Command, Stdio},
    time::{Duration, Instant},
};

struct Fixture {
    temp: tempfile::TempDir,
    db: PathBuf,
    socket: PathBuf,
}
impl Fixture {
    fn new() -> Self {
        let temp = tempfile::tempdir().unwrap();
        std::fs::set_permissions(temp.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
        let db = temp.path().join("ree.db");
        Database::open(&db, true).unwrap().register_model().unwrap();
        let socket = temp.path().join("worker.sock");
        Self { temp, db, socket }
    }
    fn command(&self) -> Command {
        let mut c = Command::new(env!("CARGO_BIN_EXE_ree"));
        for name in [
            "REE_DB",
            "REE_CONFIG",
            "REE_DEVICE",
            "REE_BATCH_SIZE",
            "REE_GPU_MEMORY_FRACTION",
            "REE_TIMING",
        ] {
            c.env_remove(name);
        }
        for name in ["XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME"] {
            c.env(name, self.temp.path());
        }
        c
    }
    fn stream(&self, input: &[u8]) -> std::process::Output {
        let mut c = self
            .command()
            .arg("--db")
            .arg(&self.db)
            .args(["search", "--stream"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        c.stdin.take().unwrap().write_all(input).unwrap();
        c.wait_with_output().unwrap()
    }
    fn worker(&self) -> Guard {
        let c = self
            .command()
            .arg("--db")
            .arg(&self.db)
            .args(["worker", "--idle-timeout", "1s", "--socket"])
            .arg(&self.socket)
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let mut g = Guard(c);
        let start = Instant::now();
        while UnixStream::connect(&self.socket).is_err() {
            assert!(g.0.try_wait().unwrap().is_none(), "worker startup failed");
            assert!(start.elapsed() < Duration::from_secs(5));
            std::thread::sleep(Duration::from_millis(10));
        }
        g
    }
    fn connect(&self) -> BufReader<UnixStream> {
        let s = UnixStream::connect(&self.socket).unwrap();
        s.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        BufReader::new(s)
    }
}
struct Guard(Child);
impl Drop for Guard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
fn frame(id: &str, mode: &str) -> String {
    json!({"v":1,"id":id,"op":"search","query":"hello","mode":mode}).to_string() + "\n"
}
fn events(out: &std::process::Output) -> Vec<Value> {
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}
fn receive(s: &mut BufReader<UnixStream>) -> Value {
    let mut line = String::new();
    assert!(s.read_line(&mut line).unwrap() > 0, "unexpected disconnect");
    serde_json::from_str(&line).unwrap()
}
#[test]
fn streaming_reuses_process_drains_eof_and_recovers_request_errors() {
    let f = Fixture::new();
    let input = frame("a", "semantic") + &frame("bad", "invalid") + &frame("b", "lexical");
    // An invalid enum is malformed framing, so closes the stream rather than
    // silently accepting unknown syntax. Accepted earlier work still drains.
    let out = f.stream(input.as_bytes());
    assert_eq!(
        out.status.code(),
        Some(2),
        "{}",
        String::from_utf8_lossy(&out.stdout)
    );
    let input = frame("a", "semantic")
        + &json!({"v":1,"id":"bad","op":"search","query":"","mode":"lexical"}).to_string()
        + "\n"
        + &frame("b", "lexical");
    let out = f.stream(input.as_bytes());
    assert_eq!(
        out.status.code(),
        Some(1),
        "{}",
        String::from_utf8_lossy(&out.stdout)
    );
    let e = events(&out);
    let a = e
        .iter()
        .find(|e| e["id"] == "a" && e["type"] == "search_completed")
        .unwrap();
    let b = e
        .iter()
        .find(|e| e["id"] == "b" && e["type"] == "search_completed")
        .unwrap();
    assert_eq!(a["engine_pid"], b["engine_pid"]);
    assert_eq!(
        e.iter()
            .filter(|e| e["id"] == "bad" && e["type"] == "request_failed")
            .count(),
        1
    );
    assert!(!f.temp.path().join("ree/models").exists());
}
#[test]
fn framing_bounds_quiet_and_conflicts_are_actionable() {
    let f = Fixture::new();
    for bytes in [&b"{}"[..], &b"not json\n"[..], &vec![b'x'; 128 * 1024 + 1]] {
        let out = f.stream(bytes);
        assert_eq!(
            out.status.code(),
            Some(2),
            "{}",
            String::from_utf8_lossy(&out.stdout)
        );
    }
    for args in [
        vec!["search", "--stream", "--quiet"],
        vec!["search", "--stream", "hello"],
        vec!["search", "--stream", "--mode", "lexical"],
    ] {
        let out = f.command().args(args).output().unwrap();
        assert_eq!(out.status.code(), Some(2));
    }
}
#[test]
fn socket_idle_releases_child_but_keeps_endpoint_and_cli_works() {
    let f = Fixture::new();
    let _worker = f.worker();
    assert_eq!(
        std::fs::metadata(&f.socket).unwrap().permissions().mode() & 0o777,
        0o600
    );
    let mut s = f.connect();
    s.get_mut()
        .write_all(frame("a", "semantic").as_bytes())
        .unwrap();
    let a = receive(&mut s);
    assert_eq!(a["type"], "search_completed");
    let pid = a["engine_pid"].as_u64().unwrap();
    s.get_mut()
        .write_all(frame("b", "lexical").as_bytes())
        .unwrap();
    assert_eq!(receive(&mut s)["engine_pid"], pid);
    let start = Instant::now();
    while std::path::Path::new(&format!("/proc/{pid}")).exists() {
        assert!(
            start.elapsed() < Duration::from_secs(5),
            "idle engine survived"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
    s.get_mut()
        .write_all(frame("c", "lexical").as_bytes())
        .unwrap();
    assert_ne!(receive(&mut s)["engine_pid"], pid);
    let out = f
        .command()
        .args(["search", "hello", "--socket"])
        .arg(&f.socket)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stdout)
    );
    assert!(events(&out).iter().any(|e| e["type"] == "search_completed"));
    assert!(!f.temp.path().join("ree/models").exists());
}
#[test]
fn queued_cancellation_fragmentation_and_duplicate_ids() {
    let f = Fixture::new();
    let _worker = f.worker();
    let mut s = f.connect();
    let input = frame("a", "lexical")
        + &frame("b", "lexical")
        + &json!({"v":1,"id":"cancel","op":"cancel","target":"b"}).to_string()
        + "\n";
    // One write puts cancel in the same admission round as its queued target.
    s.get_mut().write_all(input.as_bytes()).unwrap();
    let mut e = vec![];
    for _ in 0..3 {
        e.push(receive(&mut s));
    }
    assert_eq!(
        e.iter()
            .filter(|e| e["id"] == "b"
                && e["type"] == "request_failed"
                && e["error_code"] == "cancelled")
            .count(),
        1
    );
    assert!(
        e.iter()
            .any(|e| e["id"] == "a" && e["type"] == "search_completed")
    );
    assert!(
        e.iter()
            .any(|e| e["id"] == "cancel" && e["type"] == "cancel_completed")
    );
    let f1 = frame("frag", "lexical");
    for bytes in f1.as_bytes().chunks(3) {
        s.get_mut().write_all(bytes).unwrap();
    }
    assert_eq!(receive(&mut s)["id"], "frag");
    let mut s = f.connect();
    s.get_mut()
        .write_all((frame("same", "lexical") + &frame("same", "lexical")).as_bytes())
        .unwrap();
    let mut e = [receive(&mut s), receive(&mut s)];
    e.sort_by_key(|e| e["type"].to_string());
    assert_eq!(
        e.iter().filter(|e| e["type"] == "search_completed").count(),
        1
    );
    assert!(e.iter().any(|e| e["type"] == "connection_failed"));
}
#[test]
fn stdin_cancellation_is_read_even_with_a_full_search_queue() {
    use std::os::fd::AsRawFd;
    let f = Fixture::new();
    let lock = rusqlite::Connection::open(&f.db).unwrap();
    lock.execute_batch("PRAGMA journal_mode=DELETE; BEGIN EXCLUSIVE;")
        .unwrap();
    let process = f
        .command()
        .arg("--db")
        .arg(&f.db)
        .args(["search", "--stream"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut process = Guard(process);
    let mut input = process.0.stdin.take().unwrap();
    let mut output = BufReader::new(process.0.stdout.take().unwrap());
    input
        .write_all(
            (frame("a", "lexical") + &frame("b", "lexical") + &frame("c", "lexical")).as_bytes(),
        )
        .unwrap();
    let children = format!("/proc/{}/task/{}/children", process.0.id(), process.0.id());
    let start = Instant::now();
    while std::fs::read_to_string(&children)
        .unwrap()
        .trim()
        .is_empty()
    {
        assert!(start.elapsed() < Duration::from_secs(2));
        std::thread::sleep(Duration::from_millis(1));
    }
    std::thread::sleep(Duration::from_millis(20));
    input
        .write_all(b"{\"v\":1,\"id\":\"cancel\",\"op\":\"cancel\",\"target\":\"a\"}\n")
        .unwrap();
    let mut replies = vec![];
    for _ in 0..2 {
        if output.buffer().is_empty() {
            let mut fd = libc::pollfd {
                fd: output.get_ref().as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            };
            assert!(
                unsafe { libc::poll(&mut fd, 1, 500) } > 0,
                "cancel was blocked behind queued searches"
            );
        }
        let mut line = String::new();
        output.read_line(&mut line).unwrap();
        replies.push(serde_json::from_str::<Value>(&line).unwrap());
    }
    assert!(
        replies
            .iter()
            .any(|e| e["id"] == "a" && e["error_code"] == "cancelled")
    );
    assert!(
        replies
            .iter()
            .any(|e| e["id"] == "cancel" && e["type"] == "cancel_completed")
    );
    lock.execute_batch("ROLLBACK").unwrap();
    drop(input);
    for id in ["b", "c"] {
        let mut line = String::new();
        output.read_line(&mut line).unwrap();
        let e: Value = serde_json::from_str(&line).unwrap();
        assert_eq!(e["id"], id);
        assert_eq!(e["type"], "search_completed");
    }
    assert_eq!(process.0.wait().unwrap().code(), Some(1));
}

#[test]
fn socket_queue_is_bounded_and_a_crashed_engine_does_not_poison_the_worker() {
    let f = Fixture::new();
    let _worker = f.worker();
    let mut s = f.connect();
    let input: String = (0..40)
        .map(|i| frame(&format!("q{i}"), "lexical"))
        .collect();
    s.get_mut().write_all(input.as_bytes()).unwrap();
    let mut ids = std::collections::HashSet::new();
    let mut busy = 0;
    let mut pid = None;
    for _ in 0..40 {
        let e = receive(&mut s);
        assert!(
            ids.insert(e["id"].as_str().unwrap().to_owned()),
            "duplicate terminal event"
        );
        if e["type"] == "search_completed" {
            pid = e["engine_pid"].as_u64();
        } else {
            assert_eq!(e["error_code"], "busy");
            busy += 1;
        }
    }
    assert!(busy > 0);
    unsafe {
        libc::kill(pid.unwrap() as i32, libc::SIGKILL);
    }
    std::thread::sleep(Duration::from_millis(20));
    s.get_mut()
        .write_all(frame("crash", "lexical").as_bytes())
        .unwrap();
    assert_eq!(receive(&mut s)["type"], "request_failed");
    s.get_mut()
        .write_all(frame("recovered", "lexical").as_bytes())
        .unwrap();
    assert_eq!(receive(&mut s)["type"], "search_completed");
}

#[test]
fn socket_safety_live_stale_and_signal_cleanup() {
    let f = Fixture::new();
    std::fs::write(&f.socket, "do not delete").unwrap();
    let out = f
        .command()
        .args(["worker", "--socket"])
        .arg(&f.socket)
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert_eq!(std::fs::read_to_string(&f.socket).unwrap(), "do not delete");
    std::fs::remove_file(&f.socket).unwrap();
    drop(std::os::unix::net::UnixListener::bind(&f.socket).unwrap());
    let mut worker = f.worker();
    let out = f
        .command()
        .args(["worker", "--socket"])
        .arg(&f.socket)
        .output()
        .unwrap();
    assert!(!out.status.success());
    unsafe {
        libc::kill(worker.0.id() as i32, libc::SIGTERM);
    }
    let start = Instant::now();
    while worker.0.try_wait().unwrap().is_none() {
        assert!(start.elapsed() < Duration::from_secs(5));
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(!f.socket.exists());
}
