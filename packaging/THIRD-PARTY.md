# Third-party material in the local beta

This is a development-host beta, not a declaration of a completed distribution
license audit. No NVIDIA CUDA toolkit/driver/cuDNN binaries or model weights are
included. Only ree and the ONNX Runtime shared/CUDA providers are installed as
executables/libraries; dependencies of those providers remain system-managed.

- `upstream/onnxruntime/` contains Microsoft's ONNX Runtime 1.28.0 MIT license and
  complete upstream third-party notice file. `upstream/provenance.json` pins the
  retrieved notices by source revision and SHA-256.
- `dependencies.json` records Cargo dependency versions, declared licenses,
  authors, source locations, archive checksums, and collected notice paths.
  It is a conservative superset including build/test dependencies, not a claim
  that every listed dependency is linked into the production executable.
- `crates/` preserves license/copyright/notice files available in the resolved
  local registry source trees, including nested vendored-component notices.
- `../sources/crates/` includes the complete, unmodified `.crate` source archives,
  checked against Cargo.lock. This includes MPL-2.0 selectors source and embedded
  native code. The MPL text is also in `upstream/MPL-2.0.txt`.
- Some upstream crates omit standalone licenses from published archives.
  Supplemental pinned upstream notices for match_token and sqlite-vec and a
  revision-identified difflib notice are included under `upstream/`. The latter
  is supplemental, not proof of the old crate's original VCS revision.
  fxhash and mac declare Apache-2.0/MIT in their published metadata but have no
  standalone upstream notice file in the inspected revisions; their complete
  source/README/attributions and declared license metadata are preserved. The
  Apache-2.0 text is also supplied at the package root.
- `../sources/ree/` preserves ree's source snapshot and locked dependency recipe.
  ree is MIT OR Apache-2.0; both licenses are at the package root.

The separate pinned Arctic model is downloaded only on first model use; its
recipe/revision and declared Apache-2.0 license are documented in
`../docs/model-decision-v1.md`. Installation performs no model download.

Upstream notices can mention optional components not used by this build. A
public redistributable release still requires reviewing the exact native build
composition, notices and source obligations; including a notice inventory is
not a substitute for that review. No third-party license is changed or waived.
