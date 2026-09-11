use super::{Runtime, device::Gpu, normalize_dense};
use anyhow::{Context, Result, bail, ensure};
use ort::{
    ep::{CUDA, ExecutionProvider},
    session::{OutputSelector, RunOptions, Session, builder::GraphOptimizationLevel},
    value::Tensor,
};
use std::path::Path;

pub mod cls;
#[cfg(test)]
mod tests;

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Pooling {
    Cls,
    Mean,
}
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct InferenceRecipe {
    /// Explicit token-output override for reviewed qualification exports only.
    /// Production leaves this unset and retains the pinned Arctic output names.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token_output: Option<String>,
    pub dimensions: usize,
    pub pooling: Pooling,
    pub pad_id: i64,
    pub cls_id: i64,
    pub sep_id: i64,
}
impl Default for InferenceRecipe {
    fn default() -> Self {
        Self {
            token_output: None,
            dimensions: 768,
            pooling: Pooling::Cls,
            pad_id: 1,
            cls_id: 0,
            sep_id: 2,
        }
    }
}
pub struct Onnx {
    session: Session,
    cuda: bool,
    recipe: InferenceRecipe,
    cls_output: bool,
    verification: serde_json::Value,
}
fn cuda_provider(gpu: &Gpu, limit: u64) -> CUDA {
    CUDA::default()
        .with_device_id(gpu.index as i32)
        .with_memory_limit(limit as usize)
        .with_tf32(false)
        .with_skip_layer_norm_strict_mode(true)
}
impl Onnx {
    /// Bounded evidence from the startup verification profile, not kernel timing.
    pub fn verification(&self) -> &serde_json::Value {
        &self.verification
    }
    /// Register the provider before downloading FP16, detecting missing CUDA or
    /// cuDNN libraries without spending bandwidth on an unusable artifact.
    pub fn check_cuda(gpu: &Gpu, limit: u64) -> Result<()> {
        let _span = crate::metrics::Span::new("cuda_preflight");
        ort::init().with_name("ree").commit();
        let ep = cuda_provider(gpu, limit);
        ensure!(ep.is_available()?, "CUDA execution provider unavailable");
        Session::builder()?
            .with_execution_providers([ep.build().error_on_failure()])
            .map_err(ort::Error::<()>::from)?;
        Ok(())
    }
    pub fn load(path: &Path, gpu: Option<(&Gpu, u64)>, max_tokens: usize) -> Result<Self> {
        Self::load_for_qualification(path, gpu, max_tokens, InferenceRecipe::default())
    }
    /// Candidate recipes are exposed for the local qualification harness only;
    /// production ingestion always calls load() with the pinned Arctic recipe.
    pub fn load_for_qualification(
        path: &Path,
        gpu: Option<(&Gpu, u64)>,
        max_tokens: usize,
        recipe: InferenceRecipe,
    ) -> Result<Self> {
        Self::load_internal(path, gpu, max_tokens, recipe, false)
    }
    /// Only for checksum-verified arctic-cls-gather-v1 derivatives.
    pub fn load_cls(path: &Path, gpu: Option<(&Gpu, u64)>, max_tokens: usize) -> Result<Self> {
        Self::load_internal(path, gpu, max_tokens, InferenceRecipe::default(), true)
    }
    fn load_internal(
        path: &Path,
        gpu: Option<(&Gpu, u64)>,
        max_tokens: usize,
        recipe: InferenceRecipe,
        cls_output: bool,
    ) -> Result<Self> {
        ensure!(
            (2..=8192).contains(&max_tokens) && (1..=4096).contains(&recipe.dimensions),
            "invalid inference recipe"
        );
        // Core ONNX Runtime is statically linked; CUDA provider libraries load
        // lazily, so their absence cannot prevent CPU startup.
        ort::init().with_name("ree").commit();
        let threads = super::telemetry::inference_threads();
        let mut builder = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(ort::Error::<()>::from)?
            .with_intra_threads(threads)
            .map_err(ort::Error::<()>::from)?;
        let profile = tempfile::tempdir()?;
        if let Some((gpu, limit)) = gpu {
            let ep = cuda_provider(gpu, limit);
            ensure!(
                ep.is_available()?,
                "CUDA execution provider unavailable in ONNX Runtime"
            );
            builder = builder
                .with_execution_providers([ep.build().error_on_failure()])
                .map_err(ort::Error::<()>::from)?
                .with_profiling(profile.path().join("warmup"))
                .map_err(ort::Error::<()>::from)?;
        }
        let session = {
            let _span = crate::metrics::Span::new("session_graph_load");
            builder.commit_from_file(path).context("load ONNX graph")?
        };
        let mut runtime = Self {
            session,
            cuda: gpu.is_some(),
            recipe,
            cls_output,
            verification: serde_json::Value::Null,
        };
        let mut input = vec![42; max_tokens];
        input[0] = runtime.recipe.cls_id;
        input[max_tokens - 1] = runtime.recipe.sep_id;
        {
            let _span = crate::metrics::Span::new("verification_warmup");
            runtime.infer(&[input])?;
        }
        if gpu.is_some() {
            let _span = crate::metrics::Span::new("verification_profile");
            let filename = runtime.session.end_profiling()?;
            let trace: serde_json::Value = serde_json::from_reader(std::fs::File::open(filename)?)?;
            let transfers: Vec<_> = trace.as_array().into_iter().flatten()
                .filter(|e| e["name"].as_str().is_some_and(|n| n.contains("Memcpy")))
                .map(|e| serde_json::json!({"name":e["name"],"provider":e["args"]["provider"],"output_size":e["args"]["output_size"],"output_type_shape":e["args"]["output_type_shape"]})).take(64).collect();
            let cpu_nodes: Vec<_> = trace.as_array().into_iter().flatten()
                .filter(|e| e["args"]["provider"] == "CPUExecutionProvider")
                .map(|e| serde_json::json!({"name":e["name"],"op":e["args"]["op_name"],"input_type_shape":e["args"]["input_type_shape"],"output_type_shape":e["args"]["output_type_shape"]})).take(64).collect();
            runtime.verification = serde_json::json!({"cls_output":cls_output,"transfer_events":transfers,"cpu_nodes":cpu_nodes,"transfer_bytes_measured":false});
            let activated = trace.as_array().is_some_and(|events| {
                events
                    .iter()
                    .any(|e| e["args"]["provider"] == "CUDAExecutionProvider")
            });
            ensure!(
                activated,
                "CUDA provider registered but no graph nodes executed on CUDA"
            );
            if cls_output {
                ensure!(
                    trace
                        .as_array()
                        .is_some_and(|events| events.iter().any(|e| e["name"]
                            .as_str()
                            .is_some_and(|n| n.starts_with("ree/cls-gather-v1"))
                            && e["args"]["provider"] == "CUDAExecutionProvider")),
                    "CLS Gather did not execute on CUDA; refusing a full-tensor host transfer"
                );
            }
        }
        Ok(runtime)
    }
}
impl Runtime for Onnx {
    fn provider(&self) -> &'static str {
        if self.cuda { "cuda" } else { "cpu" }
    }
    fn infer(&mut self, inputs: &[Vec<i64>]) -> Result<Vec<Vec<f32>>> {
        if inputs.is_empty() {
            return Ok(vec![]);
        }
        let batch = inputs.len();
        let length = inputs.iter().map(Vec::len).max().unwrap_or(0);
        ensure!((2..=8192).contains(&length), "invalid sequence length");
        let packing_span = crate::metrics::Span::new("tensor_packing");
        let mut ids = vec![self.recipe.pad_id; batch * length];
        let mut mask = vec![0i64; batch * length];
        for (i, input) in inputs.iter().enumerate() {
            ids[i * length..i * length + input.len()].copy_from_slice(input);
            mask[i * length..i * length + input.len()].fill(1);
        }
        let mut ids = Some(ids);
        let mut mask = Some(mask);
        let mut values = Vec::new();
        for input in self.session.inputs() {
            let data = match input.name() {
                "input_ids" => ids.take().context("duplicate input_ids")?,
                "attention_mask" => mask.take().context("duplicate attention_mask")?,
                "token_type_ids" => vec![0; batch * length],
                name => bail!("unsupported ONNX input {name}"),
            };
            values.push((
                input.name().to_string(),
                Tensor::from_array(([batch, length], data))?,
            ));
        }
        drop(packing_span);
        let output_name = if self.cls_output {
            cls::OUTPUT
        } else if let Some(name) = &self.recipe.token_output {
            name.as_str()
        } else if self
            .session
            .outputs()
            .iter()
            .any(|o| o.name() == "token_embeddings")
        {
            "token_embeddings"
        } else {
            "last_hidden_state"
        };
        let options =
            RunOptions::new()?.with_outputs(OutputSelector::no_default().with(output_name));
        let outputs = {
            let _span = crate::metrics::Span::new("onnx_run_including_transfers");
            self.session.run_with_options(values, &options)?
        };
        let _span = crate::metrics::Span::new("output_pooling");
        let output = outputs
            .get(output_name)
            .context("ONNX embedding output is missing")?;
        if self.cls_output {
            let (shape, data) = output.try_extract_tensor::<f32>()?;
            return pool_cls(shape, data, batch, self.recipe.dimensions);
        }
        if let Ok((shape, data)) = output.try_extract_tensor::<f32>() {
            pool_tokens(shape, data, inputs, length, &self.recipe, |v| v)
        } else {
            let (shape, data) = output.try_extract_tensor::<half::f16>()?;
            pool_tokens(shape, data, inputs, length, &self.recipe, |v| v.to_f32())
        }
    }
}

fn pool_cls(shape: &[i64], data: &[f32], batch: usize, dimensions: usize) -> Result<Vec<Vec<f32>>> {
    ensure!(
        dimensions > 0
            && shape == [batch as i64, dimensions as i64]
            && data.len() == batch * dimensions,
        "unexpected CLS output shape {shape:?}"
    );
    data.chunks_exact(dimensions)
        .map(|slice| {
            let mut vector = slice.to_vec();
            normalize_dense(&mut vector)?;
            Ok(vector)
        })
        .collect()
}

fn pool_tokens<T: Copy>(
    shape: &[i64],
    data: &[T],
    inputs: &[Vec<i64>],
    length: usize,
    recipe: &InferenceRecipe,
    to_f32: impl Fn(T) -> f32,
) -> Result<Vec<Vec<f32>>> {
    let batch = inputs.len();
    let dimensions = recipe.dimensions;
    ensure!(
        shape == [batch as i64, length as i64, dimensions as i64],
        "unexpected ONNX output shape {shape:?}"
    );
    let mut vectors = Vec::with_capacity(batch);
    for (i, input) in inputs.iter().enumerate() {
        let start = i * length * dimensions;
        let mut vector = match recipe.pooling {
            // Read only CLS values from the runtime-owned output. In particular,
            // do not copy/convert the full batch × length × dimensions tensor.
            Pooling::Cls => data[start..start + dimensions]
                .iter()
                .copied()
                .map(&to_f32)
                .collect(),
            Pooling::Mean => {
                let mut v = vec![0.; dimensions];
                for token in 0..input.len() {
                    for (d, value) in v.iter_mut().enumerate() {
                        // Retain per-token division and accumulation order.
                        *value += to_f32(data[start + token * dimensions + d]) / input.len() as f32;
                    }
                }
                v
            }
        };
        normalize_dense(&mut vector)?;
        vectors.push(vector);
    }
    Ok(vectors)
}
