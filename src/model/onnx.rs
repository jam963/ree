use super::{Runtime, device::Gpu, normalize_dense};
use anyhow::{Context, Result, bail, ensure};
use ort::{
    ep::{CUDA, ExecutionProvider},
    session::{Session, builder::GraphOptimizationLevel},
    value::Tensor,
};
use std::path::Path;

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Pooling {
    Cls,
    Mean,
}
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct InferenceRecipe {
    pub dimensions: usize,
    pub pooling: Pooling,
    pub pad_id: i64,
    pub cls_id: i64,
    pub sep_id: i64,
}
impl Default for InferenceRecipe {
    fn default() -> Self {
        Self {
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
}
fn cuda_provider(gpu: &Gpu, limit: u64) -> CUDA {
    CUDA::default()
        .with_device_id(gpu.index as i32)
        .with_memory_limit(limit as usize)
        .with_tf32(false)
        .with_skip_layer_norm_strict_mode(true)
}
impl Onnx {
    /// Register the provider before downloading FP16, detecting missing CUDA or
    /// cuDNN libraries without spending bandwidth on an unusable artifact.
    pub fn check_cuda(gpu: &Gpu, limit: u64) -> Result<()> {
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
        ensure!(
            (2..=8192).contains(&max_tokens) && (1..=4096).contains(&recipe.dimensions),
            "invalid inference recipe"
        );
        // Core ONNX Runtime is statically linked; CUDA provider libraries load
        // lazily, so their absence cannot prevent CPU startup.
        ort::init().with_name("ree").commit();
        let threads = std::thread::available_parallelism()
            .map_or(4, |n| n.get())
            .saturating_sub(2)
            .clamp(1, 8);
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
        let mut runtime = Self {
            session: builder.commit_from_file(path).context("load ONNX graph")?,
            cuda: gpu.is_some(),
            recipe,
        };
        let mut input = vec![42; max_tokens];
        input[0] = runtime.recipe.cls_id;
        input[max_tokens - 1] = runtime.recipe.sep_id;
        runtime.infer(&[input])?;
        if gpu.is_some() {
            let filename = runtime.session.end_profiling()?;
            let trace: serde_json::Value = serde_json::from_reader(std::fs::File::open(filename)?)?;
            let activated = trace.as_array().is_some_and(|events| {
                events
                    .iter()
                    .any(|e| e["args"]["provider"] == "CUDAExecutionProvider")
            });
            ensure!(
                activated,
                "CUDA provider registered but no graph nodes executed on CUDA"
            );
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
        let mut ids = vec![self.recipe.pad_id; batch * length];
        let mut mask = vec![0i64; batch * length];
        for (i, input) in inputs.iter().enumerate() {
            ids[i * length..i * length + input.len()].copy_from_slice(input);
            mask[i * length..i * length + input.len()].fill(1);
        }
        let mut values = Vec::new();
        for input in self.session.inputs() {
            let data = match input.name() {
                "input_ids" => ids.clone(),
                "attention_mask" => mask.clone(),
                "token_type_ids" => vec![0; batch * length],
                name => bail!("unsupported ONNX input {name}"),
            };
            values.push((
                input.name().to_string(),
                Tensor::from_array(([batch, length], data))?,
            ));
        }
        let outputs = self.session.run(values)?;
        let output = outputs
            .get("token_embeddings")
            .or_else(|| outputs.get("last_hidden_state"))
            .context("ONNX token embedding output is missing")?;
        let (shape, data) = if let Ok((shape, data)) = output.try_extract_tensor::<f32>() {
            (shape.to_vec(), data.to_vec())
        } else {
            let (shape, data) = output.try_extract_tensor::<half::f16>()?;
            (shape.to_vec(), data.iter().map(|v| v.to_f32()).collect())
        };
        let dimensions = self.recipe.dimensions;
        ensure!(
            shape == vec![batch as i64, length as i64, dimensions as i64],
            "unexpected ONNX output shape {shape:?}"
        );
        let mut vectors = Vec::with_capacity(batch);
        for (i, input) in inputs.iter().enumerate() {
            let start = i * length * dimensions;
            let mut vector = match self.recipe.pooling {
                Pooling::Cls => data[start..start + dimensions].to_vec(),
                Pooling::Mean => {
                    let mut v = vec![0.; dimensions];
                    for token in 0..input.len() {
                        for (d, value) in v.iter_mut().enumerate() {
                            *value += data[start + token * dimensions + d] / input.len() as f32;
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
}
