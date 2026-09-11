use super::*;
use half::f16;

// Historical full-materialization algorithm. Keep its accumulation order and
// normalization separate from the optimized pooling loop as a bitwise oracle.
fn materialized_reference(
    data: Vec<f32>,
    inputs: &[Vec<i64>],
    length: usize,
    recipe: &InferenceRecipe,
) -> Result<Vec<Vec<f32>>> {
    let dimensions = recipe.dimensions;
    let mut vectors = Vec::with_capacity(inputs.len());
    for (i, input) in inputs.iter().enumerate() {
        let start = i * length * dimensions;
        let mut vector = match recipe.pooling {
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

fn assert_bits(actual: &[Vec<f32>], expected: &[Vec<f32>]) {
    let bits = |vectors: &[Vec<f32>]| {
        vectors
            .iter()
            .map(|v| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>())
            .collect::<Vec<_>>()
    };
    assert_eq!(bits(actual), bits(expected));
}

fn check_reference(pooling: Pooling) {
    for (batch, length, dimensions) in [(1, 2, 3), (3, 7, 5), (8, 512, 768)] {
        let recipe = InferenceRecipe {
            dimensions,
            pooling: pooling.clone(),
            ..Default::default()
        };
        let inputs: Vec<_> = (0..batch).map(|i| vec![42; length - i % 3]).collect();
        let shape = [batch as i64, length as i64, dimensions as i64];
        let data: Vec<f32> = (0..batch * length * dimensions)
            .map(|i| ((i * 37 % 1009) as f32 - 504.) / 113.)
            .collect();
        let expected = materialized_reference(data.clone(), &inputs, length, &recipe).unwrap();
        let actual = pool_tokens(&shape, &data, &inputs, length, &recipe, |v| v).unwrap();
        assert_bits(&actual, &expected);

        // Compare against conversion of the entire f16 tensor, not the original
        // f32 values: f16 rounding is part of the qualification contract.
        let data: Vec<_> = data.into_iter().map(f16::from_f32).collect();
        let expected = materialized_reference(
            data.iter().map(|v| v.to_f32()).collect(),
            &inputs,
            length,
            &recipe,
        )
        .unwrap();
        let actual = pool_tokens(&shape, &data, &inputs, length, &recipe, |v| v.to_f32()).unwrap();
        assert_bits(&actual, &expected);
    }
}

#[test]
fn cls_matches_materialized_f32_and_f16_bitwise() {
    check_reference(Pooling::Cls);
}

#[test]
fn mean_matches_materialized_f32_and_f16_bitwise() {
    check_reference(Pooling::Mean);
}

#[test]
fn cls_selects_each_first_token_and_ignores_non_cls_values() {
    let recipe = InferenceRecipe {
        dimensions: 3,
        ..Default::default()
    };
    let inputs = vec![vec![0, 2], vec![0, 42, 2]];
    let mut data = vec![f32::NAN; 2 * 3 * 3];
    data[..3].copy_from_slice(&[3., 4., -0.]);
    data[9..12].copy_from_slice(&[-4., 0., 3.]);
    let expected = vec![vec![0.6, 0.8, -0.], vec![-0.8, 0., 0.6]];
    assert_bits(
        &pool_tokens(&[2, 3, 3], &data, &inputs, 3, &recipe, |v| v).unwrap(),
        &expected,
    );
    let half: Vec<_> = data.into_iter().map(f16::from_f32).collect();
    assert_bits(
        &pool_tokens(&[2, 3, 3], &half, &inputs, 3, &recipe, |v| v.to_f32()).unwrap(),
        &expected,
    );
}

#[test]
fn cls_converts_only_the_retained_values() {
    let recipe = InferenceRecipe {
        dimensions: 3,
        ..Default::default()
    };
    let inputs = vec![vec![0, 2], vec![0, 42, 2]];
    let data = vec![f16::ONE; 2 * 3 * 3];
    let converted = std::cell::Cell::new(0);
    let vectors = pool_tokens(&[2, 3, 3], &data, &inputs, 3, &recipe, |v| {
        converted.set(converted.get() + 1);
        v.to_f32()
    })
    .unwrap();
    assert_eq!(vectors.len(), 2);
    assert_eq!(converted.get(), 2 * 3);
}

#[test]
fn mean_excludes_padding_but_includes_all_input_tokens() {
    let recipe = InferenceRecipe {
        dimensions: 2,
        pooling: Pooling::Mean,
        ..Default::default()
    };
    let inputs = vec![vec![0, 2], vec![0, 42, 2]];
    let data = [6., 0., 0., 8., f32::NAN, f32::NAN, 0., 3., 0., 6., 0., 9.];
    let expected = vec![vec![0.6, 0.8], vec![0., 1.]];
    assert_bits(
        &pool_tokens(&[2, 3, 2], &data, &inputs, 3, &recipe, |v| v).unwrap(),
        &expected,
    );
    let half: Vec<_> = data.into_iter().map(f16::from_f32).collect();
    assert_bits(
        &pool_tokens(&[2, 3, 2], &half, &inputs, 3, &recipe, |v| v.to_f32()).unwrap(),
        &expected,
    );
}

#[test]
fn rejects_wrong_rank_batch_length_or_dimensions_before_indexing() {
    let inputs = vec![vec![0, 2]; 2];
    for pooling in [Pooling::Cls, Pooling::Mean] {
        let recipe = InferenceRecipe {
            dimensions: 3,
            pooling,
            ..Default::default()
        };
        for shape in [
            vec![],
            vec![2, 2],
            vec![2, 2, 3, 1],
            vec![1, 2, 3],
            vec![2, 1, 3],
            vec![2, 2, 4],
            vec![2, 3, 2],
            vec![2, 0, 3],
            vec![2, -2, 3],
        ] {
            let error = pool_tokens(&shape, &[] as &[f32], &inputs, 2, &recipe, |v| v).unwrap_err();
            assert!(
                error
                    .to_string()
                    .starts_with("unexpected ONNX output shape")
            );
            let error = pool_tokens(&shape, &[] as &[f16], &inputs, 2, &recipe, |v| v.to_f32())
                .unwrap_err();
            assert!(
                error
                    .to_string()
                    .starts_with("unexpected ONNX output shape")
            );
        }
    }
}

#[test]
fn rejects_zero_and_non_finite_vectors_without_partial_results() {
    let inputs = vec![vec![0, 2]; 2];
    for pooling in [Pooling::Cls, Pooling::Mean] {
        let recipe = InferenceRecipe {
            dimensions: 2,
            pooling,
            ..Default::default()
        };
        for bad in [0., f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            // The first item is valid; failure in the second must fail the call.
            let data = [3., 4., 3., 4., bad, bad, bad, bad];
            let expected = materialized_reference(data.to_vec(), &inputs, 2, &recipe)
                .unwrap_err()
                .to_string();
            assert_eq!(
                pool_tokens(&[2, 2, 2], &data, &inputs, 2, &recipe, |v| v)
                    .unwrap_err()
                    .to_string(),
                expected,
            );
            let half: Vec<_> = data.into_iter().map(f16::from_f32).collect();
            assert_eq!(
                pool_tokens(&[2, 2, 2], &half, &inputs, 2, &recipe, |v| v.to_f32())
                    .unwrap_err()
                    .to_string(),
                expected,
            );
        }
    }
}

#[test]
fn f16_normalization_preserves_subnormals_extremes_and_signed_zero() {
    let recipe = InferenceRecipe {
        dimensions: 3,
        ..Default::default()
    };
    let inputs = vec![vec![0, 2]];
    for values in [
        [f16::from_bits(1), f16::from_bits(0x8002), f16::NEG_ZERO],
        [f16::MAX, f16::MIN, f16::NEG_ZERO],
    ] {
        let data: Vec<_> = values.into_iter().cycle().take(6).collect();
        let expected = materialized_reference(
            data.iter().map(|v| v.to_f32()).collect(),
            &inputs,
            2,
            &recipe,
        )
        .unwrap();
        assert_bits(
            &pool_tokens(&[1, 2, 3], &data, &inputs, 2, &recipe, |v| v.to_f32()).unwrap(),
            &expected,
        );
    }
}

#[test]
fn two_dimensional_cls_matches_token_pooling_and_rejects_bad_outputs() {
    let recipe = InferenceRecipe::default();
    let inputs = vec![vec![0, 2], vec![0, 42, 2]];
    let data: Vec<f32> = (0..2 * 3 * 768)
        .map(|n| ((n % 193) as f32 - 96.0) / 101.0)
        .collect();
    let selected: Vec<f32> = [0, 3 * 768]
        .into_iter()
        .flat_map(|start| data[start..start + 768].iter().copied())
        .collect();
    let expected = pool_tokens(&[2, 3, 768], &data, &inputs, 3, &recipe, |v| v).unwrap();
    assert_bits(
        &super::pool_cls(&[2, 768], &selected, 2, 768).unwrap(),
        &expected,
    );
    for shape in [vec![2, 3, 768], vec![1, 768], vec![2, 767]] {
        assert!(super::pool_cls(&shape, &selected, 2, 768).is_err());
    }
    assert!(super::pool_cls(&[1, 768], &[0.0; 768], 1, 768).is_err());
    assert!(super::pool_cls(&[1, 768], &[f32::NAN; 768], 1, 768).is_err());
    assert!(super::pool_cls(&[0, 0], &[], 0, 0).is_err());
}

#[test]
fn normalization_preserves_small_large_and_signed_zero_values() {
    for values in [
        [f32::from_bits(1), -f32::from_bits(2), -0.],
        [f32::MAX, -f32::MAX / 2., 0.],
        [f16::from_bits(1).to_f32(), f16::MAX.to_f32(), -0.],
    ] {
        let recipe = InferenceRecipe {
            dimensions: 3,
            ..Default::default()
        };
        let inputs = vec![vec![0, 2]];
        let data: Vec<_> = values.into_iter().cycle().take(6).collect();
        let expected = materialized_reference(data.clone(), &inputs, 2, &recipe).unwrap();
        assert_bits(
            &pool_tokens(&[1, 2, 3], &data, &inputs, 2, &recipe, |v| v).unwrap(),
            &expected,
        );
    }
}
