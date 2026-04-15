use crate::embedding::{Embedding, EmbeddingSpace, F32Distance, F32Embedding};

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct F32CosineSpace<const L: usize>;

impl<const L: usize> F32CosineSpace<L> {
    const NAME: &'static str = "F32CosineSpace";
}

impl<const L: usize> EmbeddingSpace for F32CosineSpace<L> {
    type EmbeddingData = F32Embedding<L>;
    type DistanceValue = F32Distance;
    type Prepared = F32Embedding<L>;

    fn space_id(&self) -> &'static str {
        F32CosineSpace::<L>::NAME
    }

    /// Cosine distance = 1 - cosine_similarity, ranging from 0 (identical) to 2 (opposite).
    ///
    /// With the `simd` feature: uses simsimd SIMD-accelerated computation.
    /// Without: scalar fallback.
    fn distance(&self, lhs: &Self::EmbeddingData, rhs: &Self::EmbeddingData) -> Self::DistanceValue {
        #[cfg(feature = "simd")]
        {
            use simsimd::SpatialSimilarity;
            let cos_dist = f32::cosine(lhs.as_slice(), rhs.as_slice())
                .expect("cosine should not fail for valid slices");
            F32Distance::new(cos_dist as f32)
        }

        #[cfg(not(feature = "simd"))]
        {
            let lhs = lhs.as_slice();
            let rhs = rhs.as_slice();
            let mut dot = 0.0f32;
            let mut norm_lhs = 0.0f32;
            let mut norm_rhs = 0.0f32;
            for i in 0..L {
                dot += lhs[i] * rhs[i];
                norm_lhs += lhs[i] * lhs[i];
                norm_rhs += rhs[i] * rhs[i];
            }
            let denom = (norm_lhs * norm_rhs).sqrt();
            if denom == 0.0 {
                // Both zero vectors — define distance as 0
                F32Distance::new(0.0)
            } else {
                F32Distance::new(1.0 - dot / denom)
            }
        }
    }

    fn prepare(&self, embedding: &Self::EmbeddingData) -> Self::Prepared {
        embedding.clone()
    }

    fn distance_prepared(
        &self,
        prepared: &Self::Prepared,
        target: &Self::EmbeddingData,
    ) -> Self::DistanceValue {
        self.distance(prepared, target)
    }

    fn slice_distance(a: &[f32], b: &[f32]) -> f32 {
        let mut dot = 0.0f32;
        let mut norm_a = 0.0f32;
        let mut norm_b = 0.0f32;
        for (x, y) in a.iter().zip(b.iter()) {
            dot += x * y;
            norm_a += x * x;
            norm_b += y * y;
        }
        let denom = (norm_a * norm_b).sqrt();
        if denom == 0.0 {
            0.0
        } else {
            1.0 - dot / denom
        }
    }

    fn length() -> usize {
        L
    }

    /// Maps the bounded cosine distance range [0, 2] to an unbounded [0, +inf) range
    /// using tan(pi * d / 4). This is used by protocols that require infinite-range
    /// distances (e.g., for greedy routing where distance ratios matter).
    fn infinite_mapping(native_distance: &Self::DistanceValue) -> f32 {
        let d: f32 = (*native_distance).into();
        (std::f32::consts::PI * d / 4.0).tan()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_space_properties() {
        let space = F32CosineSpace::<3>;
        assert_eq!(space.space_id(), "F32CosineSpace");
        assert_eq!(F32CosineSpace::<3>::length(), 3);
    }

    #[test]
    fn test_cosine_distance_same_vectors() {
        let space = F32CosineSpace::<3>;
        let embedding = F32Embedding::<3>([1.0, 2.0, 3.0]);
        let distance = space.distance(&embedding, &embedding);
        assert!(distance.value().abs() < 1e-5);
    }

    #[test]
    fn test_cosine_distance_orthogonal() {
        let space = F32CosineSpace::<2>;
        let embedding1 = F32Embedding::<2>([1.0, 0.0]);
        let embedding2 = F32Embedding::<2>([0.0, 1.0]);
        let distance = space.distance(&embedding1, &embedding2);
        assert!((distance.value() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_distance_opposite() {
        let space = F32CosineSpace::<3>;
        let embedding1 = F32Embedding::<3>([1.0, 2.0, 3.0]);
        let embedding2 = F32Embedding::<3>([-1.0, -2.0, -3.0]);
        let distance = space.distance(&embedding1, &embedding2);
        assert!((distance.value() - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_distance_parallel() {
        let space = F32CosineSpace::<3>;
        let embedding1 = F32Embedding::<3>([1.0, 2.0, 3.0]);
        let embedding2 = F32Embedding::<3>([2.0, 4.0, 6.0]);
        let distance = space.distance(&embedding1, &embedding2);
        assert!(distance.value().abs() < 1e-5);
    }

    #[test]
    fn test_infinite_mapping() {
        let zero = F32Distance::new(0.0);
        assert!(F32CosineSpace::<3>::infinite_mapping(&zero).abs() < 1e-5);

        let one = F32Distance::new(1.0);
        assert!((F32CosineSpace::<3>::infinite_mapping(&one) - 1.0).abs() < 1e-5);

        let half = F32Distance::new(0.5);
        let mapped_half = F32CosineSpace::<3>::infinite_mapping(&half);
        let mapped_one = F32CosineSpace::<3>::infinite_mapping(&one);
        assert!(mapped_half < mapped_one);
    }

    #[test]
    fn test_prepare_and_distance_prepared() {
        let space = F32CosineSpace::<3>;
        let query = F32Embedding::<3>([1.0, 2.0, 3.0]);
        let target = F32Embedding::<3>([4.0, 5.0, 6.0]);

        let prepared = space.prepare(&query);
        let dist = space.distance_prepared(&prepared, &target);

        // Should match direct distance
        assert_eq!(dist, space.distance(&query, &target));
    }
}
