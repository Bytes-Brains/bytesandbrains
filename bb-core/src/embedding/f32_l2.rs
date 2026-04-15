use crate::embedding::{Embedding, EmbeddingSpace, F32Distance, F32Embedding};

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct F32L2Space<const L: usize>;

impl<const L: usize> F32L2Space<L> {
    const NAME: &'static str = "F32L2Space";
}

impl<const L: usize> EmbeddingSpace for F32L2Space<L> {
    type EmbeddingData = F32Embedding<L>;
    type DistanceValue = F32Distance;
    type Prepared = F32Embedding<L>;

    fn space_id(&self) -> &'static str {
        F32L2Space::<L>::NAME
    }

    /// Squared Euclidean distance.
    ///
    /// With the `simd` feature: uses simsimd SIMD-accelerated computation.
    /// Without: scalar fallback.
    fn distance(&self, lhs: &Self::EmbeddingData, rhs: &Self::EmbeddingData) -> Self::DistanceValue {
        #[cfg(feature = "simd")]
        {
            use simsimd::SpatialSimilarity;
            let sq_dist = f32::sqeuclidean(lhs.as_slice(), rhs.as_slice())
                .expect("sqeuclidean should not fail for valid slices");
            F32Distance::new(sq_dist as f32)
        }

        #[cfg(not(feature = "simd"))]
        {
            let lhs = lhs.as_slice();
            let rhs = rhs.as_slice();
            let mut sum = 0.0f32;
            for i in 0..L {
                let diff = lhs[i] - rhs[i];
                sum += diff * diff;
            }
            F32Distance::new(sum)
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
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| {
                let diff = x - y;
                diff * diff
            })
            .sum()
    }

    fn length() -> usize {
        L
    }

    fn infinite_mapping(native_distance: &Self::DistanceValue) -> f32 {
        (*native_distance).into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::Embedding;

    #[test]
    fn test_space_properties() {
        let space = F32L2Space::<3>;
        assert_eq!(space.space_id(), "F32L2Space");
        assert_eq!(F32L2Space::<3>::length(), 3);
    }

    #[test]
    fn test_l2_distance_calculation() {
        let space = F32L2Space::<3>;
        let embedding1 = F32Embedding::<3>([1.0, 2.0, 3.0]);
        let embedding2 = F32Embedding::<3>([4.0, 5.0, 6.0]);
        let distance = space.distance(&embedding1, &embedding2);
        // Squared distance: (3^2 + 3^2 + 3^2) = 27
        assert_eq!(distance.value(), 27.0);
    }

    #[test]
    fn test_l2_distance_same_vectors() {
        let space = F32L2Space::<3>;
        let embedding = F32Embedding::<3>([1.0, 2.0, 3.0]);
        let distance = space.distance(&embedding, &embedding);
        assert_eq!(distance.value(), 0.0);
    }

    #[test]
    fn test_l2_distance_zero_vector() {
        let space = F32L2Space::<3>;
        let zero = F32Embedding::<3>::zeros();
        let embedding = F32Embedding::<3>([1.0, 2.0, 3.0]);
        let distance = space.distance(&zero, &embedding);
        // Squared distance: (1^2 + 2^2 + 3^2) = 14
        assert_eq!(distance.value(), 14.0);
    }

    #[test]
    fn test_create_embedding() {
        let data = vec![1.0, 2.0, 3.0];
        let embedding = F32L2Space::<3>::create_embedding(data);
        assert_eq!(embedding.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_zero_vector() {
        let zero = F32L2Space::<3>::zero_vector();
        assert_eq!(zero.as_slice(), &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_zero_distance() {
        let zero_dist = F32L2Space::<3>::zero_distance();
        assert_eq!(zero_dist.value(), 0.0);
    }

    #[test]
    fn test_prepare_and_distance_prepared() {
        let space = F32L2Space::<3>;
        let query = F32Embedding::<3>([1.0, 2.0, 3.0]);
        let target = F32Embedding::<3>([4.0, 5.0, 6.0]);

        let prepared = space.prepare(&query);
        let dist = space.distance_prepared(&prepared, &target);

        // Should match direct distance
        assert_eq!(dist, space.distance(&query, &target));
    }
}
