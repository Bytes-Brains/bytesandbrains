use super::code::{PQCode, bytes_for_nbits};

/// Precomputed centroid-to-centroid distances for Symmetric Distance Computation (SDC).
///
/// SDC enables fast distance computation between two PQ codes without needing
/// the original vectors. During quantizer training, we precompute squared L2
/// distances between all pairs of centroids in each subspace.
///
/// Table layout: `table[m * ksub * ksub + i * ksub + j]` is the squared distance
/// between centroid i and centroid j in subspace m.
///
/// The const generics must match the ProductQuantizer configuration:
/// - M: number of subquantizers
/// - NBITS: bits per centroid index
#[derive(Clone, Debug, PartialEq)]
pub struct SDCTable<const M: usize, const NBITS: usize>
where
    [(); bytes_for_nbits(NBITS)]:,
{
    /// Flat storage: M * ksub * ksub entries
    table: Vec<f32>,
    /// Number of centroids per subspace (2^NBITS)
    ksub: usize,
}

impl<const M: usize, const NBITS: usize> SDCTable<M, NBITS>
where
    [(); bytes_for_nbits(NBITS)]:,
{
    /// Number of centroids per subspace (2^NBITS)
    pub const KSUB: usize = 1 << NBITS;

    /// Create an SDCTable from centroids using squared L2 distance.
    ///
    /// # Arguments
    /// * `centroids` - Flat centroid storage: M * ksub * dsub floats
    /// * `dsub` - Dimension of each subspace
    pub fn from_centroids(centroids: &[f32], dsub: usize) -> Self {
        Self::from_centroids_with_distance(centroids, dsub, |a, b| {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| {
                    let diff = x - y;
                    diff * diff
                })
                .sum()
        })
    }

    /// Create an SDCTable from centroids using a custom subspace distance function.
    ///
    /// The distance function receives two subvector slices of length `dsub`
    /// and returns a non-negative distance value.
    ///
    /// # Arguments
    /// * `centroids` - Flat centroid storage: M * ksub * dsub floats
    /// * `dsub` - Dimension of each subspace
    /// * `subspace_distance` - Distance function for subvector pairs
    pub fn from_centroids_with_distance(
        centroids: &[f32],
        dsub: usize,
        subspace_distance: impl Fn(&[f32], &[f32]) -> f32,
    ) -> Self {
        let ksub = Self::KSUB;
        assert_eq!(
            centroids.len(),
            M * ksub * dsub,
            "centroids length mismatch"
        );

        let mut table = vec![0.0f32; M * ksub * ksub];

        for m in 0..M {
            for i in 0..ksub {
                let ci_offset = (m * ksub + i) * dsub;
                let ci = &centroids[ci_offset..ci_offset + dsub];
                for j in 0..ksub {
                    let cj_offset = (m * ksub + j) * dsub;
                    let cj = &centroids[cj_offset..cj_offset + dsub];

                    table[m * ksub * ksub + i * ksub + j] = subspace_distance(ci, cj);
                }
            }
        }

        Self { table, ksub }
    }

    /// Compute approximate squared L2 distance between two PQ codes.
    ///
    /// This is the core of SDC: sum precomputed centroid-to-centroid distances
    /// for each subspace.
    #[inline]
    pub fn distance(&self, code1: &PQCode<M, NBITS>, code2: &PQCode<M, NBITS>) -> f32 {
        let mut sum = 0.0f32;
        for m in 0..M {
            let i = code1.get(m) as usize;
            let j = code2.get(m) as usize;
            sum += self.table[m * self.ksub * self.ksub + i * self.ksub + j];
        }
        sum
    }

    /// Number of centroids per subspace.
    pub fn ksub(&self) -> usize {
        self.ksub
    }

    /// Access the raw table data for serialization.
    pub fn table_data(&self) -> &[f32] {
        &self.table
    }

    /// Reconstruct an SDCTable from raw data.
    pub fn from_raw(table: Vec<f32>, ksub: usize) -> Self {
        Self { table, ksub }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sdc_same_codes() {
        // 2 subspaces, 4 centroids each (NBITS=2), 2-dim subspace
        let centroids: Vec<f32> = vec![
            // Subspace 0: 4 centroids of dim 2
            0.0, 0.0, // centroid 0
            1.0, 0.0, // centroid 1
            0.0, 1.0, // centroid 2
            1.0, 1.0, // centroid 3
            // Subspace 1: 4 centroids of dim 2
            0.0, 0.0, // centroid 0
            2.0, 0.0, // centroid 1
            0.0, 2.0, // centroid 2
            2.0, 2.0, // centroid 3
        ];

        let sdc = SDCTable::<2, 2>::from_centroids(&centroids, 2);

        // Same codes should have distance 0
        let code1 = PQCode::<2, 2>::from_indices(&[0, 0]);
        let code2 = PQCode::<2, 2>::from_indices(&[0, 0]);
        assert_eq!(sdc.distance(&code1, &code2), 0.0);

        let code3 = PQCode::<2, 2>::from_indices(&[3, 3]);
        let code4 = PQCode::<2, 2>::from_indices(&[3, 3]);
        assert_eq!(sdc.distance(&code3, &code4), 0.0);
    }

    #[test]
    fn test_sdc_different_codes() {
        // 2 subspaces, 4 centroids each (NBITS=2), 2-dim subspace
        let centroids: Vec<f32> = vec![
            // Subspace 0: 4 centroids of dim 2
            0.0, 0.0, // centroid 0
            1.0, 0.0, // centroid 1
            0.0, 1.0, // centroid 2
            1.0, 1.0, // centroid 3
            // Subspace 1: 4 centroids of dim 2
            0.0, 0.0, // centroid 0
            2.0, 0.0, // centroid 1
            0.0, 2.0, // centroid 2
            2.0, 2.0, // centroid 3
        ];

        let sdc = SDCTable::<2, 2>::from_centroids(&centroids, 2);

        // code1=[0,0] -> centroids (0,0) and (0,0)
        // code2=[1,1] -> centroids (1,0) and (2,0)
        // Distance in subspace 0: ||(0,0) - (1,0)||^2 = 1
        // Distance in subspace 1: ||(0,0) - (2,0)||^2 = 4
        // Total: 5
        let code1 = PQCode::<2, 2>::from_indices(&[0, 0]);
        let code2 = PQCode::<2, 2>::from_indices(&[1, 1]);
        assert_eq!(sdc.distance(&code1, &code2), 5.0);
    }

    #[test]
    fn test_sdc_symmetry() {
        let centroids: Vec<f32> = vec![
            0.0, 0.0,
            1.0, 1.0,
            2.0, 2.0,
            3.0, 3.0,
            0.0, 0.0,
            1.0, 1.0,
            2.0, 2.0,
            3.0, 3.0,
        ];

        let sdc = SDCTable::<2, 2>::from_centroids(&centroids, 2);

        let code1 = PQCode::<2, 2>::from_indices(&[0, 1]);
        let code2 = PQCode::<2, 2>::from_indices(&[2, 3]);

        // Distance should be symmetric
        assert_eq!(sdc.distance(&code1, &code2), sdc.distance(&code2, &code1));
    }

    #[test]
    fn test_sdc_nbits8() {
        // Test with nbits=8 (256 centroids) - just verify it compiles and works
        // Using only 4 centroids for test simplicity
        let centroids: Vec<f32> = vec![0.0; 2 * 256 * 2]; // M=2, ksub=256, dsub=2

        let sdc = SDCTable::<2, 8>::from_centroids(&centroids, 2);

        let code1 = PQCode::<2, 8>::from_indices(&[0, 0]);
        let code2 = PQCode::<2, 8>::from_indices(&[255, 255]);

        // All centroids are zero, so distance should be 0
        assert_eq!(sdc.distance(&code1, &code2), 0.0);
    }
}
