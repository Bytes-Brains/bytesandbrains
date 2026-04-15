use std::fmt;

use bb_core::embedding::{Distance, EmbeddingSpace};

use super::code::{PQCode, bytes_for_nbits};

/// Precomputed distance table for Asymmetric Distance Computation (ADC).
///
/// Given a query vector, this table stores the squared distance from each
/// query subvector to each centroid in that subspace. During search, the
/// distance to an encoded vector is computed by summing table lookups.
///
/// Table layout: `table[m * ksub + k]` is the distance from query subvector
/// `m` to centroid `k` in subspace `m`.
///
/// The const generics must match the ProductQuantizer configuration:
/// - M: number of subquantizers
/// - NBITS: bits per centroid index
pub struct PQDistanceTable<S: EmbeddingSpace, const M: usize, const NBITS: usize>
where
    [(); bytes_for_nbits(NBITS)]:,
{
    table: Vec<S::DistanceValue>,
    ksub: usize,
}

impl<S: EmbeddingSpace, const M: usize, const NBITS: usize> fmt::Debug for PQDistanceTable<S, M, NBITS>
where
    [(); bytes_for_nbits(NBITS)]:,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PQDistanceTable")
            .field("M", &M)
            .field("NBITS", &NBITS)
            .field("ksub", &self.ksub)
            .field("table_len", &self.table.len())
            .finish()
    }
}

impl<S: EmbeddingSpace, const M: usize, const NBITS: usize> PQDistanceTable<S, M, NBITS>
where
    [(); bytes_for_nbits(NBITS)]:,
{
    /// Number of centroids per subspace (2^NBITS)
    pub const KSUB: usize = 1 << NBITS;

    pub fn new(table: Vec<S::DistanceValue>, ksub: usize) -> Self {
        debug_assert_eq!(table.len(), M * ksub);
        Self { table, ksub }
    }

    /// Compute approximate distance to an encoded vector using table lookups.
    ///
    /// This is the core of ADC: instead of computing the full distance,
    /// we sum precomputed partial distances from each subspace.
    pub fn distance(&self, code: &PQCode<M, NBITS>) -> S::DistanceValue {
        (0..M)
            .map(|m| {
                let c = code.get(m) as usize;
                self.table[m * self.ksub + c]
            })
            .fold(S::DistanceValue::zero(), |acc, d| acc + d)
    }

    pub fn m(&self) -> usize {
        M
    }

    pub fn ksub(&self) -> usize {
        self.ksub
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bb_core::embedding::{F32Distance, F32L2Space};

    #[test]
    fn test_distance_table_lookup() {
        // 2 subquantizers, 4 centroids each (NBITS=2)
        let ksub = 4;

        // Table: distances from query subvectors to centroids
        let table: Vec<F32Distance> = vec![
            // Subspace 0: distances to centroids 0,1,2,3
            1.0.into(),
            2.0.into(),
            3.0.into(),
            4.0.into(),
            // Subspace 1: distances to centroids 0,1,2,3
            5.0.into(),
            6.0.into(),
            7.0.into(),
            8.0.into(),
        ];

        let dt = PQDistanceTable::<F32L2Space<4>, 2, 2>::new(table, ksub);

        // Code [0, 2] -> table[0*4 + 0] + table[1*4 + 2] = 1.0 + 7.0 = 8.0
        let code = PQCode::<2, 2>::from_indices(&[0, 2]);
        assert_eq!(dt.distance(&code).value(), 8.0);

        // Code [3, 1] -> table[0*4 + 3] + table[1*4 + 1] = 4.0 + 6.0 = 10.0
        let code = PQCode::<2, 2>::from_indices(&[3, 1]);
        assert_eq!(dt.distance(&code).value(), 10.0);
    }

    #[test]
    fn test_distance_table_nbits8() {
        // 2 subquantizers, 256 centroids each (NBITS=8)
        let ksub = 256;

        // Create a sparse table (all zeros except a few entries)
        let mut table: Vec<F32Distance> = vec![0.0.into(); 2 * 256];
        table[0 * 256 + 100] = 5.0.into();  // subspace 0, centroid 100
        table[1 * 256 + 200] = 3.0.into();  // subspace 1, centroid 200

        let dt = PQDistanceTable::<F32L2Space<4>, 2, 8>::new(table, ksub);

        // Code [100, 200] -> 5.0 + 3.0 = 8.0
        let code = PQCode::<2, 8>::from_indices(&[100, 200]);
        assert_eq!(dt.distance(&code).value(), 8.0);

        // Code [0, 0] -> 0.0 + 0.0 = 0.0
        let code = PQCode::<2, 8>::from_indices(&[0, 0]);
        assert_eq!(dt.distance(&code).value(), 0.0);
    }
}
