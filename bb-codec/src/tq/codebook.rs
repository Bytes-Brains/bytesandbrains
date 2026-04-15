/// Precomputed Lloyd-Max optimal scalar quantizer for N(0,1).
///
/// These centroids minimize MSE for the standard normal distribution.
/// At runtime they are scaled by 1/√D to match the marginal distribution
/// of a randomly-rotated unit-sphere vector (Paper Section 3.1, Eq. 4).
///
/// Reference: Lloyd (1982), Max (1960).

// ── b = 1 (2 levels) ────────────────────────────────────────────────────────
static BOUNDARIES_1: [f32; 1] = [0.0];
static CENTROIDS_1: [f32; 2] = [-0.7978845608, 0.7978845608];

// ── b = 2 (4 levels) ────────────────────────────────────────────────────────
static BOUNDARIES_2: [f32; 3] = [-0.9815515773, 0.0, 0.9815515773];
static CENTROIDS_2: [f32; 4] = [-1.5104176087, -0.4527800399, 0.4527800399, 1.5104176087];

// ── b = 3 (8 levels) ────────────────────────────────────────────────────────
static BOUNDARIES_3: [f32; 7] = [
    -1.7479307400, -1.0500762340, -0.5005750181,
    0.0,
    0.5005750181, 1.0500762340, 1.7479307400,
];
static CENTROIDS_3: [f32; 8] = [
    -2.1519456699, -1.3439655754, -0.7560052490, -0.2450940376,
    0.2450940376, 0.7560052490, 1.3439655754, 2.1519456699,
];

// ── b = 4 (16 levels) ───────────────────────────────────────────────────────
static BOUNDARIES_4: [f32; 15] = [
    -2.4008497854, -1.8435068261, -1.4370155399, -1.0993107638,
    -0.7995655283, -0.5224032460, -0.2582021029,
    0.0,
    0.2582021029, 0.5224032460, 0.7995655283, 1.0993107638,
    1.4370155399, 1.8435068261, 2.4008497854,
];
static CENTROIDS_4: [f32; 16] = [
    -2.7326368847, -2.0689965904, -1.6180460517, -1.2562309480,
    -0.9423402689, -0.6567589958, -0.3880482306, -0.1283950167,
    0.1283950167, 0.3880482306, 0.6567589958, 0.9423402689,
    1.2562309480, 1.6180460517, 2.0689965904, 2.7326368847,
];

/// Number of quantization levels for given bit-width.
pub const fn num_levels(nbits: u8) -> usize {
    1 << nbits
}

/// Returns (boundaries, centroids) for the given bit-width.
///
/// Centroids are for N(0,1). Caller must scale by 1/√D.
///
/// # Panics
/// Panics if nbits is not in [1, 4].
pub fn lloyd_max_codebook(nbits: u8) -> (&'static [f32], &'static [f32]) {
    match nbits {
        1 => (&BOUNDARIES_1, &CENTROIDS_1),
        2 => (&BOUNDARIES_2, &CENTROIDS_2),
        3 => (&BOUNDARIES_3, &CENTROIDS_3),
        4 => (&BOUNDARIES_4, &CENTROIDS_4),
        _ => panic!("unsupported nbits={}, must be 1-4", nbits),
    }
}

/// Find the quantization bin for a scalar value using binary search on boundaries.
///
/// Returns an index in [0, 2^nbits).
#[inline]
pub fn quantize_scalar(value: f32, boundaries: &[f32]) -> usize {
    // partition_point returns the number of boundaries < value,
    // which is exactly the bin index.
    boundaries.partition_point(|&b| b < value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codebook_sizes() {
        for nbits in 1..=4u8 {
            let (boundaries, centroids) = lloyd_max_codebook(nbits);
            let k = num_levels(nbits);
            assert_eq!(centroids.len(), k);
            assert_eq!(boundaries.len(), k - 1);
        }
    }

    #[test]
    fn test_codebook_sorted() {
        for nbits in 1..=4u8 {
            let (boundaries, centroids) = lloyd_max_codebook(nbits);
            for w in boundaries.windows(2) {
                assert!(w[0] < w[1], "boundaries not sorted for nbits={}", nbits);
            }
            for w in centroids.windows(2) {
                assert!(w[0] < w[1], "centroids not sorted for nbits={}", nbits);
            }
        }
    }

    #[test]
    fn test_codebook_symmetry() {
        for nbits in 1..=4u8 {
            let (boundaries, centroids) = lloyd_max_codebook(nbits);
            let k = centroids.len();
            for i in 0..k / 2 {
                assert!(
                    (centroids[i] + centroids[k - 1 - i]).abs() < 1e-6,
                    "centroids not symmetric for nbits={}", nbits
                );
            }
            let b = boundaries.len();
            for i in 0..b / 2 {
                assert!(
                    (boundaries[i] + boundaries[b - 1 - i]).abs() < 1e-6,
                    "boundaries not symmetric for nbits={}", nbits
                );
            }
        }
    }

    #[test]
    fn test_quantize_scalar_basic() {
        // b=1: boundary at 0. Negative → bin 0, positive → bin 1.
        assert_eq!(quantize_scalar(-1.0, &BOUNDARIES_1), 0);
        assert_eq!(quantize_scalar(1.0, &BOUNDARIES_1), 1);
        assert_eq!(quantize_scalar(0.0, &BOUNDARIES_1), 1); // 0 is not < 0

        // b=2: boundaries at [-0.98, 0.0, 0.98]
        assert_eq!(quantize_scalar(-2.0, &BOUNDARIES_2), 0);
        assert_eq!(quantize_scalar(-0.5, &BOUNDARIES_2), 1);
        assert_eq!(quantize_scalar(0.5, &BOUNDARIES_2), 2);
        assert_eq!(quantize_scalar(2.0, &BOUNDARIES_2), 3);
    }

    #[test]
    fn test_boundaries_between_centroids() {
        // Each boundary should lie between consecutive centroids
        for nbits in 1..=4u8 {
            let (boundaries, centroids) = lloyd_max_codebook(nbits);
            for (i, &b) in boundaries.iter().enumerate() {
                assert!(
                    centroids[i] < b && b < centroids[i + 1],
                    "boundary {} not between centroids {} and {} for nbits={}",
                    b, centroids[i], centroids[i + 1], nbits
                );
            }
        }
    }
}
