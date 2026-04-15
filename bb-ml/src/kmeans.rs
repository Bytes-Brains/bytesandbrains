use rand::Rng;

/// k-means++ clustering algorithm.
///
/// This implementation uses k-means++ initialization for better centroid
/// placement, followed by Lloyd's algorithm for iterative refinement.
pub struct KMeans {
    /// Cluster centroids (k vectors of dimension dim)
    pub centroids: Vec<Vec<f32>>,
    /// Number of clusters
    pub k: usize,
    /// Dimensionality of data points
    pub dim: usize,
}

impl KMeans {
    /// Run k-means++ clustering on the given data points.
    ///
    /// # Arguments
    /// * `data` - Slice of data points, each a Vec<f32> of the same dimension
    /// * `k` - Number of clusters
    /// * `max_iterations` - Maximum iterations for Lloyd's algorithm
    ///
    /// # Panics
    /// Panics if data is empty or k is 0.
    pub fn fit(data: &[Vec<f32>], k: usize, max_iterations: usize) -> Self {
        assert!(!data.is_empty(), "data cannot be empty");
        assert!(k > 0, "k must be positive");
        assert!(k <= data.len(), "k cannot exceed number of data points");

        let dim = data[0].len();

        // k-means++ initialization
        let centroids = Self::kmeans_plusplus_init(data, k);

        let mut kmeans = Self { centroids, k, dim };

        // Lloyd's algorithm iterations
        for _ in 0..max_iterations {
            let assignments = kmeans.assign(data);
            let new_centroids = kmeans.update_centroids(data, &assignments);

            // Check convergence
            if kmeans.has_converged(&new_centroids) {
                break;
            }
            kmeans.centroids = new_centroids;
        }

        kmeans
    }

    /// k-means++ initialization: choose centroids that are spread apart.
    ///
    /// 1. Choose first centroid uniformly at random
    /// 2. For each subsequent centroid, choose with probability proportional
    ///    to D(x)^2, where D(x) is distance to nearest existing centroid
    fn kmeans_plusplus_init(data: &[Vec<f32>], k: usize) -> Vec<Vec<f32>> {
        let mut rng = rand::thread_rng();
        let mut centroids = Vec::with_capacity(k);

        // 1. Choose first centroid uniformly at random
        let first_idx = rng.gen_range(0..data.len());
        centroids.push(data[first_idx].clone());

        // 2. Choose remaining centroids with probability proportional to D(x)^2
        for _ in 1..k {
            // Compute squared distances to nearest centroid
            let distances: Vec<f32> = data
                .iter()
                .map(|point| {
                    centroids
                        .iter()
                        .map(|c| squared_euclidean(point, c))
                        .fold(f32::MAX, f32::min)
                })
                .collect();

            // Compute cumulative distribution
            let total: f32 = distances.iter().sum();
            if total == 0.0 {
                // All points are at existing centroids, pick randomly
                let idx = rng.gen_range(0..data.len());
                centroids.push(data[idx].clone());
                continue;
            }

            let threshold = rng.gen::<f32>() * total;

            // Sample according to D(x)^2 distribution
            let mut cumulative = 0.0;
            let mut selected = data.len() - 1;
            for (i, &d) in distances.iter().enumerate() {
                cumulative += d;
                if cumulative >= threshold {
                    selected = i;
                    break;
                }
            }
            centroids.push(data[selected].clone());
        }

        centroids
    }

    /// Assign each point to the nearest centroid.
    fn assign(&self, data: &[Vec<f32>]) -> Vec<usize> {
        data.iter()
            .map(|point| {
                self.centroids
                    .iter()
                    .enumerate()
                    .map(|(i, c)| (i, squared_euclidean(point, c)))
                    .min_by(|a, b| {
                        a.1.partial_cmp(&b.1)
                            .unwrap_or(std::cmp::Ordering::Greater)
                    })
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            })
            .collect()
    }

    /// Update centroids as the mean of assigned points.
    fn update_centroids(&self, data: &[Vec<f32>], assignments: &[usize]) -> Vec<Vec<f32>> {
        let mut new_centroids = vec![vec![0.0; self.dim]; self.k];
        let mut counts = vec![0usize; self.k];

        for (point, &cluster) in data.iter().zip(assignments.iter()) {
            for (i, &val) in point.iter().enumerate() {
                new_centroids[cluster][i] += val;
            }
            counts[cluster] += 1;
        }

        for (centroid, &count) in new_centroids.iter_mut().zip(counts.iter()) {
            if count > 0 {
                for val in centroid.iter_mut() {
                    *val /= count as f32;
                }
            }
        }

        // Handle empty clusters by reinitializing from random data point
        let mut rng = rand::thread_rng();
        for (i, &count) in counts.iter().enumerate() {
            if count == 0 {
                let idx = rng.gen_range(0..data.len());
                new_centroids[i] = data[idx].clone();
            }
        }

        new_centroids
    }

    /// Check if centroids have converged.
    fn has_converged(&self, new_centroids: &[Vec<f32>]) -> bool {
        const EPSILON: f32 = 1e-6;
        self.centroids
            .iter()
            .zip(new_centroids.iter())
            .all(|(old, new)| squared_euclidean(old, new) < EPSILON)
    }

    /// Find the index of the nearest centroid for a point.
    pub fn predict(&self, point: &[f32]) -> usize {
        self.centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, squared_euclidean(point, c)))
            .min_by(|a, b| {
                a.1.partial_cmp(&b.1)
                    .unwrap_or(std::cmp::Ordering::Greater)
            })
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Get the centroid for a given cluster index.
    pub fn centroid(&self, cluster: usize) -> &[f32] {
        &self.centroids[cluster]
    }
}

/// Compute squared Euclidean distance between two vectors.
#[inline]
fn squared_euclidean(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = x - y;
            diff * diff
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kmeans_basic() {
        // Two clearly separated clusters
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![0.2, 0.0],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
            vec![10.2, 10.0],
        ];

        let kmeans = KMeans::fit(&data, 2, 100);

        // Should find 2 clusters
        assert_eq!(kmeans.k, 2);
        assert_eq!(kmeans.centroids.len(), 2);

        // Points in same cluster should be assigned together
        let a0 = kmeans.predict(&data[0]);
        let a1 = kmeans.predict(&data[1]);
        let a2 = kmeans.predict(&data[2]);
        let a3 = kmeans.predict(&data[3]);
        let a4 = kmeans.predict(&data[4]);
        let a5 = kmeans.predict(&data[5]);

        assert_eq!(a0, a1);
        assert_eq!(a1, a2);
        assert_eq!(a3, a4);
        assert_eq!(a4, a5);
        assert_ne!(a0, a3);
    }

    #[test]
    fn test_kmeans_single_cluster() {
        let data = vec![vec![1.0, 1.0], vec![1.1, 1.0], vec![1.0, 1.1]];

        let kmeans = KMeans::fit(&data, 1, 100);

        assert_eq!(kmeans.k, 1);
        // Centroid should be near (1.033, 1.033)
        assert!((kmeans.centroids[0][0] - 1.033).abs() < 0.1);
    }

    #[test]
    fn test_kmeans_predict() {
        let data = vec![
            vec![0.0, 0.0],
            vec![10.0, 10.0],
        ];

        let kmeans = KMeans::fit(&data, 2, 100);

        // New point close to first cluster
        let cluster = kmeans.predict(&[0.5, 0.5]);
        let expected = kmeans.predict(&[0.0, 0.0]);
        assert_eq!(cluster, expected);
    }

    #[test]
    fn test_squared_euclidean() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        let dist = squared_euclidean(&a, &b);
        // (3^2 + 3^2 + 3^2) = 27
        assert_eq!(dist, 27.0);
    }
}
