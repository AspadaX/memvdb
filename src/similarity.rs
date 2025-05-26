//! # Similarity Module
//!
//! This module provides distance calculation functions, vector operations, and
//! similarity search utilities for the MemVDB vector database.
//!
//! ## Key Functions
//!
//! - [`get_cache_attr`]: Pre-computes attributes for optimized distance calculations
//! - [`get_distance_fn`]: Returns the appropriate distance function for a metric
//! - [`normalize`]: Normalizes vectors to unit length for cosine similarity
//! - [`ScoreIndex`]: Helper struct for efficient k-nearest neighbor search
//!
//! ## Distance Functions
//!
//! The module implements three optimized distance functions:
//!
//! - **Euclidean Distance**: Uses sum of squares optimization
//! - **Dot Product**: Simple and efficient for normalized vectors
//! - **Cosine Similarity**: Implemented as dot product on normalized vectors
//!
//! ## Performance Optimizations
//!
//! - Pre-computed cache attributes reduce redundant calculations
//! - Binary heap-based k-NN search for efficient top-k retrieval
//! - SIMD-friendly vector operations where possible

use std::cmp::Ordering;

use crate::db::Distance;

/// Pre-computes cacheable attributes for distance calculations.
///
/// This function calculates values that can be reused across multiple distance
/// computations to improve performance. The returned value depends on the distance metric:
///
/// - **Euclidean/Dot Product**: Returns 0.0 (no cacheable attributes)
/// - **Cosine**: Returns the magnitude (L2 norm) of the vector
///
/// # Arguments
///
/// * `metric` - The distance metric being used
/// * `vec` - The vector to compute cache attributes for
///
/// # Returns
///
/// A floating-point value that can be reused in distance calculations,
/// or 0.0 if no optimization is possible.
///
/// # Examples
///
/// ```rust
/// use memvdb::{get_cache_attr, Distance};
///
/// let vector = vec![3.0, 4.0]; // Magnitude = 5.0
/// let cache_attr = get_cache_attr(Distance::Cosine, &vector);
/// assert!((cache_attr - 5.0).abs() < 1e-6);
///
/// // No caching for Euclidean distance
/// let cache_attr = get_cache_attr(Distance::Euclidean, &vector);
/// assert_eq!(cache_attr, 0.0);
/// ```
pub fn get_cache_attr(metric: Distance, vec: &[f32]) -> f32 {
    match metric {
        // Dot product doesn't allow any caching
        Distance::DotProduct | Distance::Euclidean => 0.0,
        // Precompute the magnitude of the vector
        Distance::Cosine => vec.iter().map(|&x| x.powi(2)).sum::<f32>().sqrt(),
    }
}

/// Returns the appropriate distance function for the specified metric.
///
/// This function provides access to optimized distance calculation functions
/// based on the chosen similarity metric. The returned function takes two vectors
/// and a cached attribute value as parameters.
///
/// # Arguments
///
/// * `metric` - The distance metric to use
///
/// # Returns
///
/// A function that calculates distance between two vectors:
/// `fn(&[f32], &[f32], f32) -> f32`
///
/// The parameters are: (vector_a, vector_b, cached_attribute) -> distance
///
/// # Examples
///
/// ```rust
/// use memvdb::{get_distance_fn, Distance};
///
/// let distance_fn = get_distance_fn(Distance::Euclidean);
/// let vec1 = vec![1.0, 0.0];
/// let vec2 = vec![0.0, 1.0];
/// let distance = distance_fn(&vec1, &vec2, 0.0);
/// println!("Euclidean distance: {}", distance);
/// ```
///
/// # Distance Function Signatures
///
/// All returned functions have the signature:
/// ```ignore
/// fn(a: &[f32], b: &[f32], cached_attr: f32) -> f32
/// ```
///
/// Where `cached_attr` is the value returned by [`get_cache_attr`].
pub fn get_distance_fn(metric: Distance) -> impl Fn(&[f32], &[f32], f32) -> f32 {
    match metric {
        Distance::Euclidean => euclidian_distance,
        // We use dot product for cosine because we've normalized the vectors on insertion
        Distance::Cosine | Distance::DotProduct => dot_product,
    }
}

/// Calculates the Euclidean distance between two vectors.
///
/// Uses an optimized formula that leverages pre-computed sum of squares
/// to reduce computational overhead. The formula is:
///
/// `sqrt(2 * (1 - dot_product) + a_sum_squares + b_sum_squares)`
///
/// # Arguments
///
/// * `a` - First vector
/// * `b` - Second vector  
/// * `a_sum_squares` - Pre-computed sum of squares for vector `a` (currently unused, always 0.0)
///
/// # Returns
///
/// The Euclidean distance as a non-negative floating-point number.
///
/// # Performance
///
/// This implementation uses `mul_add` for better numerical precision and
/// performance on systems with fused multiply-add instructions.
fn euclidian_distance(a: &[f32], b: &[f32], a_sum_squares: f32) -> f32 {
    let mut cross_terms = 0.0;
    let mut b_sum_squares = 0.0;

    for (i, j) in a.iter().zip(b) {
        cross_terms += i * j;
        b_sum_squares += j.powi(2);
    }

    2.0f32
        .mul_add(-cross_terms, a_sum_squares + b_sum_squares)
        .max(0.0)
        .sqrt()
}

/// Calculates the dot product of two vectors.
///
/// Computes the sum of element-wise products: `sum(a_i * b_i)`.
/// This function is used for both dot product similarity and cosine similarity
/// (when vectors are pre-normalized).
///
/// # Arguments
///
/// * `a` - First vector
/// * `b` - Second vector
/// * `_` - Unused cached attribute parameter (for API consistency)
///
/// # Returns
///
/// The dot product as a floating-point number. For normalized vectors,
/// this represents the cosine similarity.
///
/// # Examples
///
/// ```ignore
/// let a = vec![1.0, 2.0, 3.0];
/// let b = vec![4.0, 5.0, 6.0];
/// let result = dot_product(&a, &b, 0.0);
/// // result = 1*4 + 2*5 + 3*6 = 32.0
/// ```
fn dot_product(a: &[f32], b: &[f32], _: f32) -> f32 {
    a.iter().zip(b).fold(0.0, |acc, (x, y)| acc + x * y)
}

/// Normalizes a vector to unit length (L2 normalization).
///
/// Converts the input vector to have a magnitude of 1.0 while preserving
/// its direction. This is essential for cosine similarity calculations.
/// Vectors with magnitude below `f32::EPSILON` are returned unchanged
/// to avoid division by zero.
///
/// # Arguments
///
/// * `vec` - The vector to normalize
///
/// # Returns
///
/// A new vector with unit magnitude, or the original vector if it's too small.
///
/// # Examples
///
/// ```rust
/// use memvdb::normalize;
///
/// let vector = vec![3.0, 4.0]; // Magnitude = 5.0
/// let normalized = normalize(&vector);
/// // Result: [0.6, 0.8] with magnitude ≈ 1.0
///
/// let magnitude: f32 = normalized.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
/// assert!((magnitude - 1.0).abs() < 1e-6);
/// ```
///
/// # Edge Cases
///
/// - **Zero vector**: Returns the original vector unchanged
/// - **Very small vectors**: Returns the original vector to avoid numerical instability
/// - **Single element**: Normalizes to [1.0] or [-1.0]
pub fn normalize(vec: &[f32]) -> Vec<f32> {
    let magnitude = (vec.iter().fold(0.0, |acc, &val| val.mul_add(val, acc))).sqrt();

    if magnitude > std::f32::EPSILON {
        vec.iter().map(|&val| val / magnitude).collect()
    } else {
        vec.to_vec()
    }
}

/// A helper structure for k-nearest neighbor search operations.
///
/// `ScoreIndex` pairs a similarity score with an index into the embeddings vector.
/// It implements custom ordering to work with Rust's `BinaryHeap` as a min-heap,
/// allowing efficient retrieval of the k most similar embeddings.
///
/// # Fields
///
/// * `score` - The similarity score (lower values indicate better matches)
/// * `index` - Index into the embeddings vector
///
/// # Ordering Behavior
///
/// The ordering is intentionally reversed to create min-heap behavior:
/// - Lower scores are considered "greater"
/// - Higher scores are considered "less"
/// - This allows `BinaryHeap` to efficiently maintain the k best (lowest) scores
///
/// # Examples
///
/// ```rust
/// use memvdb::ScoreIndex;
/// use std::collections::BinaryHeap;
///
/// let mut heap = BinaryHeap::new();
/// heap.push(ScoreIndex { score: 0.8, index: 0 });
/// heap.push(ScoreIndex { score: 0.2, index: 1 });
///
/// let best = heap.pop().unwrap();
/// assert_eq!(best.score, 0.2); // Lower score comes first
/// ```
#[derive(Debug)]
pub struct ScoreIndex {
    /// The similarity score - lower values indicate better matches
    pub score: f32,
    /// Index of the embedding in the collection's embeddings vector
    pub index: usize,
}

/// Equality comparison based only on the score value.
///
/// Two `ScoreIndex` instances are considered equal if their scores are equal,
/// regardless of their index values. This is important for the binary heap
/// operations used in similarity search.
impl PartialEq for ScoreIndex {
    fn eq(&self, other: &Self) -> bool {
        self.score.eq(&other.score)
    }
}

impl Eq for ScoreIndex {}

/// Partial ordering implementation for min-heap behavior.
///
/// The comparison is intentionally reversed: a `ScoreIndex` with a lower score
/// is considered "greater" than one with a higher score. This allows
/// `BinaryHeap` to function as a min-heap, where the smallest scores
/// (best matches) have the highest priority.
///
/// # Returns
///
/// - `Some(Ordering::Greater)` if self has a lower (better) score
/// - `Some(Ordering::Less)` if self has a higher (worse) score  
/// - `Some(Ordering::Equal)` if scores are equal
/// - `None` if either score is NaN
impl PartialOrd for ScoreIndex {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // The comparison is intentionally reversed here to make the heap a min-heap
        other.score.partial_cmp(&self.score)
    }
}

/// Total ordering implementation that handles NaN values gracefully.
///
/// Falls back to `Ordering::Equal` when `partial_cmp` returns `None`
/// (which happens when comparing with NaN values). This ensures that
/// the binary heap operations remain stable even with invalid scores.
impl Ord for ScoreIndex {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}
