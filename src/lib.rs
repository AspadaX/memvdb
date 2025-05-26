//! # MemVDB - An In-Memory Vector Database
//!
//! MemVDB is a fast, lightweight in-memory vector database written in Rust.
//! It supports multiple distance metrics and provides efficient similarity search
//! for machine learning applications, recommendation systems, and semantic search.
//!
//! ## Features
//!
//! - **Multiple Distance Metrics**: Euclidean, Cosine, and Dot Product
//! - **High Performance**: Optimized similarity search with binary heap algorithms
//! - **Flexible Metadata**: Store arbitrary metadata with each embedding
//! - **Batch Operations**: Efficient batch insertion and updates
//! - **Thread Safety**: Safe concurrent access with proper locking
//! - **Zero Dependencies**: Minimal external dependencies for core functionality
//!
//! ## Quick Start
//!
//! ```rust
//! use memvdb::{CacheDB, Distance, Embedding};
//! use std::collections::HashMap;
//!
//! // Create a new in-memory vector database
//! let mut db = CacheDB::new();
//!
//! // Create a collection with 128-dimensional vectors using cosine similarity
//! db.create_collection("documents".to_string(), 128, Distance::Cosine).unwrap();
//!
//! // Create an embedding with metadata
//! let mut id = HashMap::new();
//! id.insert("doc_id".to_string(), "doc_001".to_string());
//!
//! let mut metadata = HashMap::new();
//! metadata.insert("title".to_string(), "Sample Document".to_string());
//! metadata.insert("category".to_string(), "AI".to_string());
//!
//! let vector = vec![0.1; 128]; // 128-dimensional vector
//! let embedding = Embedding {
//!     id,
//!     vector,
//!     metadata: Some(metadata),
//! };
//!
//! // Insert the embedding
//! db.insert_into_collection("documents", embedding).unwrap();
//!
//! // Perform similarity search
//! let query_vector = vec![0.2; 128];
//! let collection = db.get_collection("documents").unwrap();
//! let results = collection.get_similarity(&query_vector, 5);
//!
//! println!("Found {} similar documents", results.len());
//! ```
//!
//! ## Distance Metrics
//!
//! MemVDB supports three distance metrics optimized for different use cases:
//!
//! - **Euclidean Distance**: Best for spatial data and when absolute distances matter
//! - **Cosine Similarity**: Ideal for text embeddings and high-dimensional sparse data
//! - **Dot Product**: Efficient for normalized vectors and neural network outputs
//!
//! ## Architecture
//!
//! The library is organized into three main modules:
//!
//! - `db`: Core database functionality, collections, and embeddings management
//! - `similarity`: Distance calculation and vector operations
//! - Public API exports for easy integration
//!
//! ## Performance Characteristics
//!
//! - **Insertion**: O(1) average case for single embeddings
//! - **Similarity Search**: O(n) where n is the number of embeddings in the collection
//! - **Memory Usage**: Linear with number of embeddings and vector dimensions
//! - **Concurrency**: Thread-safe operations with mutex-based protection

mod db;
mod similarity;

// Re-export public APIs for easy access
pub use db::{
    BatchInsertEmbeddingsStruct, CacheDB, Collection, CollectionHandlerStruct,
    CreateCollectionStruct, Distance, Embedding, Error, GetSimilarityStruct, InsertEmbeddingStruct,
    SimilarityResult,
};

pub use similarity::{ScoreIndex, get_cache_attr, get_distance_fn, normalize};

/// Convenience function for quick database setup
///
/// Creates a new `CacheDB` instance with default settings.
/// This is equivalent to calling `CacheDB::new()`.
///
/// # Examples
///
/// ```rust
/// use memvdb::create_database;
///
/// let db = create_database();
/// assert!(db.collections.is_empty());
/// ```
pub fn create_database() -> CacheDB {
    CacheDB::new()
}

/// Simple addition function for testing purposes
///
/// This function is primarily used for testing the library setup
/// and can be removed in production versions.
///
/// # Arguments
///
/// * `left` - First number to add
/// * `right` - Second number to add
///
/// # Returns
///
/// The sum of `left` and `right`
///
/// # Examples
///
/// ```rust
/// use memvdb::add;
///
/// assert_eq!(add(2, 3), 5);
/// assert_eq!(add(0, 0), 0);
/// ```
pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Tests the basic add function
    #[test]
    fn test_add_function() {
        assert_eq!(add(2, 2), 4);
        assert_eq!(add(0, 0), 0);
        assert_eq!(add(100, 200), 300);
    }

    /// Tests that all main library types and functions are exported correctly
    #[test]
    fn test_library_exports() {
        // Test that all main types are accessible
        let db = CacheDB::new();
        assert!(db.collections.is_empty());

        // Test Distance enum
        let _euclidean = Distance::Euclidean;
        let _cosine = Distance::Cosine;
        let _dot_product = Distance::DotProduct;

        // Test that we can create structs
        let _create_struct = CreateCollectionStruct {
            collection_name: "test".to_string(),
            dimension: 128,
            distance: Distance::Euclidean,
        };
    }

    /// Tests the convenience database creation function
    #[test]
    fn test_create_database_function() {
        let db = create_database();
        assert!(db.collections.is_empty());

        // Should be equivalent to CacheDB::new()
        let db2 = CacheDB::new();
        assert_eq!(db.collections.len(), db2.collections.len());
    }

    /// Tests embedding creation with metadata
    #[test]
    fn test_embedding_creation() {
        let mut id = HashMap::new();
        id.insert("id".to_string(), "test_embedding".to_string());

        let mut metadata = HashMap::new();
        metadata.insert("type".to_string(), "document".to_string());
        metadata.insert("source".to_string(), "test".to_string());

        let embedding = Embedding {
            id,
            vector: vec![0.1, 0.2, 0.3, 0.4],
            metadata: Some(metadata),
        };

        assert_eq!(embedding.vector.len(), 4);
        assert!(embedding.metadata.is_some());
        assert_eq!(embedding.id.get("id"), Some(&"test_embedding".to_string()));
    }

    /// Tests distance enum equality comparisons
    #[test]
    fn test_distance_enum_equality() {
        assert_eq!(Distance::Euclidean, Distance::Euclidean);
        assert_eq!(Distance::Cosine, Distance::Cosine);
        assert_eq!(Distance::DotProduct, Distance::DotProduct);
        assert_ne!(Distance::Euclidean, Distance::Cosine);
    }

    #[test]
    fn test_similarity_functions_accessible() {
        let vector = vec![1.0, 2.0, 3.0];

        // Test cache attribute calculation
        let euclidean_cache = get_cache_attr(Distance::Euclidean, &vector);
        let cosine_cache = get_cache_attr(Distance::Cosine, &vector);
        let dot_cache = get_cache_attr(Distance::DotProduct, &vector);

        assert_eq!(euclidean_cache, 0.0);
        assert_eq!(dot_cache, 0.0);
        assert!(cosine_cache > 0.0); // Should be the magnitude

        // Test normalization
        let normalized = normalize(&vector);
        assert_eq!(normalized.len(), vector.len());

        // Test magnitude is approximately 1.0 after normalization
        let magnitude: f32 = normalized.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
        assert!((magnitude - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_score_index_ordering() {
        let score1 = ScoreIndex {
            score: 0.5,
            index: 0,
        };
        let score2 = ScoreIndex {
            score: 0.8,
            index: 1,
        };
        let score3 = ScoreIndex {
            score: 0.3,
            index: 2,
        };

        // ScoreIndex implements reverse ordering for min-heap behavior
        assert!(score3 > score1); // Lower scores are "greater" for min-heap
        assert!(score1 > score2); // Higher scores are "less" for min-heap
    }

    #[test]
    fn test_error_enum() {
        let unique_error = Error::UniqueViolation;
        let not_found_error = Error::NotFound;
        let dimension_error = Error::DimensionMismatch;

        // Test that errors can be created and compared
        assert_eq!(unique_error, Error::UniqueViolation);
        assert_ne!(unique_error, not_found_error);

        // Test error display (should not panic)
        let _error_string = format!("{:?}", dimension_error);
    }

    #[test]
    fn test_collection_creation_and_basic_operations() {
        let mut db = CacheDB::new();

        // Create a collection
        let result = db.create_collection("test_collection".to_string(), 3, Distance::Euclidean);
        assert!(result.is_ok());

        // Verify collection exists
        let collection = db.get_collection("test_collection");
        assert!(collection.is_some());
        assert_eq!(collection.unwrap().dimension, 3);
    }

    #[test]
    fn test_end_to_end_workflow() {
        let mut db = CacheDB::new();

        // Create collection
        db.create_collection("documents".to_string(), 4, Distance::Cosine)
            .unwrap();

        // Create embeddings
        let mut id1 = HashMap::new();
        id1.insert("doc_id".to_string(), "doc1".to_string());

        let mut metadata1 = HashMap::new();
        metadata1.insert("title".to_string(), "Document 1".to_string());

        let embedding1 = Embedding {
            id: id1,
            vector: vec![1.0, 0.0, 0.0, 0.0],
            metadata: Some(metadata1),
        };

        let mut id2 = HashMap::new();
        id2.insert("doc_id".to_string(), "doc2".to_string());

        let embedding2 = Embedding {
            id: id2,
            vector: vec![0.0, 1.0, 0.0, 0.0],
            metadata: None,
        };

        // Insert embeddings
        assert!(db.insert_into_collection("documents", embedding1).is_ok());
        assert!(db.insert_into_collection("documents", embedding2).is_ok());

        // Get embeddings back
        let embeddings = db.get_embeddings("documents");
        assert!(embeddings.is_some());
        assert_eq!(embeddings.unwrap().len(), 2);

        // Test similarity search
        let collection = db.get_collection("documents").unwrap();
        let query_vector = vec![1.0, 0.1, 0.0, 0.0];
        let results = collection.get_similarity(&query_vector, 2);

        assert_eq!(results.len(), 2);
        // First result should be most similar (closest to [1,0,0,0])
        assert!(results[0].score >= results[1].score);
    }

    #[test]
    fn test_normalize_function_edge_cases() {
        // Test zero vector
        let zero_vec = vec![0.0, 0.0, 0.0];
        let normalized_zero = normalize(&zero_vec);
        assert_eq!(normalized_zero, zero_vec);

        // Test very small vector
        let small_vec = vec![1e-10, 1e-10, 1e-10];
        let normalized_small = normalize(&small_vec);
        assert_eq!(normalized_small, small_vec); // Should remain unchanged due to epsilon check

        // Test unit vector
        let unit_vec = vec![1.0, 0.0, 0.0];
        let normalized_unit = normalize(&unit_vec);
        let magnitude: f32 = normalized_unit
            .iter()
            .map(|x| x.powi(2))
            .sum::<f32>()
            .sqrt();
        assert!((magnitude - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_distance_functions() {
        let vec1 = vec![1.0, 2.0, 3.0];
        let vec2 = vec![4.0, 5.0, 6.0];

        // Test Euclidean distance function
        let euclidean_fn = get_distance_fn(Distance::Euclidean);
        let euclidean_dist = euclidean_fn(&vec1, &vec2, 0.0);
        assert!(euclidean_dist > 0.0);

        // Test dot product function
        let dot_fn = get_distance_fn(Distance::DotProduct);
        let dot_result = dot_fn(&vec1, &vec2, 0.0);
        assert_eq!(dot_result, 32.0); // 1*4 + 2*5 + 3*6 = 32

        // Test cosine function (uses dot product internally)
        let cosine_fn = get_distance_fn(Distance::Cosine);
        let cosine_result = cosine_fn(&vec1, &vec2, 0.0);
        assert_eq!(cosine_result, dot_result); // Should be same as dot product
    }
}
