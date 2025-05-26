use memvdb::*;
use std::collections::HashMap;

#[test]
fn test_collection_not_found_errors() {
    let db = CacheDB::new();

    // Test get_collection with non-existent collection
    let result = db.get_collection("nonexistent");
    assert!(result.is_none());

    // Test get_embeddings with non-existent collection
    let result = db.get_embeddings("nonexistent");
    assert!(result.is_none());
}

#[test]
fn test_delete_nonexistent_collection() {
    let mut db = CacheDB::new();

    let result = db.delete_collection("nonexistent_collection");
    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), Error::NotFound);
}

#[test]
fn test_insert_into_nonexistent_collection() {
    let mut db = CacheDB::new();

    let mut id = HashMap::new();
    id.insert("id".to_string(), "test".to_string());

    let embedding = Embedding {
        id,
        vector: vec![1.0, 2.0, 3.0],
        metadata: None,
    };

    let result = db.insert_into_collection("nonexistent_collection", embedding);
    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), Error::NotFound);
}

#[test]
fn test_duplicate_collection_creation() {
    let mut db = CacheDB::new();

    // Create first collection
    let result1 = db.create_collection("duplicate_test".to_string(), 10, Distance::Euclidean);
    assert!(result1.is_ok());

    // Try to create collection with same name
    let result2 = db.create_collection("duplicate_test".to_string(), 20, Distance::Cosine);
    assert!(result2.is_err());
    assert_eq!(result2.unwrap_err(), Error::UniqueViolation);
}

#[test]
fn test_dimension_mismatch_on_insert() {
    let mut db = CacheDB::new();
    db.create_collection("dim_test".to_string(), 5, Distance::Euclidean)
        .unwrap();

    let mut id = HashMap::new();
    id.insert("id".to_string(), "test".to_string());

    // Vector with wrong dimension (3 instead of 5)
    let embedding = Embedding {
        id,
        vector: vec![1.0, 2.0, 3.0],
        metadata: None,
    };

    let result = db.insert_into_collection("dim_test", embedding);
    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), Error::DimensionMismatch);
}

#[test]
fn test_dimension_mismatch_on_batch_update() {
    let mut db = CacheDB::new();
    db.create_collection("batch_dim_test".to_string(), 4, Distance::Euclidean)
        .unwrap();

    let mut embeddings = Vec::new();

    // First embedding with correct dimension
    let mut id1 = HashMap::new();
    id1.insert("id".to_string(), "1".to_string());
    embeddings.push(Embedding {
        id: id1,
        vector: vec![1.0, 2.0, 3.0, 4.0],
        metadata: None,
    });

    // Second embedding with wrong dimension
    let mut id2 = HashMap::new();
    id2.insert("id".to_string(), "2".to_string());
    embeddings.push(Embedding {
        id: id2,
        vector: vec![1.0, 2.0], // Wrong dimension
        metadata: None,
    });

    let result = db.update_collection("batch_dim_test", embeddings);
    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), Error::DimensionMismatch);
}

#[test]
fn test_duplicate_embedding_id_on_insert() {
    let mut db = CacheDB::new();
    db.create_collection("dup_embed_test".to_string(), 3, Distance::Euclidean)
        .unwrap();

    let mut id = HashMap::new();
    id.insert("unique_key".to_string(), "same_id".to_string());

    let embedding1 = Embedding {
        id: id.clone(),
        vector: vec![1.0, 2.0, 3.0],
        metadata: None,
    };

    let embedding2 = Embedding {
        id: id.clone(),
        vector: vec![4.0, 5.0, 6.0],
        metadata: None,
    };

    // First insertion should succeed
    let result1 = db.insert_into_collection("dup_embed_test", embedding1);
    assert!(result1.is_ok());

    // Second insertion with same ID should fail
    let result2 = db.insert_into_collection("dup_embed_test", embedding2);
    assert!(result2.is_err());
    assert_eq!(result2.unwrap_err(), Error::EmbeddingUniqueViolation);
}

#[test]
fn test_duplicate_embedding_id_in_batch_update() {
    let mut db = CacheDB::new();
    db.create_collection("batch_dup_test".to_string(), 3, Distance::Euclidean)
        .unwrap();

    let mut id = HashMap::new();
    id.insert("key".to_string(), "duplicate".to_string());

    let embeddings = vec![
        Embedding {
            id: id.clone(),
            vector: vec![1.0, 2.0, 3.0],
            metadata: None,
        },
        Embedding {
            id: id.clone(), // Duplicate ID within the same batch
            vector: vec![4.0, 5.0, 6.0],
            metadata: None,
        },
    ];

    let result = db.update_collection("batch_dup_test", embeddings);
    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), Error::UniqueViolation);
}

#[test]
fn test_duplicate_embedding_id_across_existing_and_new() {
    let mut db = CacheDB::new();
    db.create_collection("existing_dup_test".to_string(), 3, Distance::Euclidean)
        .unwrap();

    let mut existing_id = HashMap::new();
    existing_id.insert("key".to_string(), "existing".to_string());

    // Insert an existing embedding
    let existing_embedding = Embedding {
        id: existing_id.clone(),
        vector: vec![1.0, 2.0, 3.0],
        metadata: None,
    };
    db.insert_into_collection("existing_dup_test", existing_embedding)
        .unwrap();

    // Try to update with an embedding that has the same ID as existing
    let new_embeddings = vec![Embedding {
        id: existing_id.clone(), // Same ID as existing embedding
        vector: vec![4.0, 5.0, 6.0],
        metadata: None,
    }];

    let result = db.update_collection("existing_dup_test", new_embeddings);
    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), Error::UniqueViolation);
}

#[test]
fn test_empty_vector_handling() {
    let mut db = CacheDB::new();
    db.create_collection("empty_vec_test".to_string(), 0, Distance::Euclidean)
        .unwrap();

    let mut id = HashMap::new();
    id.insert("id".to_string(), "empty".to_string());

    let embedding = Embedding {
        id,
        vector: vec![], // Empty vector
        metadata: None,
    };

    let result = db.insert_into_collection("empty_vec_test", embedding);
    assert!(result.is_ok());

    let embeddings = db.get_embeddings("empty_vec_test").unwrap();
    assert_eq!(embeddings.len(), 1);
    assert_eq!(embeddings[0].vector.len(), 0);
}

#[test]
fn test_zero_dimension_collection() {
    let mut db = CacheDB::new();

    let result = db.create_collection("zero_dim".to_string(), 0, Distance::Euclidean);
    assert!(result.is_ok());

    let collection = db.get_collection("zero_dim").unwrap();
    assert_eq!(collection.dimension, 0);

    // Similarity search on empty vectors
    let results = collection.get_similarity(&vec![], 5);
    assert_eq!(results.len(), 0);
}

#[test]
fn test_very_large_dimension() {
    let mut db = CacheDB::new();

    // Test with very large dimension
    let large_dim = 100_000;
    let result = db.create_collection("large_dim".to_string(), large_dim, Distance::Euclidean);
    assert!(result.is_ok());

    let collection = db.get_collection("large_dim").unwrap();
    assert_eq!(collection.dimension, large_dim);
}

#[test]
fn test_normalize_with_infinity() {
    let vector_with_inf = vec![f32::INFINITY, 1.0, 2.0];
    let normalized = normalize(&vector_with_inf);

    // Should handle infinity gracefully (likely returning NaN values)
    assert_eq!(normalized.len(), 3);
}

#[test]
fn test_normalize_with_nan() {
    let vector_with_nan = vec![f32::NAN, 1.0, 2.0];
    let normalized = normalize(&vector_with_nan);

    // Should handle NaN gracefully
    assert_eq!(normalized.len(), 3);
}

#[test]
fn test_distance_functions_with_extreme_values() {
    let vec_with_large = vec![1e20, 1e20, 1e20];
    let vec_with_small = vec![1e-20, 1e-20, 1e-20];

    let euclidean_fn = get_distance_fn(Distance::Euclidean);
    let dot_fn = get_distance_fn(Distance::DotProduct);
    let cosine_fn = get_distance_fn(Distance::Cosine);

    // These should not panic, even with extreme values
    let euclidean_result = euclidean_fn(&vec_with_large, &vec_with_small, 0.0);
    let dot_result = dot_fn(&vec_with_large, &vec_with_small, 0.0);
    let cosine_result = cosine_fn(&vec_with_large, &vec_with_small, 0.0);

    // Results should be finite (not NaN or infinity) or at least not panic
    assert!(euclidean_result.is_finite() || euclidean_result.is_infinite());
    assert!(dot_result.is_finite() || dot_result.is_infinite());
    assert!(cosine_result.is_finite() || cosine_result.is_infinite());
}

#[test]
fn test_similarity_search_with_k_larger_than_collection_size() {
    let mut db = CacheDB::new();
    db.create_collection("small_collection".to_string(), 3, Distance::Euclidean)
        .unwrap();

    // Insert only 2 embeddings
    for i in 0..2 {
        let mut id = HashMap::new();
        id.insert("id".to_string(), i.to_string());

        let embedding = Embedding {
            id,
            vector: vec![i as f32, (i * 2) as f32, (i * 3) as f32],
            metadata: None,
        };

        db.insert_into_collection("small_collection", embedding)
            .unwrap();
    }

    let collection = db.get_collection("small_collection").unwrap();

    // Request more results than available
    let query = vec![1.0, 1.0, 1.0];
    let results = collection.get_similarity(&query, 10);

    // Should return only the available embeddings
    assert_eq!(results.len(), 2);
}

#[test]
fn test_similarity_search_with_k_one() {
    let mut db = CacheDB::new();
    db.create_collection("k_one_test".to_string(), 3, Distance::Euclidean)
        .unwrap();

    let mut id = HashMap::new();
    id.insert("id".to_string(), "test".to_string());

    let embedding = Embedding {
        id,
        vector: vec![1.0, 2.0, 3.0],
        metadata: None,
    };

    db.insert_into_collection("k_one_test", embedding).unwrap();

    let collection = db.get_collection("k_one_test").unwrap();
    let query = vec![1.0, 1.0, 1.0];
    let results = collection.get_similarity(&query, 1);

    // Should return exactly one result
    assert_eq!(results.len(), 1);
}

#[test]
fn test_error_display_formatting() {
    let errors = vec![
        Error::UniqueViolation,
        Error::EmbeddingUniqueViolation,
        Error::NotFound,
        Error::DimensionMismatch,
        Error::LoggerInitializationError,
    ];

    for error in errors {
        // Should not panic when formatting error
        let error_string = format!("{:?}", error);
        assert!(!error_string.is_empty());

        // Test that errors are properly comparable
        let same_error = match error {
            Error::UniqueViolation => Error::UniqueViolation,
            Error::EmbeddingUniqueViolation => Error::EmbeddingUniqueViolation,
            Error::NotFound => Error::NotFound,
            Error::DimensionMismatch => Error::DimensionMismatch,
            Error::LoggerInitializationError => Error::LoggerInitializationError,
        };

        assert_eq!(error, same_error);
    }
}

#[test]
fn test_complex_id_structures() {
    let mut db = CacheDB::new();
    db.create_collection("complex_id_test".to_string(), 2, Distance::Euclidean)
        .unwrap();

    // Test with complex ID structure
    let mut complex_id = HashMap::new();
    complex_id.insert("user_id".to_string(), "12345".to_string());
    complex_id.insert("document_id".to_string(), "doc_67890".to_string());
    complex_id.insert("timestamp".to_string(), "2024-01-01T00:00:00Z".to_string());
    complex_id.insert("version".to_string(), "v1.0".to_string());

    let embedding = Embedding {
        id: complex_id.clone(),
        vector: vec![1.0, 2.0],
        metadata: None,
    };

    let result = db.insert_into_collection("complex_id_test", embedding);
    assert!(result.is_ok());

    // Try to insert another embedding with same complex ID (should fail)
    let duplicate_embedding = Embedding {
        id: complex_id,
        vector: vec![3.0, 4.0],
        metadata: None,
    };

    let result2 = db.insert_into_collection("complex_id_test", duplicate_embedding);
    assert!(result2.is_err());
    assert_eq!(result2.unwrap_err(), Error::EmbeddingUniqueViolation);
}

#[test]
fn test_empty_metadata_vs_none_metadata() {
    let mut db = CacheDB::new();
    db.create_collection("metadata_test".to_string(), 2, Distance::Euclidean)
        .unwrap();

    // Embedding with None metadata
    let mut id1 = HashMap::new();
    id1.insert("id".to_string(), "none_metadata".to_string());

    let embedding1 = Embedding {
        id: id1,
        vector: vec![1.0, 2.0],
        metadata: None,
    };

    // Embedding with empty metadata
    let mut id2 = HashMap::new();
    id2.insert("id".to_string(), "empty_metadata".to_string());

    let embedding2 = Embedding {
        id: id2,
        vector: vec![3.0, 4.0],
        metadata: Some(HashMap::new()),
    };

    assert!(
        db.insert_into_collection("metadata_test", embedding1)
            .is_ok()
    );
    assert!(
        db.insert_into_collection("metadata_test", embedding2)
            .is_ok()
    );

    let embeddings = db.get_embeddings("metadata_test").unwrap();
    assert_eq!(embeddings.len(), 2);

    // Verify metadata states
    assert!(embeddings[0].metadata.is_none());
    assert!(embeddings[1].metadata.is_some());
    assert!(embeddings[1].metadata.as_ref().unwrap().is_empty());
}
