use memvdb::*;
use std::collections::HashMap;
use std::time::Instant;

#[test]
fn test_large_scale_insertion_performance() {
    let mut db = CacheDB::new();
    db.create_collection("perf_test".to_string(), 128, Distance::Euclidean)
        .unwrap();

    let start = Instant::now();

    // Insert 10,000 embeddings
    for i in 0..10_000 {
        let mut id = HashMap::new();
        id.insert("id".to_string(), i.to_string());

        let vector: Vec<f32> = (0..128).map(|j| (i * j) as f32 / 1000.0).collect();

        let embedding = Embedding {
            id,
            vector,
            metadata: None,
        };

        db.insert_into_collection("perf_test", embedding).unwrap();
    }

    let duration = start.elapsed();
    println!("Inserted 10,000 embeddings in {:?}", duration);

    // Verify all embeddings were inserted
    let embeddings = db.get_embeddings("perf_test").unwrap();
    assert_eq!(embeddings.len(), 10_000);

    // Should complete within reasonable time (adjust threshold as needed)
    assert!(duration.as_secs() < 120);
}

#[test]
fn test_similarity_search_performance() {
    let mut db = CacheDB::new();
    db.create_collection("similarity_perf".to_string(), 256, Distance::Cosine)
        .unwrap();

    // Insert 5,000 embeddings
    for i in 0..5_000 {
        let mut id = HashMap::new();
        id.insert("id".to_string(), i.to_string());

        let vector: Vec<f32> = (0..256).map(|j| ((i + j) as f32).sin()).collect();

        let embedding = Embedding {
            id,
            vector,
            metadata: None,
        };

        db.insert_into_collection("similarity_perf", embedding)
            .unwrap();
    }

    let collection = db.get_collection("similarity_perf").unwrap();
    let query_vector: Vec<f32> = (0..256).map(|i| (i as f32).cos()).collect();

    let start = Instant::now();

    // Perform 100 similarity searches
    for _ in 0..100 {
        let _results = collection.get_similarity(&query_vector, 10);
    }

    let duration = start.elapsed();
    println!("Performed 100 similarity searches in {:?}", duration);

    // Each search should be reasonably fast
    let avg_time_per_search = duration.as_millis() / 100;
    assert!(avg_time_per_search < 500); // Less than 500ms per search
}

#[test]
fn test_batch_insertion_performance() {
    let mut db = CacheDB::new();
    db.create_collection("batch_perf".to_string(), 64, Distance::DotProduct)
        .unwrap();

    // Create 1,000 embeddings
    let mut embeddings = Vec::with_capacity(1_000);
    for i in 0..1_000 {
        let mut id = HashMap::new();
        id.insert("batch_id".to_string(), i.to_string());

        let vector: Vec<f32> = (0..64).map(|j| ((i * j) as f32 / 100.0).tanh()).collect();

        embeddings.push(Embedding {
            id,
            vector,
            metadata: None,
        });
    }

    let start = Instant::now();
    db.update_collection("batch_perf", embeddings).unwrap();
    let duration = start.elapsed();

    println!("Batch inserted 1,000 embeddings in {:?}", duration);

    let stored_embeddings = db.get_embeddings("batch_perf").unwrap();
    assert_eq!(stored_embeddings.len(), 1_000);

    // Batch insertion should be faster than individual insertions
    assert!(duration.as_millis() < 5000);
}

#[test]
fn test_memory_efficiency() {
    let mut db = CacheDB::new();

    // Test creating multiple collections
    for i in 0..10 {
        let collection_name = format!("memory_test_{}", i);
        db.create_collection(collection_name.clone(), 32, Distance::Euclidean)
            .unwrap();

        // Add some embeddings to each collection
        for j in 0..100 {
            let mut id = HashMap::new();
            id.insert("id".to_string(), format!("{}_{}", i, j));

            let vector: Vec<f32> = (0..32).map(|k| (i * j * k) as f32 / 1000.0).collect();

            let embedding = Embedding {
                id,
                vector,
                metadata: None,
            };

            db.insert_into_collection(&collection_name, embedding)
                .unwrap();
        }
    }

    // Verify all collections and embeddings exist
    assert_eq!(db.collections.len(), 10);

    for i in 0..10 {
        let collection_name = format!("memory_test_{}", i);
        let embeddings = db.get_embeddings(&collection_name).unwrap();
        assert_eq!(embeddings.len(), 100);
    }
}

#[test]
fn test_distance_function_performance() {
    let vec1: Vec<f32> = (0..1000).map(|i| i as f32 / 1000.0).collect();
    let vec2: Vec<f32> = (0..1000).map(|i| (i as f32 / 1000.0).sin()).collect();

    // Test Euclidean distance performance
    let euclidean_fn = get_distance_fn(Distance::Euclidean);
    let start = Instant::now();
    for _ in 0..10_000 {
        let _dist = euclidean_fn(&vec1, &vec2, 0.0);
    }
    let euclidean_time = start.elapsed();

    // Test Cosine distance performance
    let cosine_fn = get_distance_fn(Distance::Cosine);
    let start = Instant::now();
    for _ in 0..10_000 {
        let _dist = cosine_fn(&vec1, &vec2, 0.0);
    }
    let cosine_time = start.elapsed();

    // Test Dot product performance
    let dot_fn = get_distance_fn(Distance::DotProduct);
    let start = Instant::now();
    for _ in 0..10_000 {
        let _dist = dot_fn(&vec1, &vec2, 0.0);
    }
    let dot_time = start.elapsed();

    println!(
        "Euclidean: {:?}, Cosine: {:?}, Dot: {:?}",
        euclidean_time, cosine_time, dot_time
    );

    // All distance functions should complete within reasonable time
    assert!(euclidean_time.as_millis() < 5000);
    assert!(cosine_time.as_millis() < 5000);
    assert!(dot_time.as_millis() < 5000);
}

#[test]
fn test_normalization_performance() {
    let large_vector: Vec<f32> = (0..10_000).map(|i| (i as f32).sin()).collect();

    let start = Instant::now();
    for _ in 0..1_000 {
        let _normalized = normalize(&large_vector);
    }
    let duration = start.elapsed();

    println!("Normalized 1,000 vectors of size 10,000 in {:?}", duration);
    assert!(duration.as_millis() < 10000);
}

#[test]
fn test_concurrent_read_performance() {
    use std::sync::{Arc, Mutex};
    use std::thread;

    let mut db = CacheDB::new();
    db.create_collection("concurrent_read_test".to_string(), 128, Distance::Euclidean)
        .unwrap();

    // Insert 1,000 embeddings
    for i in 0..1_000 {
        let mut id = HashMap::new();
        id.insert("id".to_string(), i.to_string());

        let vector: Vec<f32> = (0..128).map(|j| (i * j) as f32 / 1000.0).collect();

        let embedding = Embedding {
            id,
            vector,
            metadata: None,
        };

        db.insert_into_collection("concurrent_read_test", embedding)
            .unwrap();
    }

    let db = Arc::new(Mutex::new(db));
    let mut handles = vec![];

    let start = Instant::now();

    // Spawn 4 threads to perform concurrent reads
    for _ in 0..4 {
        let db_clone = Arc::clone(&db);
        let handle = thread::spawn(move || {
            for _ in 0..100 {
                let db_lock = db_clone.lock().unwrap();
                let _embeddings = db_lock.get_embeddings("concurrent_read_test");
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    let duration = start.elapsed();
    println!("Concurrent reads completed in {:?}", duration);
    assert!(duration.as_secs() < 30);
}

#[test]
fn test_similarity_search_scaling() {
    let dimensions = vec![32, 64, 128, 256, 512];
    let num_embeddings = 1_000;
    let num_searches = 10;

    for dim in dimensions {
        let mut db = CacheDB::new();
        let collection_name = format!("scaling_test_{}", dim);
        db.create_collection(collection_name.clone(), dim, Distance::Cosine)
            .unwrap();

        // Insert embeddings
        for i in 0..num_embeddings {
            let mut id = HashMap::new();
            id.insert("id".to_string(), i.to_string());

            let vector: Vec<f32> = (0..dim).map(|j| ((i + j) as f32).sin()).collect();

            let embedding = Embedding {
                id,
                vector,
                metadata: None,
            };

            db.insert_into_collection(&collection_name, embedding)
                .unwrap();
        }

        let collection = db.get_collection(&collection_name).unwrap();
        let query_vector: Vec<f32> = (0..dim).map(|i| (i as f32).cos()).collect();

        let start = Instant::now();
        for _ in 0..num_searches {
            let _results = collection.get_similarity(&query_vector, 10);
        }
        let duration = start.elapsed();

        let avg_time = duration.as_millis() / num_searches as u128;
        println!("Dimension {}: average search time {}ms", dim, avg_time);

        // Search time should scale reasonably with dimension
        assert!(avg_time < dim as u128 * 10); // Rough heuristic
    }
}

#[test]
fn test_cache_attr_performance() {
    let vector: Vec<f32> = (0..1000).map(|i| (i as f32).sin()).collect();

    let start = Instant::now();
    for _ in 0..10_000 {
        let _cache = get_cache_attr(Distance::Cosine, &vector);
    }
    let duration = start.elapsed();

    println!(
        "Cache attribute calculation for 10,000 iterations: {:?}",
        duration
    );
    assert!(duration.as_millis() < 5000);
}
