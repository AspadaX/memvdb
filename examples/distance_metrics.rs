//! Distance Metrics Comparison Example
//!
//! This example demonstrates the differences between the three distance metrics
//! supported by MemVDB: Euclidean, Cosine, and Dot Product.
//!
//! Features demonstrated:
//! - Creating collections with different distance metrics
//! - Comparing search results across metrics
//! - Understanding when to use each metric
//! - Performance characteristics of each metric

use memvdb::{CacheDB, Distance, Embedding, normalize};
use std::collections::HashMap;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("📏 Distance Metrics Comparison Example");
    println!("======================================\n");

    // Create database with three collections using different metrics
    let mut db = CacheDB::new();

    let collections = vec![
        ("euclidean_docs", Distance::Euclidean),
        ("cosine_docs", Distance::Cosine),
        ("dotproduct_docs", Distance::DotProduct),
    ];

    // Create collections
    for (name, metric) in &collections {
        db.create_collection(name.to_string(), 5, *metric)?;
        println!(
            "✅ Created collection '{}' with {:?} distance",
            name, metric
        );
    }

    // Create sample vectors that demonstrate differences between metrics
    let sample_vectors = vec![
        (
            "vector_a",
            vec![1.0, 0.0, 0.0, 0.0, 0.0],
            "Unit vector along X-axis",
        ),
        (
            "vector_b",
            vec![0.0, 1.0, 0.0, 0.0, 0.0],
            "Unit vector along Y-axis",
        ),
        (
            "vector_c",
            vec![2.0, 0.0, 0.0, 0.0, 0.0],
            "Scaled vector along X-axis",
        ),
        ("vector_d", vec![1.0, 1.0, 0.0, 0.0, 0.0], "Diagonal vector"),
        (
            "vector_e",
            vec![0.5, 0.5, 0.0, 0.0, 0.0],
            "Smaller diagonal vector",
        ),
        (
            "vector_f",
            vec![-1.0, 0.0, 0.0, 0.0, 0.0],
            "Negative X-axis vector",
        ),
        (
            "vector_g",
            vec![0.1, 0.1, 0.1, 0.1, 0.1],
            "Small uniform vector",
        ),
    ];

    println!("\n📊 Sample Vectors:");
    for (name, vector, description) in &sample_vectors {
        println!("  {}: {:?} - {}", name, vector, description);
    }

    // Insert vectors into all collections
    println!("\n📝 Inserting vectors into all collections...");
    for (collection_name, _) in &collections {
        for (vector_name, vector, description) in &sample_vectors {
            let mut id = HashMap::new();
            id.insert("vector_id".to_string(), vector_name.to_string());

            let mut metadata = HashMap::new();
            metadata.insert("description".to_string(), description.to_string());
            metadata.insert(
                "magnitude".to_string(),
                format!(
                    "{:.3}",
                    vector.iter().map(|&x: &f32| x.powi(2)).sum::<f32>().sqrt()
                ),
            );

            let embedding = Embedding {
                id,
                vector: vector.clone(),
                metadata: Some(metadata),
            };

            db.insert_into_collection(collection_name, embedding)?;
        }
        println!(
            "  ✅ Inserted {} vectors into {}",
            sample_vectors.len(),
            collection_name
        );
    }

    // Query vector for comparison
    let query_vector = vec![1.0, 0.5, 0.0, 0.0, 0.0];
    println!("\n🔍 Query Vector: {:?}", query_vector);
    println!("Query description: Similar to vector_a but with some Y component");

    // Perform similarity search with each metric
    println!("\n📋 Similarity Search Results");
    println!("============================");

    for (collection_name, metric) in &collections {
        let collection = db.get_collection(collection_name).unwrap();
        let results = collection.get_similarity(&query_vector, 5);

        println!(
            "\n🎯 {} Distance Results:",
            format!("{:?}", metric).to_uppercase()
        );
        println!("Collection: {}", collection_name);

        for (rank, result) in results.iter().enumerate() {
            let vector_id = result.embedding.id.get("vector_id").unwrap();
            let description = result
                .embedding
                .metadata
                .as_ref()
                .unwrap()
                .get("description")
                .unwrap();
            let magnitude = result
                .embedding
                .metadata
                .as_ref()
                .unwrap()
                .get("magnitude")
                .unwrap();

            println!("  {}. {} (Score: {:.4})", rank + 1, vector_id, result.score);
            println!("     Vector: {:?}", result.embedding.vector);
            println!("     Description: {}", description);
            println!("     Magnitude: {}", magnitude);
            println!();
        }
    }

    // Detailed analysis of metric characteristics
    println!("🔬 Detailed Metric Analysis");
    println!("===========================");

    analyze_metric_properties();

    // Performance comparison
    println!("\n⚡ Performance Comparison");
    println!("========================");

    let test_vectors = generate_test_vectors(1000, 100);
    compare_performance(&test_vectors)?;

    // Use case recommendations
    println!("\n💡 Use Case Recommendations");
    println!("===========================");

    print_use_case_guide();

    // Demonstrate normalization effects
    println!("\n🔄 Vector Normalization Effects");
    println!("===============================");

    demonstrate_normalization();

    println!("\n✅ Distance metrics comparison completed!");
    println!("💡 Choose your distance metric based on your data characteristics and use case.");

    Ok(())
}

fn analyze_metric_properties() {
    println!("\n📊 Metric Properties Analysis:");

    let test_cases = vec![
        (vec![1.0, 0.0], vec![0.0, 1.0], "Orthogonal unit vectors"),
        (
            vec![1.0, 0.0],
            vec![2.0, 0.0],
            "Parallel vectors (different magnitude)",
        ),
        (
            vec![1.0, 1.0],
            vec![2.0, 2.0],
            "Parallel vectors (same direction)",
        ),
        (
            vec![1.0, 0.0],
            vec![-1.0, 0.0],
            "Opposite direction vectors",
        ),
        (
            vec![1.0, 1.0],
            vec![1.0, -1.0],
            "Perpendicular non-unit vectors",
        ),
    ];

    for (vec_a, vec_b, description) in test_cases {
        println!("\n  Case: {}", description);
        println!("    Vector A: {:?}", vec_a);
        println!("    Vector B: {:?}", vec_b);

        // Calculate Euclidean distance manually
        let euclidean_dist = vec_a
            .iter()
            .zip(&vec_b)
            .map(|(&a, &b): (&f32, &f32)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();

        // Calculate dot product
        let dot_product = vec_a
            .iter()
            .zip(&vec_b)
            .map(|(&a, &b): (&f32, &f32)| a * b)
            .sum::<f32>();

        // Calculate cosine similarity
        let mag_a = vec_a.iter().map(|&x: &f32| x.powi(2)).sum::<f32>().sqrt();
        let mag_b = vec_b.iter().map(|&x: &f32| x.powi(2)).sum::<f32>().sqrt();
        let cosine_sim = if mag_a > 0.0 && mag_b > 0.0 {
            dot_product / (mag_a * mag_b)
        } else {
            0.0
        };

        println!("    Euclidean distance: {:.4}", euclidean_dist);
        println!("    Dot product: {:.4}", dot_product);
        println!("    Cosine similarity: {:.4}", cosine_sim);
    }
}

fn compare_performance(
    test_vectors: &[(Vec<f32>, Vec<f32>)],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut db = CacheDB::new();

    // Create collections for performance testing
    db.create_collection("perf_euclidean".to_string(), 100, Distance::Euclidean)?;
    db.create_collection("perf_cosine".to_string(), 100, Distance::Cosine)?;
    db.create_collection("perf_dotproduct".to_string(), 100, Distance::DotProduct)?;

    // Insert test vectors
    for (i, (vector, _)) in test_vectors.iter().enumerate().take(500) {
        let mut id = HashMap::new();
        id.insert("id".to_string(), i.to_string());

        let embedding = Embedding {
            id,
            vector: vector.clone(),
            metadata: None,
        };

        db.insert_into_collection("perf_euclidean", embedding.clone())?;
        db.insert_into_collection("perf_cosine", embedding.clone())?;
        db.insert_into_collection("perf_dotproduct", embedding)?;
    }

    let collections = vec![
        ("perf_euclidean", Distance::Euclidean),
        ("perf_cosine", Distance::Cosine),
        ("perf_dotproduct", Distance::DotProduct),
    ];

    println!("\nPerformance test with 500 vectors, 100 queries each:");

    for (collection_name, metric) in collections {
        let collection = db.get_collection(&collection_name).unwrap();
        let start = Instant::now();

        // Perform 100 similarity searches
        for (_, query_vector) in test_vectors.iter().take(100) {
            let _ = collection.get_similarity(query_vector, 10);
        }

        let duration = start.elapsed();
        let avg_time = duration.as_micros() / 100;

        println!(
            "  {:?}: {:?} total, {}μs average per query",
            metric, duration, avg_time
        );
    }

    Ok(())
}

fn generate_test_vectors(num_pairs: usize, dimension: usize) -> Vec<(Vec<f32>, Vec<f32>)> {
    (0..num_pairs)
        .map(|i| {
            let vec1: Vec<f32> = (0..dimension)
                .map(|j| ((i * j) as f32 * 0.1).sin())
                .collect();
            let vec2: Vec<f32> = (0..dimension)
                .map(|j| ((i * j + 1) as f32 * 0.1).cos())
                .collect();
            (vec1, vec2)
        })
        .collect()
}

fn print_use_case_guide() {
    println!("\n📚 Distance Metric Selection Guide:");

    println!("\n  🎯 EUCLIDEAN DISTANCE");
    println!("    Best for:");
    println!("    • Spatial data (coordinates, positions)");
    println!("    • When absolute magnitude matters");
    println!("    • Low-dimensional dense vectors");
    println!("    • Computer vision features (sometimes)");
    println!("    Characteristics:");
    println!("    • Sensitive to vector magnitude");
    println!("    • Measures geometric distance");
    println!("    • Range: [0, ∞)");

    println!("\n  🎯 COSINE SIMILARITY");
    println!("    Best for:");
    println!("    • Text embeddings and NLP");
    println!("    • High-dimensional sparse vectors");
    println!("    • When direction matters more than magnitude");
    println!("    • Document similarity, user preferences");
    println!("    Characteristics:");
    println!("    • Ignores vector magnitude");
    println!("    • Measures angle between vectors");
    println!("    • Range: [-1, 1] (often converted to [0, 2])");

    println!("\n  🎯 DOT PRODUCT");
    println!("    Best for:");
    println!("    • Pre-normalized vectors");
    println!("    • Neural network outputs");
    println!("    • When both direction and magnitude matter");
    println!("    • Recommendation systems");
    println!("    Characteristics:");
    println!("    • Considers both angle and magnitude");
    println!("    • Most computationally efficient");
    println!("    • Range: (-∞, ∞)");
}

fn demonstrate_normalization() {
    let original_vectors = vec![
        vec![1.0, 0.0, 0.0],
        vec![2.0, 0.0, 0.0],
        vec![10.0, 0.0, 0.0],
        vec![1.0, 1.0, 0.0],
        vec![3.0, 4.0, 0.0],
    ];

    println!("\nNormalization effects on distance calculations:");
    println!("Original Vector → Normalized Vector (Magnitude)");

    for (i, vector) in original_vectors.iter().enumerate() {
        let normalized = normalize(vector);
        let original_mag = vector.iter().map(|&x: &f32| x.powi(2)).sum::<f32>().sqrt();
        let normalized_mag = normalized
            .iter()
            .map(|&x: &f32| x.powi(2))
            .sum::<f32>()
            .sqrt();

        println!(
            "  Vector {}: {:?} → {:?} ({:.3} → {:.3})",
            i + 1,
            vector,
            normalized
                .iter()
                .map(|x| format!("{:.3}", x))
                .collect::<Vec<_>>(),
            original_mag,
            normalized_mag
        );
    }

    println!("\nKey insight: Cosine similarity automatically normalizes vectors");
    println!("internally, making it equivalent to dot product on unit vectors.");
}
