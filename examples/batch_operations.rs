//! Batch Operations Example
//!
//! This example demonstrates efficient batch operations in MemVDB:
//! - Batch insertion of multiple embeddings
//! - Batch updates and modifications
//! - Performance comparison between single and batch operations
//! - Memory-efficient processing of large datasets

use memvdb::{CacheDB, Distance, Embedding};
use std::collections::HashMap;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("📦 Batch Operations Example");
    println!("===========================\n");

    let mut db = CacheDB::new();

    // Create collections for different batch operation demonstrations
    db.create_collection("products".to_string(), 128, Distance::Cosine)?;
    db.create_collection("users".to_string(), 64, Distance::Euclidean)?;

    println!("✅ Created collections for batch operations demo");

    // Demonstrate batch insertion vs individual insertion
    println!("\n⚡ Performance Comparison: Batch vs Individual Operations");
    println!("========================================================");

    compare_insertion_methods(&mut db)?;

    // Demonstrate batch processing of large datasets
    println!("\n📊 Large Dataset Batch Processing");
    println!("=================================");

    process_large_dataset(&mut db)?;

    // Demonstrate batch updates and modifications
    println!("\n🔄 Batch Updates and Modifications");
    println!("==================================");

    demonstrate_batch_updates(&mut db)?;

    // Demonstrate memory-efficient streaming
    println!("\n💾 Memory-Efficient Streaming Operations");
    println!("========================================");

    demonstrate_streaming_operations(&mut db)?;

    // Real-world scenario: E-commerce product catalog
    println!("\n🛒 Real-world Scenario: E-commerce Product Catalog");
    println!("==================================================");

    ecommerce_batch_scenario(&mut db)?;

    println!("\n✅ Batch operations example completed!");
    println!("💡 Batch operations provide significant performance benefits for large datasets.");

    Ok(())
}

fn compare_insertion_methods(db: &mut CacheDB) -> Result<(), Box<dyn std::error::Error>> {
    let collection_name = "performance_test";
    db.create_collection(collection_name.to_string(), 50, Distance::Cosine)?;

    let num_embeddings = 1000;

    // Generate test embeddings
    let embeddings = generate_embeddings(num_embeddings, 50, "perf_test");

    // Method 1: Individual insertions
    println!(
        "🔄 Testing individual insertions ({} embeddings)...",
        num_embeddings
    );
    let start = Instant::now();

    for embedding in &embeddings {
        db.insert_into_collection(collection_name, embedding.clone())?;
    }

    let individual_time = start.elapsed();
    println!("   Individual insertions: {:?}", individual_time);

    // Clear collection for fair comparison
    db.delete_collection(collection_name)?;
    db.create_collection(collection_name.to_string(), 50, Distance::Cosine)?;

    // Method 2: Batch insertion
    println!(
        "🔄 Testing batch insertion ({} embeddings)...",
        num_embeddings
    );
    let start = Instant::now();

    db.update_collection(collection_name, embeddings)?;

    let batch_time = start.elapsed();
    println!("   Batch insertion: {:?}", batch_time);

    // Calculate speedup
    let speedup = individual_time.as_secs_f64() / batch_time.as_secs_f64();
    println!(
        "   🚀 Speedup: {:.2}x faster with batch operations",
        speedup
    );

    // Verify data integrity
    let stored_embeddings = db.get_embeddings(collection_name).unwrap();
    println!(
        "   ✅ Verified: {} embeddings stored correctly",
        stored_embeddings.len()
    );

    Ok(())
}

fn process_large_dataset(db: &mut CacheDB) -> Result<(), Box<dyn std::error::Error>> {
    let collection_name = "large_dataset";
    db.create_collection(collection_name.to_string(), 256, Distance::DotProduct)?;

    let total_embeddings = 10000;
    let batch_size = 500;

    println!(
        "📈 Processing {} embeddings in batches of {}",
        total_embeddings, batch_size
    );

    let start = Instant::now();
    let mut total_processed = 0;

    for batch_num in 0..(total_embeddings / batch_size) {
        // Generate batch
        let batch_embeddings =
            generate_embeddings(batch_size, 256, &format!("batch_{}", batch_num));

        // Insert batch
        let batch_start = Instant::now();
        db.update_collection(collection_name, batch_embeddings)?;
        let batch_time = batch_start.elapsed();

        total_processed += batch_size;

        println!(
            "   Batch {}: {} embeddings processed in {:?} (Total: {})",
            batch_num + 1,
            batch_size,
            batch_time,
            total_processed
        );
    }

    let total_time = start.elapsed();
    let rate = total_embeddings as f64 / total_time.as_secs_f64();

    println!("📊 Processing complete:");
    println!("   Total time: {:?}", total_time);
    println!("   Rate: {:.0} embeddings/second", rate);

    // Verify final count
    let final_count = db.get_embeddings(collection_name).unwrap().len();
    println!("   ✅ Final count: {} embeddings", final_count);

    Ok(())
}

fn demonstrate_batch_updates(db: &mut CacheDB) -> Result<(), Box<dyn std::error::Error>> {
    let collection_name = "update_demo";
    db.create_collection(collection_name.to_string(), 32, Distance::Euclidean)?;

    // Initial batch insertion
    let initial_embeddings = generate_embeddings(100, 32, "initial");
    db.update_collection(collection_name, initial_embeddings)?;
    println!("📥 Inserted initial batch of 100 embeddings");

    // Prepare update batch with new embeddings
    let update_embeddings = generate_embeddings(50, 32, "update");
    let additional_embeddings = generate_embeddings(25, 32, "additional");

    // Combine for batch update
    let mut combined_batch = update_embeddings;
    combined_batch.extend(additional_embeddings);

    println!(
        "🔄 Performing batch update with {} new embeddings",
        combined_batch.len()
    );
    let update_start = Instant::now();
    db.update_collection(collection_name, combined_batch)?;
    let update_time = update_start.elapsed();

    println!("   Update completed in {:?}", update_time);

    // Verify final state
    let final_embeddings = db.get_embeddings(collection_name).unwrap();
    println!(
        "   ✅ Total embeddings after update: {}",
        final_embeddings.len()
    );

    Ok(())
}

fn demonstrate_streaming_operations(db: &mut CacheDB) -> Result<(), Box<dyn std::error::Error>> {
    let collection_name = "streaming_demo";
    db.create_collection(collection_name.to_string(), 64, Distance::Cosine)?;

    println!("🌊 Simulating streaming data processing...");

    let total_streams = 5;
    let embeddings_per_stream = 200;

    for stream_id in 0..total_streams {
        println!(
            "   Processing stream {} of {}",
            stream_id + 1,
            total_streams
        );

        // Simulate processing data in chunks as it arrives
        let chunk_size = 50;
        let mut stream_total = 0;

        for chunk_num in 0..(embeddings_per_stream / chunk_size) {
            // Simulate data arrival delay
            std::thread::sleep(std::time::Duration::from_millis(10));

            let chunk_embeddings = generate_embeddings(
                chunk_size,
                64,
                &format!("stream_{}_{}", stream_id, chunk_num),
            );

            db.update_collection(collection_name, chunk_embeddings)?;
            stream_total += chunk_size;

            print!(
                "     Chunk {}: {} embeddings (Stream total: {})\r",
                chunk_num + 1,
                chunk_size,
                stream_total
            );
        }
        println!();
    }

    let final_count = db.get_embeddings(collection_name).unwrap().len();
    println!(
        "🎯 Streaming complete: {} total embeddings processed",
        final_count
    );

    Ok(())
}

fn ecommerce_batch_scenario(db: &mut CacheDB) -> Result<(), Box<dyn std::error::Error>> {
    println!("🏪 Simulating e-commerce product catalog batch operations...");

    // Create collections for different product categories
    let categories = vec![
        ("electronics", 512, Distance::Cosine),
        ("clothing", 256, Distance::Euclidean),
        ("books", 384, Distance::DotProduct),
    ];

    for (category, dim, metric) in &categories {
        db.create_collection(category.to_string(), *dim, *metric)?;
        println!(
            "   Created {} collection ({} dims, {:?})",
            category, dim, metric
        );
    }

    // Simulate daily product catalog updates
    println!("\n📅 Simulating daily catalog updates...");

    for day in 1..=7 {
        println!("   Day {}: Processing daily updates", day);

        for (category, dim, _) in &categories {
            // Simulate new products for each category
            let new_products = match *category {
                "electronics" => generate_product_embeddings(50, *dim, category, day),
                "clothing" => generate_product_embeddings(75, *dim, category, day),
                "books" => generate_product_embeddings(30, *dim, category, day),
                _ => vec![],
            };

            let start = Instant::now();
            db.update_collection(category, new_products.clone())?;
            let duration = start.elapsed();

            println!(
                "     {}: {} products added in {:?}",
                category,
                new_products.len(),
                duration
            );
        }
    }

    // Final statistics
    println!("\n📊 Final catalog statistics:");
    for (category, _, _) in &categories {
        let count = db.get_embeddings(category).unwrap().len();
        println!("   {}: {} products", category, count);
    }

    // Demonstrate cross-category similarity search
    println!("\n🔍 Cross-category similarity search demo:");
    let electronics_collection = db.get_collection("electronics").unwrap();
    let query_vector = vec![0.5; 512];
    let results = electronics_collection.get_similarity(&query_vector, 3);

    println!("   Top 3 electronics products:");
    for (i, result) in results.iter().enumerate() {
        let product_id = result.embedding.id.get("product_id").unwrap();
        let category = result
            .embedding
            .metadata
            .as_ref()
            .unwrap()
            .get("category")
            .unwrap();
        println!(
            "     {}. {} (Category: {}, Score: {:.4})",
            i + 1,
            product_id,
            category,
            result.score
        );
    }

    Ok(())
}

fn generate_embeddings(count: usize, dimension: usize, prefix: &str) -> Vec<Embedding> {
    (0..count)
        .map(|i| {
            let mut id = HashMap::new();
            id.insert("id".to_string(), format!("{}_{}", prefix, i));

            let mut metadata = HashMap::new();
            metadata.insert("type".to_string(), prefix.to_string());
            metadata.insert("index".to_string(), i.to_string());
            metadata.insert("batch_created".to_string(), "true".to_string());

            let vector: Vec<f32> = (0..dimension)
                .map(|j| ((i * j) as f32 * 0.1).sin())
                .collect();

            Embedding {
                id,
                vector,
                metadata: Some(metadata),
            }
        })
        .collect()
}

fn generate_product_embeddings(
    count: usize,
    dimension: usize,
    category: &str,
    day: u32,
) -> Vec<Embedding> {
    (0..count)
        .map(|i| {
            let mut id = HashMap::new();
            id.insert(
                "product_id".to_string(),
                format!("{}_day{}_{}", category, day, i),
            );

            let mut metadata = HashMap::new();
            metadata.insert("category".to_string(), category.to_string());
            metadata.insert("day_added".to_string(), day.to_string());
            metadata.insert("name".to_string(), format!("{} Product {}", category, i));
            metadata.insert(
                "price".to_string(),
                format!("{:.2}", 10.0 + (i as f64 * 5.0)),
            );
            metadata.insert("in_stock".to_string(), "true".to_string());

            // Generate category-specific embedding patterns
            let vector: Vec<f32> = match category {
                "electronics" => (0..dimension)
                    .map(|j| {
                        let tech_factor = 0.8;
                        (tech_factor * (i * j) as f32 * 0.1).sin() + 0.1
                    })
                    .collect(),
                "clothing" => (0..dimension)
                    .map(|j| {
                        let fashion_factor = 0.6;
                        (fashion_factor * (i + j) as f32 * 0.15).cos() + 0.2
                    })
                    .collect(),
                "books" => (0..dimension)
                    .map(|j| {
                        let knowledge_factor = 0.7;
                        (knowledge_factor * (i * j + day as usize) as f32 * 0.12)
                            .tan()
                            .tanh()
                    })
                    .collect(),
                _ => (0..dimension)
                    .map(|j| ((i * j) as f32 * 0.1).sin())
                    .collect(),
            };

            Embedding {
                id,
                vector,
                metadata: Some(metadata),
            }
        })
        .collect()
}
