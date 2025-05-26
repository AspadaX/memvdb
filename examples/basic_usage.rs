//! Basic usage example for MemVDB
//!
//! This example demonstrates the fundamental operations of the MemVDB library:
//! - Creating a database and collection
//! - Inserting embeddings with metadata
//! - Performing similarity searches
//! - Working with different distance metrics

use memvdb::{CacheDB, Distance, Embedding};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 MemVDB Basic Usage Example");
    println!("===============================\n");

    // Create a new in-memory vector database
    let mut db = CacheDB::new();
    // // You may also load an existing db
    // let mut db = CacheDB::load("./db.json")?;
    println!("✅ Created new MemVDB instance");

    // Create a collection for document embeddings
    // Using 384-dimensional vectors with cosine similarity (common for text embeddings)
    let collection_name = "documents".to_string();
    let dimension = 384;
    let distance_metric = Distance::Cosine;

    db.create_collection(collection_name.clone(), dimension, distance_metric)?;
    println!(
        "✅ Created collection '{}' with {} dimensions using {:?} distance",
        collection_name, dimension, distance_metric
    );

    // Create sample embeddings representing documents
    let sample_documents = vec![
        (
            "doc1",
            "Artificial Intelligence and Machine Learning",
            vec![0.1, 0.2, 0.3],
        ),
        (
            "doc2",
            "Deep Learning Neural Networks",
            vec![0.15, 0.25, 0.35],
        ),
        ("doc3", "Cooking Recipes and Food", vec![0.8, 0.1, 0.05]),
        ("doc4", "Travel Guide to Europe", vec![0.05, 0.9, 0.1]),
        (
            "doc5",
            "Machine Learning Algorithms",
            vec![0.12, 0.22, 0.32],
        ),
    ];

    // Insert embeddings into the collection
    println!("\n📝 Inserting sample documents...");
    for (doc_id, title, mut vector) in sample_documents {
        // Extend vector to match collection dimension
        vector.resize(dimension, 0.0);

        // Create unique ID
        let mut id = HashMap::new();
        id.insert("document_id".to_string(), doc_id.to_string());

        // Create metadata
        let mut metadata = HashMap::new();
        metadata.insert("title".to_string(), title.to_string());
        metadata.insert(
            "category".to_string(),
            if title.contains("Learning") || title.contains("AI") {
                "Technology".to_string()
            } else if title.contains("Cooking") || title.contains("Food") {
                "Food".to_string()
            } else {
                "Travel".to_string()
            },
        );
        metadata.insert("indexed_at".to_string(), "2024-01-01".to_string());

        // Create and insert embedding
        let embedding = Embedding {
            id,
            vector,
            metadata: Some(metadata),
        };

        db.insert_into_collection(&collection_name, embedding)?;
        println!("  📄 Inserted: {} - {}", doc_id, title);
    }

    // Get collection for similarity search
    let collection = db
        .get_collection(&collection_name)
        .ok_or("Collection not found")?;

    println!("\n📊 Collection statistics:");
    println!("  - Name: {}", collection_name);
    println!("  - Dimension: {}", collection.dimension);
    println!("  - Distance metric: {:?}", collection.distance);
    println!("  - Number of embeddings: {}", collection.embeddings.len());

    // Perform similarity search
    println!("\n🔍 Performing similarity search...");

    // Query for AI/ML related content
    let mut query_vector = vec![0.11, 0.21, 0.31]; // Similar to AI/ML documents
    query_vector.resize(dimension, 0.0);

    println!("Query: Looking for AI/Machine Learning related documents");
    let results = collection.get_similarity(&query_vector, 3);

    println!("\n📋 Top 3 similar documents:");
    for (rank, result) in results.iter().enumerate() {
        let doc_id = result
            .embedding
            .id
            .get("document_id")
            .map(|s| s.as_str())
            .unwrap_or("unknown");
        let title = result
            .embedding
            .metadata
            .as_ref()
            .and_then(|m| m.get("title"))
            .map(|s| s.as_str())
            .unwrap_or("No title");
        let category = result
            .embedding
            .metadata
            .as_ref()
            .and_then(|m| m.get("category"))
            .map(|s| s.as_str())
            .unwrap_or("Unknown");

        println!(
            "  {}. Document: {} (Score: {:.4})",
            rank + 1,
            doc_id,
            result.score
        );
        println!("     Title: {}", title);
        println!("     Category: {}", category);
        println!();
    }

    // Demonstrate different queries
    println!("🔍 Performing another search...");
    let mut food_query = vec![0.75, 0.15, 0.1]; // Similar to food-related content
    food_query.resize(dimension, 0.0);

    println!("Query: Looking for food/cooking related documents");
    let food_results = collection.get_similarity(&food_query, 2);

    println!("\n📋 Top 2 food-related documents:");
    for (rank, result) in food_results.iter().enumerate() {
        let doc_id = result
            .embedding
            .id
            .get("document_id")
            .map(|s| s.as_str())
            .unwrap_or("unknown");
        let title = result
            .embedding
            .metadata
            .as_ref()
            .and_then(|m| m.get("title"))
            .map(|s| s.as_str())
            .unwrap_or("No title");

        println!(
            "  {}. Document: {} (Score: {:.4})",
            rank + 1,
            doc_id,
            result.score
        );
        println!("     Title: {}", title);
        println!();
    }

    // Show all embeddings in the collection
    let all_embeddings = db.get_embeddings(&collection_name).unwrap();
    println!("📚 All documents in collection:");
    for embedding in all_embeddings {
        let doc_id = embedding
            .id
            .get("document_id")
            .map(|s| s.as_str())
            .unwrap_or("unknown");
        let title = embedding
            .metadata
            .as_ref()
            .and_then(|m| m.get("title"))
            .map(|s| s.as_str())
            .unwrap_or("No title");
        let category = embedding
            .metadata
            .as_ref()
            .and_then(|m| m.get("category"))
            .map(|s| s.as_str())
            .unwrap_or("Unknown");

        println!("  - {}: {} [{}]", doc_id, title, category);
    }

    println!("\n✅ Basic usage example completed successfully!");
    println!("💡 Try modifying the query vectors or adding more documents to experiment further.");
    
    // // You may also save the database to the disk
    // db.save("./");
    
    Ok(())
}
