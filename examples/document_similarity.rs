//! Document Similarity Search Example
//!
//! This example demonstrates how to use MemVDB for document similarity search,
//! a common use case in information retrieval and recommendation systems.
//!
//! Features demonstrated:
//! - Batch insertion of document embeddings
//! - Similarity search with different query types
//! - Filtering results by metadata
//! - Performance measurement

use memvdb::{CacheDB, Distance, Embedding};
use std::collections::HashMap;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("📚 Document Similarity Search Example");
    println!("=====================================\n");

    // Create database and collection
    let mut db = CacheDB::new();
    let collection_name = "research_papers".to_string();

    // Using 768 dimensions (common for transformer-based embeddings like BERT)
    db.create_collection(collection_name.clone(), 768, Distance::Cosine)?;
    println!("✅ Created collection for research papers with 768-dimensional embeddings");

    // Sample research papers with simulated embeddings
    let papers = vec![
        Paper {
            id: "paper_001",
            title: "Attention Is All You Need",
            authors: vec!["Vaswani", "Shazeer", "Parmar"],
            abstract_text: "A novel neural network architecture based entirely on attention mechanisms",
            category: "Deep Learning",
            year: 2017,
            citations: 50000,
            // Simulated embedding focusing on attention/transformers
            embedding: create_themed_vector(768, &[0.8, 0.7, 0.6, 0.5]),
        },
        Paper {
            id: "paper_002",
            title: "BERT: Pre-training of Deep Bidirectional Transformers",
            authors: vec!["Devlin", "Chang", "Lee"],
            abstract_text: "Bidirectional encoder representations from transformers for language understanding",
            category: "Natural Language Processing",
            year: 2018,
            citations: 45000,
            embedding: create_themed_vector(768, &[0.75, 0.8, 0.65, 0.55]),
        },
        Paper {
            id: "paper_003",
            title: "ResNet: Deep Residual Learning for Image Recognition",
            authors: vec!["He", "Zhang", "Ren"],
            abstract_text: "Deep residual networks for image classification with skip connections",
            category: "Computer Vision",
            year: 2015,
            citations: 60000,
            embedding: create_themed_vector(768, &[0.2, 0.9, 0.8, 0.1]),
        },
        Paper {
            id: "paper_004",
            title: "Generative Adversarial Networks",
            authors: vec!["Goodfellow", "Pouget-Abadie", "Mirza"],
            abstract_text: "Framework for training generative models via adversarial training",
            category: "Deep Learning",
            year: 2014,
            citations: 40000,
            embedding: create_themed_vector(768, &[0.6, 0.3, 0.9, 0.7]),
        },
        Paper {
            id: "paper_005",
            title: "You Only Look Once: Unified Real-Time Object Detection",
            authors: vec!["Redmon", "Divvala", "Girshick"],
            abstract_text: "Real-time object detection with a single neural network",
            category: "Computer Vision",
            year: 2016,
            citations: 35000,
            embedding: create_themed_vector(768, &[0.1, 0.85, 0.9, 0.2]),
        },
        Paper {
            id: "paper_006",
            title: "Word2Vec: Efficient Estimation of Word Representations",
            authors: vec!["Mikolov", "Chen", "Corrado"],
            abstract_text: "Learning word embeddings using continuous bag of words and skip-gram",
            category: "Natural Language Processing",
            year: 2013,
            citations: 55000,
            embedding: create_themed_vector(768, &[0.9, 0.6, 0.4, 0.8]),
        },
        Paper {
            id: "paper_007",
            title: "AlexNet: ImageNet Classification with Deep CNNs",
            authors: vec!["Krizhevsky", "Sutskever", "Hinton"],
            abstract_text: "Deep convolutional neural networks for large-scale image classification",
            category: "Computer Vision",
            year: 2012,
            citations: 70000,
            embedding: create_themed_vector(768, &[0.15, 0.95, 0.7, 0.25]),
        },
        Paper {
            id: "paper_008",
            title: "GPT-3: Language Models are Few-Shot Learners",
            authors: vec!["Brown", "Mann", "Ryder"],
            abstract_text: "Large-scale language model demonstrating few-shot learning capabilities",
            category: "Natural Language Processing",
            year: 2020,
            citations: 25000,
            embedding: create_themed_vector(768, &[0.85, 0.75, 0.5, 0.9]),
        },
    ];

    // Batch insert papers
    println!("📝 Inserting {} research papers...", papers.len());
    let start_time = Instant::now();

    let embeddings: Vec<Embedding> = papers
        .iter()
        .map(|paper| {
            let mut id = HashMap::new();
            id.insert("paper_id".to_string(), paper.id.to_string());

            let mut metadata = HashMap::new();
            metadata.insert("title".to_string(), paper.title.to_string());
            metadata.insert("authors".to_string(), paper.authors.join(", "));
            metadata.insert("abstract".to_string(), paper.abstract_text.to_string());
            metadata.insert("category".to_string(), paper.category.to_string());
            metadata.insert("year".to_string(), paper.year.to_string());
            metadata.insert("citations".to_string(), paper.citations.to_string());

            Embedding {
                id,
                vector: paper.embedding.clone(),
                metadata: Some(metadata),
            }
        })
        .collect();

    db.update_collection(&collection_name, embeddings)?;
    let insert_time = start_time.elapsed();
    println!("✅ Inserted {} papers in {:?}", papers.len(), insert_time);

    let collection = db.get_collection(&collection_name).unwrap();

    // Demonstrate various similarity searches
    println!("\n🔍 Similarity Search Examples");
    println!("==============================");

    // Search 1: Find papers similar to transformer/attention research
    println!("\n1. 🎯 Query: Papers similar to transformer/attention mechanisms");
    let transformer_query = create_themed_vector(768, &[0.8, 0.7, 0.6, 0.5]);
    let results = collection.get_similarity(&transformer_query, 3);
    print_search_results(&results, "Transformer-related papers");

    // Search 2: Find computer vision papers
    println!("\n2. 🎯 Query: Computer vision research papers");
    let cv_query = create_themed_vector(768, &[0.2, 0.9, 0.8, 0.1]);
    let results = collection.get_similarity(&cv_query, 3);
    print_search_results(&results, "Computer vision papers");

    // Search 3: Find generative model papers
    println!("\n3. 🎯 Query: Generative modeling papers");
    let generative_query = create_themed_vector(768, &[0.6, 0.3, 0.9, 0.7]);
    let results = collection.get_similarity(&generative_query, 3);
    print_search_results(&results, "Generative model papers");

    // Demonstrate filtering by metadata
    println!("\n📊 Metadata-based Analysis");
    println!("==========================");

    analyze_by_category(&db, &collection_name);
    analyze_by_year(&db, &collection_name);
    analyze_by_citations(&db, &collection_name);

    // Performance analysis
    println!("\n⚡ Performance Analysis");
    println!("======================");

    let start = Instant::now();
    let query = create_themed_vector(768, &[0.5, 0.5, 0.5, 0.5]);
    for _ in 0..100 {
        let _ = collection.get_similarity(&query, 5);
    }
    let avg_time = start.elapsed() / 100;
    println!("Average search time (100 queries): {:?}", avg_time);

    println!("\n✅ Document similarity example completed!");
    println!("💡 This example shows how MemVDB can be used for academic paper search,");
    println!("   recommendation systems, and content discovery applications.");

    Ok(())
}

struct Paper {
    id: &'static str,
    title: &'static str,
    authors: Vec<&'static str>,
    abstract_text: &'static str,
    category: &'static str,
    year: u32,
    citations: u32,
    embedding: Vec<f32>,
}

fn create_themed_vector(dimension: usize, theme: &[f32]) -> Vec<f32> {
    let mut vector = vec![0.0; dimension];

    // Set the first few dimensions to the theme values
    for (i, &value) in theme.iter().enumerate() {
        if i < dimension {
            vector[i] = value;
        }
    }

    // Fill the rest with random-like values based on theme
    for i in theme.len()..dimension {
        let base = theme[i % theme.len()];
        let noise = (i as f32 * 0.1).sin() * 0.1;
        vector[i] = (base + noise).max(0.0).min(1.0);
    }

    vector
}

fn print_search_results(results: &[memvdb::SimilarityResult], query_description: &str) {
    println!("📋 {} (Top {} results):", query_description, results.len());

    for (rank, result) in results.iter().enumerate() {
        let paper_id = result.embedding.id.get("paper_id").unwrap();
        let title = result
            .embedding
            .metadata
            .as_ref()
            .unwrap()
            .get("title")
            .unwrap();
        let authors = result
            .embedding
            .metadata
            .as_ref()
            .unwrap()
            .get("authors")
            .unwrap();
        let category = result
            .embedding
            .metadata
            .as_ref()
            .unwrap()
            .get("category")
            .unwrap();
        let year = result
            .embedding
            .metadata
            .as_ref()
            .unwrap()
            .get("year")
            .unwrap();

        println!("  {}. {} (Score: {:.4})", rank + 1, paper_id, result.score);
        println!("     📖 {}", title);
        println!("     👥 Authors: {}", authors);
        println!("     🏷️  Category: {} | Year: {}", category, year);
        println!();
    }
}

fn analyze_by_category(db: &CacheDB, collection_name: &str) {
    let embeddings = db.get_embeddings(collection_name).unwrap();
    let mut categories = HashMap::new();

    for embedding in embeddings {
        if let Some(metadata) = &embedding.metadata {
            if let Some(category) = metadata.get("category") {
                *categories.entry(category.clone()).or_insert(0) += 1;
            }
        }
    }

    println!("📂 Papers by category:");
    for (category, count) in categories {
        println!("  - {}: {} papers", category, count);
    }
}

fn analyze_by_year(db: &CacheDB, collection_name: &str) {
    let embeddings = db.get_embeddings(collection_name).unwrap();
    let mut years = HashMap::new();

    for embedding in embeddings {
        if let Some(metadata) = &embedding.metadata {
            if let Some(year) = metadata.get("year") {
                *years.entry(year.clone()).or_insert(0) += 1;
            }
        }
    }

    println!("\n📅 Papers by year:");
    let mut year_vec: Vec<_> = years.into_iter().collect();
    year_vec.sort_by_key(|(year, _)| year.parse::<u32>().unwrap_or(0));

    for (year, count) in year_vec {
        println!("  - {}: {} papers", year, count);
    }
}

fn analyze_by_citations(db: &CacheDB, collection_name: &str) {
    let embeddings = db.get_embeddings(collection_name).unwrap();
    let mut papers_with_citations: Vec<_> = embeddings
        .iter()
        .filter_map(|e| {
            let metadata = e.metadata.as_ref()?;
            let title = metadata.get("title")?;
            let citations = metadata.get("citations")?.parse::<u32>().ok()?;
            Some((title.clone(), citations))
        })
        .collect();

    papers_with_citations.sort_by_key(|(_, citations)| std::cmp::Reverse(*citations));

    println!("\n📈 Most cited papers:");
    for (title, citations) in papers_with_citations.into_iter().take(3) {
        println!("  - {} ({} citations)", title, citations);
    }
}
