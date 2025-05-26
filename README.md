# MemVDB - In-Memory Vector Database

[![Crates.io](https://img.shields.io/crates/v/memvdb.svg)](https://crates.io/crates/memvdb)
[![Documentation](https://docs.rs/memvdb/badge.svg)](https://docs.rs/memvdb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

MemVDB is a fast, lightweight in-memory vector database written in Rust. It provides efficient similarity search capabilities with support for multiple distance metrics, making it ideal for machine learning applications, recommendation systems, and semantic search.

## 🚀 Features

- **Multiple Distance Metrics**: Euclidean, Cosine, and Dot Product similarity
- **High Performance**: Optimized similarity search with binary heap algorithms
- **Flexible Metadata**: Store arbitrary metadata with each embedding
- **Batch Operations**: Efficient batch insertion and updates
- **Thread Safety**: Safe concurrent access with proper locking
- **Zero Dependencies**: Minimal external dependencies for core functionality
- **Memory Efficient**: Optimized data structures for large-scale operations

## 📦 Installation

Add MemVDB to your `Cargo.toml`:

```bash
cargo add memvdb
```

## 🎯 Quick Start

```rust
use memvdb::{CacheDB, Distance, Embedding};
use std::collections::HashMap;

// Create a new in-memory vector database
let mut db = CacheDB::new();

// Create a collection with 128-dimensional vectors using cosine similarity
db.create_collection("documents".to_string(), 128, Distance::Cosine).unwrap();

// Create an embedding with metadata
let mut id = HashMap::new();
id.insert("doc_id".to_string(), "doc_001".to_string());

let mut metadata = HashMap::new();
metadata.insert("title".to_string(), "Sample Document".to_string());
metadata.insert("category".to_string(), "AI".to_string());

let vector = vec![0.1; 128]; // 128-dimensional vector
let embedding = Embedding {
    id,
    vector,
    metadata: Some(metadata),
};

// Insert the embedding
db.insert_into_collection("documents", embedding).unwrap();

// Perform similarity search
let query_vector = vec![0.2; 128];
let collection = db.get_collection("documents").unwrap();
let results = collection.get_similarity(&query_vector, 5);

println!("Found {} similar documents", results.len());
```

## 📚 Core Concepts

### Collections

Collections are containers for embeddings with a specific dimensionality and distance metric. All embeddings within a collection must have the same vector dimension.

```rust
// Create collections with different distance metrics
db.create_collection("images".to_string(), 512, Distance::Euclidean).unwrap();
db.create_collection("text".to_string(), 384, Distance::Cosine).unwrap();
db.create_collection("recommendations".to_string(), 256, Distance::DotProduct).unwrap();
```

### Embeddings

Embeddings consist of a unique identifier, vector data, and optional metadata:

```rust
let mut id = HashMap::new();
id.insert("user_id".to_string(), "user_123".to_string());
id.insert("item_id".to_string(), "item_456".to_string());

let mut metadata = HashMap::new();
metadata.insert("category".to_string(), "electronics".to_string());
metadata.insert("price".to_string(), "299.99".to_string());

let embedding = Embedding {
    id,
    vector: vec![0.1, 0.2, 0.3, 0.4],
    metadata: Some(metadata),
};
```

### Distance Metrics

Choose the appropriate distance metric based on your data and use case:

#### Euclidean Distance
- **Best for**: Spatial data, computer vision features
- **Characteristics**: Sensitive to magnitude, measures geometric distance
- **Range**: [0, ∞)

```rust
db.create_collection("spatial_data".to_string(), 3, Distance::Euclidean).unwrap();
```

#### Cosine Similarity
- **Best for**: Text embeddings, high-dimensional sparse data
- **Characteristics**: Ignores magnitude, measures angle between vectors
- **Range**: [0, 2] (converted from [-1, 1])

```rust
db.create_collection("documents".to_string(), 768, Distance::Cosine).unwrap();
```

#### Dot Product
- **Best for**: Pre-normalized vectors, neural network outputs
- **Characteristics**: Considers both angle and magnitude
- **Range**: (-∞, ∞)

```rust
db.create_collection("embeddings".to_string(), 512, Distance::DotProduct).unwrap();
```

## 🔧 API Reference

### Database Operations

```rust
// Create a new database
let mut db = CacheDB::new();

// Create a collection
db.create_collection("my_collection".to_string(), 128, Distance::Cosine)?;

// Get a collection
let collection = db.get_collection("my_collection");

// Delete a collection
db.delete_collection("my_collection")?;

// Get all embeddings from a collection
let embeddings = db.get_embeddings("my_collection");
```

### Embedding Operations

```rust
// Insert a single embedding
db.insert_into_collection("my_collection", embedding)?;

// Batch insert/update embeddings
let embeddings = vec![embedding1, embedding2, embedding3];
db.update_collection("my_collection", embeddings)?;

// Similarity search
let collection = db.get_collection("my_collection").unwrap();
let results = collection.get_similarity(&query_vector, 10);
```

### Similarity Search

```rust
let results = collection.get_similarity(&query_vector, k);

for result in results {
    println!("Score: {}", result.score);
    println!("ID: {:?}", result.embedding.id);
    println!("Metadata: {:?}", result.embedding.metadata);
}
```

## 📖 Examples

### Document Similarity Search

```rust
use memvdb::{CacheDB, Distance, Embedding};
use std::collections::HashMap;

let mut db = CacheDB::new();
db.create_collection("articles".to_string(), 384, Distance::Cosine)?;

// Insert documents
let articles = vec![
    ("AI in Healthcare", vec![0.8, 0.7, 0.1, 0.2]),
    ("Machine Learning Basics", vec![0.7, 0.8, 0.2, 0.1]),
    ("Cooking Recipes", vec![0.1, 0.2, 0.9, 0.8]),
];

for (title, mut vector) in articles {
    vector.resize(384, 0.0); // Pad to collection dimension
    
    let mut id = HashMap::new();
    id.insert("title".to_string(), title.to_string());
    
    let mut metadata = HashMap::new();
    metadata.insert("category".to_string(), 
                   if title.contains("AI") || title.contains("Machine") {
                       "Technology".to_string()
                   } else {
                       "Lifestyle".to_string()
                   });
    
    let embedding = Embedding {
        id,
        vector,
        metadata: Some(metadata),
    };
    
    db.insert_into_collection("articles", embedding)?;
}

// Search for AI-related content
let mut query = vec![0.75, 0.75, 0.1, 0.1];
query.resize(384, 0.0);

let collection = db.get_collection("articles").unwrap();
let results = collection.get_similarity(&query, 3);

for (i, result) in results.iter().enumerate() {
    println!("{}. {} (Score: {:.4})", 
             i + 1, 
             result.embedding.id.get("title").unwrap(),
             result.score);
}
```

### Batch Operations

```rust
// Generate many embeddings
let embeddings: Vec<Embedding> = (0..1000).map(|i| {
    let mut id = HashMap::new();
    id.insert("id".to_string(), i.to_string());
    
    Embedding {
        id,
        vector: vec![i as f32 / 1000.0; 128],
        metadata: None,
    }
}).collect();

// Batch insert for better performance
db.update_collection("large_collection", embeddings)?;
```

### E-commerce Recommendations

```rust
let mut db = CacheDB::new();
db.create_collection("products".to_string(), 256, Distance::DotProduct)?;

// Product embeddings with metadata
let products = vec![
    ("laptop_001", "Gaming Laptop", "Electronics", 1299.99),
    ("book_001", "Rust Programming", "Books", 39.99),
    ("headphones_001", "Wireless Headphones", "Electronics", 199.99),
];

for (product_id, name, category, price) in products {
    let mut id = HashMap::new();
    id.insert("product_id".to_string(), product_id.to_string());
    
    let mut metadata = HashMap::new();
    metadata.insert("name".to_string(), name.to_string());
    metadata.insert("category".to_string(), category.to_string());
    metadata.insert("price".to_string(), price.to_string());
    
    // Simulate product embedding based on features
    let vector = generate_product_embedding(name, category);
    
    let embedding = Embedding {
        id,
        vector,
        metadata: Some(metadata),
    };
    
    db.insert_into_collection("products", embedding)?;
}

// Find similar products
let user_preference_vector = vec![0.5; 256];
let collection = db.get_collection("products").unwrap();
let recommendations = collection.get_similarity(&user_preference_vector, 5);
```

## 🔧 Building and Testing

### Build

```bash
cargo build --release
```

### Run Tests

```bash
# Run all tests
cargo test

# Run specific test categories
cargo test --lib                    # Unit tests
cargo test --test integration_tests # Integration tests
cargo test --test similarity_tests  # Similarity function tests
cargo test --test error_tests       # Error handling tests
cargo test --test performance_tests # Performance tests
```

### Run Examples

```bash
# Basic usage example
cargo run --example basic_usage

# Document similarity search
cargo run --example document_similarity

# Distance metrics comparison
cargo run --example distance_metrics

# Batch operations
cargo run --example batch_operations
```

## 🛣️ Use Cases

### Machine Learning
- **Feature similarity search**: Find similar data points in feature space
- **Nearest neighbor algorithms**: Implement k-NN classifiers
- **Clustering validation**: Evaluate cluster quality with distance metrics

### Natural Language Processing
- **Semantic search**: Find semantically similar documents or sentences
- **Document retrieval**: Information retrieval systems
- **Question answering**: Find relevant context for questions

### Computer Vision
- **Image similarity**: Find visually similar images
- **Face recognition**: Match face embeddings
- **Object detection**: Compare object feature vectors

### Recommendation Systems
- **Content-based filtering**: Recommend similar items
- **User similarity**: Find users with similar preferences
- **Hybrid recommendations**: Combine multiple embedding types

### Information Retrieval
- **Search engines**: Semantic document search
- **Knowledge bases**: Find related concepts
- **Data deduplication**: Identify similar records

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

1. Clone the repository
2. Install Rust (1.70+ recommended)
3. Run tests: `cargo test`
4. Run examples: `cargo run --example basic_usage`

### Code Style

We use standard Rust formatting:

```bash
cargo fmt
cargo clippy
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by [memvectordb](https://github.com/KevKibe/memvectordb)
- Built with Rust's performance and safety in mind
- Optimized for machine learning and AI applications

## 📞 Support

- **Documentation**: [docs.rs/memvdb](https://docs.rs/memvdb)
- **Issues**: [GitHub Issues](https://github.com/AspadaX/memvdb/issues)
- **Discussions**: [GitHub Discussions](https://github.com/AspadaX/memvdb/discussions)

---

**MemVDB** - Fast, lightweight, and efficient in-memory vector search for Rust applications.
