//! # Database Module
//!
//! This module contains the core database functionality for MemVDB, including
//! collection management, embedding storage, and similarity search operations.
//!
//! ## Key Components
//!
//! - [`CacheDB`]: The main database structure that manages collections
//! - [`Collection`]: A container for embeddings with a specific dimensionality and distance metric
//! - [`Embedding`]: Individual vector entries with optional metadata
//! - [`Distance`]: Supported distance metrics for similarity calculations
//! - [`Error`]: Error types for database operations

use crate::similarity::{ScoreIndex, get_cache_attr, get_distance_fn, normalize};
use anyhow::Result;
use log::{debug, error, info};
use rayon::prelude::*;
use std::collections::HashSet;
use std::collections::hash_map::DefaultHasher;
use std::collections::{BinaryHeap, HashMap};
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::{BufReader, Write};

use serde::{Deserialize, Serialize};

/// The main in-memory vector database structure.
///
/// `CacheDB` manages multiple collections of embeddings, each with their own
/// dimensionality and distance metric. It provides thread-safe operations
/// for creating, updating, and querying vector collections.
///
/// # Examples
///
/// ```rust
/// use memvdb::{CacheDB, Distance};
///
/// let mut db = CacheDB::new();
/// db.create_collection("images".to_string(), 512, Distance::Cosine).unwrap();
/// assert!(db.get_collection("images").is_some());
/// ```
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct CacheDB {
    /// HashMap containing all collections, indexed by collection name
    pub collections: HashMap<String, Collection>,
}

/// Result of a similarity search operation.
///
/// Contains the similarity score and the corresponding embedding.
/// Results are typically returned in order of similarity (best matches first).
///
/// # Fields
///
/// * `score` - The similarity score (interpretation depends on distance metric)
/// * `embedding` - The matching embedding with its metadata
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct SimilarityResult {
    /// Similarity score - lower values indicate closer matches for distance metrics
    pub score: f32,
    /// The embedding that matched the query
    pub embedding: Embedding,
}

/// A collection of embeddings with a specific dimensionality and distance metric.
///
/// Collections ensure that all embeddings have the same vector dimensionality
/// and use a consistent distance metric for similarity calculations.
///
/// # Examples
///
/// ```rust
/// use memvdb::{Collection, Distance};
///
/// let collection = Collection {
///     dimension: 128,
///     distance: Distance::Euclidean,
///     embeddings: Vec::new(),
/// };
/// assert_eq!(collection.dimension, 128);
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct Collection {
    /// The required dimensionality for all vectors in this collection
    pub dimension: usize,
    /// The distance metric used for similarity calculations
    pub distance: Distance,
    /// Vector of all embeddings stored in this collection
    #[serde(default)]
    pub embeddings: Vec<Embedding>,
}

/// An individual embedding (vector) with associated metadata.
///
/// Embeddings consist of a unique identifier, the vector data, and optional metadata.
/// The ID can be a composite key using multiple string fields for flexibility.
///
/// # Examples
///
/// ```rust
/// use memvdb::Embedding;
/// use std::collections::HashMap;
///
/// let mut id = HashMap::new();
/// id.insert("doc_id".to_string(), "document_123".to_string());
///
/// let mut metadata = HashMap::new();
/// metadata.insert("title".to_string(), "Example Document".to_string());
///
/// let embedding = Embedding {
///     id,
///     vector: vec![0.1, 0.2, 0.3],
///     metadata: Some(metadata),
/// };
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct Embedding {
    /// Unique identifier for this embedding (can be composite)
    pub id: HashMap<String, String>,
    /// The vector data as floating-point numbers
    pub vector: Vec<f32>,
    /// Optional metadata associated with this embedding
    pub metadata: Option<HashMap<String, String>>,
}

/// Supported distance metrics for similarity calculations.
///
/// Each metric is optimized for different types of vector data and use cases:
///
/// - **Euclidean**: Traditional geometric distance, good for spatial data
/// - **Cosine**: Measures angle between vectors, ideal for text embeddings
/// - **Dot Product**: Efficient for normalized vectors and neural network outputs
///
/// # Examples
///
/// ```rust
/// use memvdb::Distance;
///
/// let metric = Distance::Cosine;
/// assert_eq!(metric, Distance::Cosine);
/// assert_ne!(metric, Distance::Euclidean);
/// ```
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum Distance {
    /// Euclidean (L2) distance - sqrt(sum((a_i - b_i)^2))
    #[serde(rename = "euclidean")]
    Euclidean,
    /// Cosine similarity - measures angle between vectors
    #[serde(rename = "cosine")]
    Cosine,
    /// Dot product - sum(a_i * b_i)
    #[serde(rename = "dot")]
    DotProduct,
}

/// Error types for database operations.
///
/// These errors cover all possible failure modes in the database operations,
/// from validation errors to resource not found conditions.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum Error {
    #[error("Collection already exists")]
    UniqueViolation,

    #[error("Embedding already exists")]
    EmbeddingUniqueViolation,

    #[error("Collection doesn't exist")]
    NotFound,

    #[error("The dimension of the vector doesn't match the dimension of the collection")]
    DimensionMismatch,

    #[error("Failed to initialize the logger")]
    LoggerInitializationError,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
/// Configuration for creating a new collection.
///
/// This struct encapsulates all the parameters needed to create a new collection
/// in the database.
///
/// # Examples
///
/// ```rust
/// use memvdb::{CreateCollectionStruct, Distance};
///
/// let config = CreateCollectionStruct {
///     collection_name: "documents".to_string(),
///     dimension: 768,
///     distance: Distance::Cosine,
/// };
/// ```
pub struct CreateCollectionStruct {
    /// Name of the collection to create
    pub collection_name: String,
    /// Vector dimensionality for this collection
    pub dimension: usize,
    /// Distance metric to use for similarity calculations
    pub distance: Distance,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]

/// Configuration for inserting a single embedding.
///
/// Contains the collection name and the embedding to insert.
pub struct InsertEmbeddingStruct {
    /// Name of the target collection
    pub collection_name: String,
    /// The embedding to insert
    pub embedding: Embedding,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
/// Configuration for collection operations.
///
/// Simple struct containing just the collection name for operations
/// that only need to identify a collection.
pub struct CollectionHandlerStruct {
    /// Name of the target collection
    pub collection_name: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
/// Configuration for batch embedding operations.
///
/// Used for inserting or updating multiple embeddings at once,
/// which is more efficient than individual operations.
pub struct BatchInsertEmbeddingsStruct {
    /// Name of the target collection
    pub collection_name: String,
    /// Vector of embeddings to insert/update
    pub embeddings: Vec<Embedding>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
/// Configuration for similarity search operations.
///
/// Contains all parameters needed to perform a k-nearest neighbors search
/// within a specific collection.
///
/// # Examples
///
/// ```rust
/// use memvdb::GetSimilarityStruct;
///
/// let query = GetSimilarityStruct {
///     collection_name: "documents".to_string(),
///     query_vector: vec![0.1, 0.2, 0.3],
///     k: 10,
/// };
/// ```
pub struct GetSimilarityStruct {
    /// Name of the collection to search
    pub collection_name: String,
    /// Query vector to find similarities for
    pub query_vector: Vec<f32>,
    /// Number of nearest neighbors to return
    pub k: usize,
}

// Define a function to hash a HashMap<String, String>.
// A custom hash function, you ensure that the hash value is based solely on the content of the HashMap
pub fn hash_map_id(id: &HashMap<String, String>) -> u64 {
    let mut hasher = DefaultHasher::new();
    for (key, value) in id {
        key.hash(&mut hasher);
        value.hash(&mut hasher);
    }
    hasher.finish()
}

/// A collection that stores embeddings and handles similarity calculations.
impl Collection {
    /// Calculate similarity results for a given query and number of results (k).
    ///
    /// # Arguments
    ///
    /// * `query`: The query vector for which to calculate similarity.
    /// * `k`: The number of top similar results to return.
    ///
    /// # Returns
    ///
    /// A vector of similarity results, sorted by their similarity scores.
    pub fn get_similarity(&self, query: &[f32], k: usize) -> Vec<SimilarityResult> {
        debug!(
            "Starting similarity computation with query vector of length {} and top k = {}",
            query.len(),
            k
        );

        // Prepare cache attributes and distance function based on collection's distance metric.
        let memo_attr = get_cache_attr(self.distance, query);
        let distance_fn = get_distance_fn(self.distance);

        debug!("Using distance function: {:?}", self.distance);
        debug!("Memo attributes for distance function: {:?}", memo_attr);

        // Calculate similarity scores for each embedding in parallel.
        let scores = self
            .embeddings
            .par_iter()
            .enumerate()
            .map(|(index, embedding)| {
                let score = distance_fn(&embedding.vector, query, memo_attr);
                ScoreIndex { score, index }
            })
            .collect::<Vec<_>>();
        debug!("Calculated {} similarity scores", scores.len());
        // Use a binary heap to efficiently find the top k similarity results.
        let mut heap = BinaryHeap::new();
        for score_index in scores {
            // Only keep top k results in the heap.
            if heap.len() < k || score_index < *heap.peek().unwrap() {
                heap.push(score_index);
                if heap.len() > k {
                    heap.pop();
                }
            }
        }
        debug!("Top k heap size: {}", heap.len());

        // Convert the heap into a sorted vector and map each score to a SimilarityResult.
        let result: Vec<SimilarityResult> = heap
            .into_sorted_vec()
            .into_iter()
            .map(|ScoreIndex { score, index }| SimilarityResult {
                score,
                embedding: self.embeddings[index].clone(),
            })
            .collect();
        info!(
            "Similarity computed successfully'{}' ",
            format!("{:?}", result)
        );
        result
    }
}

/// Database management functionality for collections of embeddings.
impl CacheDB {
    /// Initialize a new CacheDB instance.
    pub fn new() -> Self {
        Self {
            collections: HashMap::new(),
        }
    }
    /// Create a new collection in the database.
    ///
    /// # Arguments
    ///
    /// * `name`: The name of the collection to create.
    /// * `dimension`: The dimension of the embeddings in the collection.
    /// * `distance`: The distance metric to use for similarity calculations.
    ///
    /// # Returns
    ///
    /// A result containing the new collection or an error if a collection with the same name already exists.
    pub fn create_collection(
        &mut self,
        name: String,
        dimension: usize,
        distance: Distance,
    ) -> Result<Collection, Error> {
        // Check if a collection with the same name already exists.
        if self.collections.contains_key(&name) {
            error!("Collection: '{}', already exists", name);
            return Err(Error::UniqueViolation);
        }

        // Create a new collection and add it to the database.
        let collection = Collection {
            dimension,
            distance,
            embeddings: Vec::new(),
        };
        self.collections.insert(name.clone(), collection.clone());

        info!(
            "Created new collection with name: '{}', dimension: '{}', distance: '{:?}'",
            name, dimension, distance
        );
        Ok(collection)
    }

    /// Delete a collection from the database.
    ///
    /// # Arguments
    ///
    /// * `name`: The name of the collection to delete.
    ///
    /// # Returns
    ///
    /// A result indicating success or an error if the collection was not found.
    pub fn delete_collection(&mut self, name: &str) -> Result<(), Error> {
        // Check if the collection exists before attempting to delete it.
        if !self.collections.contains_key(name) {
            error!("Collection name: '{}', does not exist", name);
            return Err(Error::NotFound);
        }

        // Remove the collection from the database.
        self.collections.remove(name);

        info!("Deleted collection: '{}'", name);
        Ok(())
    }

    /// Insert a new embedding into a specified collection.
    ///
    /// # Arguments
    ///
    /// * `collection_name`: The name of the collection to insert the embedding into.
    /// * `embedding`: The embedding to insert.
    ///
    /// # Returns
    ///
    /// A result indicating success or an error if the collection was not found, the embedding is a duplicate, or the embedding dimension does not match the collection.
    pub fn insert_into_collection(
        &mut self,
        collection_name: &str,
        mut embedding: Embedding,
    ) -> Result<(), Error> {
        // Get the collection to insert the embedding into.
        let collection = self
            .collections
            .get_mut(collection_name)
            .ok_or(Error::NotFound)?;

        // Create a HashSet to track unique hashed IDs.
        let mut unique_ids: HashSet<u64> = collection
            .embeddings
            .iter()
            .map(|e| hash_map_id(&e.id))
            .collect();

        // Check for duplicate embeddings by hashed ID.
        if !unique_ids.insert(hash_map_id(&embedding.id)) {
            error!(
                "Embedding with ID '{}' already exists in collection '{}'",
                format!("{:?}", embedding.id),
                collection_name
            );
            return Err(Error::EmbeddingUniqueViolation);
        }

        // Check if the embedding's dimension matches the collection's dimension.
        if embedding.vector.len() != collection.dimension {
            error!(
                "Dimension mismatch: embedding vector length is '{}' but collection '{}' expects dimension '{}'",
                embedding.vector.len(),
                collection_name,
                collection.dimension
            );
            return Err(Error::DimensionMismatch);
        }

        // Normalize the embedding vector if using cosine distance for more efficient calculations.
        if collection.distance == Distance::Cosine {
            embedding.vector = normalize(&embedding.vector);
        }

        // Add the embedding to the collection.
        collection.embeddings.push(embedding.clone());

        info!(
            "Embedding: '{:?}', successfully inserted into collection '{}'",
            embedding, collection_name
        );
        Ok(())
    }

    /// Update a collection with new embeddings.
    ///
    /// # Arguments
    ///
    /// * `collection_name`: The name of the collection to update.
    /// * `new_embeddings`: A vector of new embeddings to add to the collection.
    ///
    /// # Returns
    ///
    /// A result indicating success or an error if the collection was not found, there are duplicate embeddings, or the embedding dimensions do not match the collection's dimension.
    pub fn update_collection(
        &mut self,
        collection_name: &str,
        mut new_embeddings: Vec<Embedding>,
    ) -> Result<(), Error> {
        // Get the collection to update.
        let collection = self
            .collections
            .get_mut(collection_name)
            .ok_or(Error::NotFound)?;

        // Iterate through each new embedding.
        for embedding in &mut new_embeddings {
            // Create a HashSet to track unique hashed IDs.
            let mut unique_ids: HashSet<u64> = collection
                .embeddings
                .iter()
                .map(|e| hash_map_id(&e.id))
                .collect();

            // Check for duplicate embeddings by hashed ID.
            if !unique_ids.insert(hash_map_id(&embedding.id)) {
                error!(
                    "Embedding with ID '{}' already exists in collection '{}'",
                    format!("{:?}", embedding.id),
                    collection_name
                );
                return Err(Error::UniqueViolation);
            }

            // Check if the embedding's dimension matches the collection's dimension.
            if embedding.vector.len() != collection.dimension {
                error!(
                    "Dimension mismatch: embedding vector length is '{}' but collection '{}' expects dimension '{}'",
                    embedding.vector.len(),
                    collection_name,
                    collection.dimension
                );
                return Err(Error::DimensionMismatch);
            }

            // Normalize the vector if using cosine distance for efficient calculations.
            if collection.distance == Distance::Cosine {
                embedding.vector = normalize(&embedding.vector);
            }

            // Add the embedding to the collection.
            collection.embeddings.push(embedding.clone());
        }

        info!(
            "Embedding: '{:?}' successfully updated to collection '{}'",
            new_embeddings, collection_name
        );
        Ok(())
    }

    /// Retrieve a collection from the database.
    ///
    /// # Arguments
    ///
    /// * `collection_name`: The name of the collection to retrieve.
    ///
    /// # Returns
    ///
    /// An optional reference to the collection if found.
    pub fn get_collection(&self, collection_name: &str) -> Option<&Collection> {
        match self.collections.get(collection_name) {
            Some(collection) => {
                info!("Collection '{}' found", collection_name);
                Some(collection)
            }
            None => {
                error!("Collection '{}' not found", collection_name);
                None
            }
        }
    }

    /// Retrieve embeddings from a collection in the database.
    ///
    /// # Arguments
    ///
    /// * `collection_name`: The name of the collection to retrieve.
    ///
    /// # Returns
    ///
    /// An optional reference to the embeddings if found.
    pub fn get_embeddings(&self, collection_name: &str) -> Option<Vec<Embedding>> {
        match self.collections.get(collection_name) {
            Some(collection) => {
                info!(
                    "Successfully retrieved embeddings for collection '{}'",
                    collection_name
                );
                Some(collection.embeddings.clone())
            }
            None => {
                error!("Collection '{}' not found", collection_name);
                None
            }
        }
    }

    /// Persist the database to disk as a JSON file.
    ///
    /// Serializes the entire CacheDB instance, including all collections and their embeddings,
    /// to a JSON file. This allows for data persistence across application restarts.
    ///
    /// # Arguments
    ///
    /// * `filepath` - Optional path to the output file. If `None`, defaults to "./memvdb.json"
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on successful save, or an error if serialization or file operations fail.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use memvdb::{CacheDB, Distance};
    /// use std::collections::HashMap;
    ///
    /// let mut db = CacheDB::new();
    /// db.create_collection("documents".to_string(), 128, Distance::Cosine).unwrap();
    ///
    /// // Save to a specific file
    /// db.save(Some("my_database.json")).unwrap();
    ///
    /// // Save to default location
    /// db.save(None).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The database cannot be serialized to JSON
    /// - The file cannot be created or written to
    /// - There are insufficient permissions to write to the specified path
    pub fn save(&self, filepath: Option<&str>) -> Result<()> {
        let file_content: String = serde_json::to_string_pretty(&self)?;

        // Use a default path to the file if not supplied
        let filepath: &str = if let Some(filepath) = filepath {
            filepath
        } else {
            "memvdb.json"
        };

        // Write contents to the file
        let mut file: File = File::create(filepath)?;
        file.write_all(file_content.as_bytes())?;

        info!("Database successfully saved to '{}'", filepath);
        Ok(())
    }

    /// Load a database from a JSON file on disk.
    ///
    /// Deserializes a previously saved CacheDB instance from a JSON file,
    /// restoring all collections, embeddings, and their associated metadata.
    ///
    /// # Arguments
    ///
    /// * `filepath` - Path to the JSON file containing the serialized database
    ///
    /// # Returns
    ///
    /// Returns the loaded `CacheDB` instance on success, or an error if the file
    /// cannot be read or deserialized.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use memvdb::CacheDB;
    ///
    /// // Load a previously saved database
    /// let db = CacheDB::load("my_database.json").unwrap();
    ///
    /// // Verify collections were loaded
    /// if let Some(collection) = db.get_collection("documents") {
    ///     println!("Loaded collection with {} embeddings", collection.embeddings.len());
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The specified file does not exist or cannot be opened
    /// - The file content is not valid JSON
    /// - The JSON structure doesn't match the expected CacheDB format
    /// - There are insufficient permissions to read the file
    pub fn load(filepath: &str) -> Result<Self> {
        let file: File = std::fs::OpenOptions::new().open(filepath)?;
        let buffer: BufReader<File> = BufReader::new(file);

        let db: CacheDB = serde_json::from_reader(buffer)?;
        info!("Database successfully loaded from '{}'", filepath);
        Ok(db)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_collection_success_eucledean() {
        let mut db = CacheDB::new();
        let result = db.create_collection("test_collection".to_string(), 100, Distance::Euclidean);

        assert!(result.is_ok());
        let collection = result.unwrap();
        assert_eq!(collection.dimension, 100);
        assert_eq!(collection.distance, Distance::Euclidean);
        assert!(db.collections.contains_key("test_collection"));
    }

    #[test]
    fn test_create_collection_success_cosine() {
        let mut db = CacheDB::new();
        let result = db.create_collection("test_collection".to_string(), 100, Distance::Cosine);

        assert!(result.is_ok());
        let collection = result.unwrap();
        assert_eq!(collection.dimension, 100);
        assert_eq!(collection.distance, Distance::Cosine);
        assert!(db.collections.contains_key("test_collection"));
    }

    #[test]
    fn test_create_collection_success_dot_product() {
        let mut db = CacheDB::new();
        let result = db.create_collection("test_collection".to_string(), 100, Distance::DotProduct);

        assert!(result.is_ok());
        let collection = result.unwrap();
        assert_eq!(collection.dimension, 100);
        assert_eq!(collection.distance, Distance::DotProduct);
        assert!(db.collections.contains_key("test_collection"));
    }

    #[test]
    fn test_create_collection_already_exists() {
        let mut db = CacheDB::new();
        db.create_collection("test_collection".to_string(), 100, Distance::Euclidean)
            .unwrap();

        let result = db.create_collection("test_collection".to_string(), 200, Distance::Cosine);
        assert!(result.is_err());
    }

    #[test]
    fn test_insert_into_collection_success() {
        let mut db = CacheDB::new();
        let collection = Collection {
            dimension: 3,
            distance: Distance::Euclidean,
            embeddings: Vec::new(),
        };
        db.collections
            .insert("test_collection".to_string(), collection);
        let mut metadata = HashMap::new();
        metadata.insert("page".to_string(), "1".to_string());
        metadata.insert(
            "text".to_string(),
            "This is a test metadata text".to_string(),
        );

        let mut id = HashMap::new();
        id.insert("unique_id".to_string(), "1".to_string());

        let embedding = Embedding {
            id: id,
            vector: vec![1.0, 2.0, 3.0],
            metadata: Some(metadata),
        };

        let result = db.insert_into_collection("test_collection", embedding.clone());
        assert!(result.is_ok());

        // Check if the embedding is inserted into the collection
        let collection = db.collections.get("test_collection").unwrap();
        assert_eq!(collection.embeddings.len(), 1);
        assert_eq!(collection.embeddings[0], embedding);
    }

    #[test]
    fn test_update_collection_success() {
        let mut db = CacheDB::new();

        let mut metadata = HashMap::new();
        metadata.insert("page".to_string(), "1".to_string());
        metadata.insert(
            "text".to_string(),
            "This is a test metadata text".to_string(),
        );

        let mut id = HashMap::new();
        id.insert("unique_id".to_string(), "0".to_string());

        let collection = Collection {
            dimension: 3,
            distance: Distance::Euclidean,
            embeddings: vec![Embedding {
                id: id,
                vector: vec![1.0, 2.0, 3.0],
                metadata: Some(metadata.clone()),
            }],
        };

        db.collections
            .insert("test_collection".to_string(), collection);

        let mut id_1 = HashMap::new();
        id_1.insert("unique_id".to_string(), "1".to_string());
        let mut id_2 = HashMap::new();
        id_2.insert("unique_id".to_string(), "2".to_string());

        let new_embeddings = vec![
            Embedding {
                id: id_1, // Duplicate ID
                vector: vec![4.0, 5.0, 6.0],
                metadata: Some(metadata.clone()),
            },
            Embedding {
                id: id_2,
                vector: vec![7.0, 8.0, 9.0],
                metadata: Some(metadata.clone()),
            },
        ];

        let result = db.update_collection("test_collection", new_embeddings.clone());
        assert!(result.is_ok());

        // Check if the new embeddings are added to the collection
        let collection = db.collections.get("test_collection").unwrap();
        assert_eq!(collection.embeddings.len(), 3);
        assert_eq!(collection.embeddings[1..], new_embeddings[..]);
    }

    #[test]
    fn test_update_collection_duplicate_embedding() {
        let mut db = CacheDB::new();
        let mut metadata = HashMap::new();
        metadata.insert("page".to_string(), "1".to_string());
        metadata.insert(
            "text".to_string(),
            "This is a test metadata text".to_string(),
        );

        let mut id = HashMap::new();
        id.insert("unique_id".to_string(), "0".to_string());

        let collection = Collection {
            dimension: 3,
            distance: Distance::Euclidean,
            embeddings: vec![Embedding {
                id: id.clone(),
                vector: vec![1.0, 2.0, 3.0],
                metadata: Some(metadata.clone()),
            }],
        };
        db.collections
            .insert("test_collection".to_string(), collection);

        let mut id_1 = HashMap::new();
        id_1.insert("unique_id".to_string(), "1".to_string());
        let mut id_2 = HashMap::new();
        id_2.insert("unique_id".to_string(), "2".to_string());

        let new_embeddings = vec![
            Embedding {
                id: id, // Duplicate ID
                vector: vec![4.0, 5.0, 6.0],
                metadata: Some(metadata.clone()),
            },
            Embedding {
                id: id_2,
                vector: vec![7.0, 8.0, 9.0],
                metadata: Some(metadata.clone()),
            },
        ];

        let result = db.update_collection("test_collection", new_embeddings);
        assert!(result.is_err());
        assert_eq!(result.err(), Some(Error::UniqueViolation));
    }

    #[test]
    fn test_update_collection_dimension_mismatch() {
        let mut db = CacheDB::new();
        let collection = Collection {
            dimension: 3,
            distance: Distance::Euclidean,
            embeddings: Vec::new(),
        };
        db.collections
            .insert("test_collection".to_string(), collection);

        let mut metadata = HashMap::new();
        metadata.insert("page".to_string(), "1".to_string());
        metadata.insert(
            "text".to_string(),
            "This is a test metadata text".to_string(),
        );

        let mut id = HashMap::new();
        id.insert("unique_id".to_string(), "0".to_string());

        let new_embeddings = vec![Embedding {
            id: id,
            vector: vec![1.0, 2.0],
            metadata: Some(metadata), // Dimension mismatch
        }];

        let result = db.update_collection("test_collection", new_embeddings);
        assert!(result.is_err());
        assert_eq!(result.err(), Some(Error::DimensionMismatch));
    }

    #[test]
    fn test_delete_collection_success() {
        let mut db = CacheDB::new();
        db.collections.insert(
            "test_collection".to_string(),
            Collection {
                dimension: 3,
                distance: Distance::Euclidean,
                embeddings: Vec::new(),
            },
        );

        let result = db.delete_collection("test_collection");
        assert!(result.is_ok());

        // Check if the collection is removed from the database
        assert!(!db.collections.contains_key("test_collection"));
    }

    #[test]
    fn test_delete_collection_not_found() {
        let mut db = CacheDB::new();

        let result = db.delete_collection("non_existent_collection");
        assert!(result.is_err());
        assert_eq!(result.err(), Some(Error::NotFound));
    }

    #[test]
    fn test_get_collection_success() {
        let mut db = CacheDB::new();
        let collection = Collection {
            dimension: 3,
            distance: Distance::Euclidean,
            embeddings: Vec::new(),
        };
        db.collections
            .insert("test_collection".to_string(), collection.clone());

        let result = db.get_collection("test_collection");
        assert!(result.is_some());

        // Check if the retrieved collection is the same as the original one
        assert_eq!(result.unwrap(), &collection);
    }

    #[test]
    fn test_get_collection_not_found() {
        let db = CacheDB::new();

        let result = db.get_collection("non_existent_collection");
        assert!(result.is_none());
    }

    #[test]
    fn test_get_embedding_success() {
        let mut db = CacheDB::new();

        let mut id = HashMap::new();
        id.insert("unique_id".to_string(), "0".to_string());

        let mut id_1 = HashMap::new();
        id_1.insert("unique_id".to_string(), "1".to_string());

        let mut id_2 = HashMap::new();
        id_2.insert("unique_id".to_string(), "2".to_string());

        let collection = Collection {
            dimension: 3,
            distance: Distance::Euclidean,
            embeddings: vec![
                Embedding {
                    id: id,
                    vector: vec![1.0, 1.0, 1.0],
                    metadata: None,
                },
                Embedding {
                    id: id_1,
                    vector: vec![2.0, 2.0, 2.0],
                    metadata: None,
                },
                Embedding {
                    id: id_2,
                    vector: vec![3.0, 3.0, 3.0],
                    metadata: None,
                },
            ],
        };
        db.collections
            .insert("test_collection".to_string(), collection.clone());
        let result = db.get_embeddings("test_collection");
        assert!(result.is_some());
        assert_eq!(result, Some(collection.embeddings));
    }

    #[test]
    fn test_get_embeddings_not_found() {
        let db = CacheDB::new();

        let result = db.get_embeddings("non_existent_collection");
        assert!(result.is_none());
    }

    #[test]
    fn test_get_similarity() {
        let mut id = HashMap::new();
        id.insert("unique_id".to_string(), "0".to_string());

        let mut id_1 = HashMap::new();
        id_1.insert("unique_id".to_string(), "1".to_string());

        let mut id_2 = HashMap::new();
        id_2.insert("unique_id".to_string(), "2".to_string());

        let collection = Collection {
            dimension: 3,
            distance: Distance::Euclidean,
            embeddings: vec![
                Embedding {
                    id: id.clone(),
                    vector: vec![1.0, 1.0, 1.0],
                    metadata: None,
                },
                Embedding {
                    id: id_1.clone(),
                    vector: vec![2.0, 2.0, 2.0],
                    metadata: None,
                },
                Embedding {
                    id: id_2.clone(),
                    vector: vec![3.0, 3.0, 3.0],
                    metadata: None,
                },
            ],
        };

        // Define a query vector
        let query = vec![0.0, 0.0, 0.0];

        // Define the expected similarity results
        let expected_results = vec![
            SimilarityResult {
                score: 0.0,
                embedding: Embedding {
                    id: id_1,
                    vector: vec![2.0, 2.0, 2.0],
                    metadata: None,
                },
            },
            SimilarityResult {
                score: 0.0,
                embedding: Embedding {
                    id: id_2,
                    vector: vec![3.0, 3.0, 3.0],
                    metadata: None,
                },
            },
            SimilarityResult {
                score: 0.0,
                embedding: Embedding {
                    id: id,
                    vector: vec![1.0, 1.0, 1.0],
                    metadata: None,
                },
            },
        ];

        // Call the get_similarity method
        let results = collection.get_similarity(&query, 3);

        // Assert that the results are as expected
        assert_eq!(results, expected_results);
    }
}
