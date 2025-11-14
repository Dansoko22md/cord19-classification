# 🦠 COVID-19 Scientific Articles Classification & Analysis

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CORD-19](https://img.shields.io/badge/Dataset-CORD--19-orange.svg)](https://www.kaggle.com/datasets/allen-institute-for-ai/CORD-19-research-challenge)

> **An end-to-end system for classifying, analyzing, and retrieving COVID-19 scientific articles using NLP, Graph Neural Networks, and Graph-RAG.**
<div align="center">
<img src="output.png" alt="output" align="right" />
</div>
---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Documentation](#-documentation)
- [Project Structure](#-project-structure)
- [Technologies](#-technologies)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 🎯 Overview

This project implements a comprehensive system for analyzing scientific literature on COVID-19 using state-of-the-art Natural Language Processing and Graph Neural Networks. Built on the **CORD-19 dataset**, it enables:

-  **Automatic classification** of scientific articles by topic
-  **Semantic search** across thousands of papers
-  **Graph-based exploration** of article relationships
-  **AI-powered recommendations** using Graph-RAG
-  **Community detection** to identify research themes

### 🎓 Academic Context

This project was developed as part of a data science/AI academic project focusing on:
- Scientific text analysis with transformer models
- Graph-based machine learning
- Information retrieval systems
- COVID-19 research navigation and exploration

---

## ✨ Key Features

### 🧠 Advanced NLP Pipeline
- **SciBERT embeddings** for scientific text understanding
- Support for multiple embedding models (BioBERT, PubMedBERT, SPECTER)
- 768-dimensional semantic representations
- Batch processing for efficient computation

### 🕸️ Graph Analysis
- **Automatic graph construction** based on semantic similarity
- **Community detection** using Louvain algorithm
- **Network analysis** to identify influential papers
- Graph visualization and exploration tools

### 📊 Machine Learning
- **Graph Attention Network (GAT)** for article classification
- Multi-class categorization of research topics
- Transfer learning with pre-trained models
- Evaluation metrics and model comparison

### 🔍 Graph-RAG System
- **Semantic search** with context expansion through graph
- **Article recommendations** based on content and structure
- **Query-based retrieval** augmented by graph relationships
- Community-based thematic exploration

---

##  Architecture


<div align="center">
<img src="architecture.png" alt="conding" align="center" />
</div>

---

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU
- 16GB+ RAM recommended

### Clone the Repository

```bash
git clone https://github.com/Dansoko22md/cord19-classification.git
cd cord19-classification
```

### Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

**Or install individually:**

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy torch transformers torch-geometric networkx python-louvain umap-learn tqdm
```

### Download Dataset

Download the CORD-19 dataset from [Kaggle](https://www.kaggle.com/datasets/allen-institute-for-ai/CORD-19-research-challenge) and place `metadata.csv` in:

```
S1_CORD19_Classification/data/raw/metadata.csv
```

---

## ⚡ Quick Start

### Option 1: Run Complete Pipeline

```bash
python main.py
```

This executes all steps automatically:
1.  Data exploration and cleaning
2.  Embedding creation with SciBERT
3.  Graph construction and community detection
4.  GNN model training
5.  Graph-RAG system setup

**Estimated time:** 30 minutes - 2 hours depending on dataset size and hardware.

### Option 2: Run Step-by-Step

Execute each phase independently:

```bash
python step1_exploration.py    # Data exploration
python step2_embeddings.py     # Create embeddings
python step3_graph.py          # Build graph
python step4_gnn.py            # Train GNN model
python step5_rag.py            # Setup Graph-RAG
```

### Option 3: Interactive Notebook

For exploration and analysis:

```bash
jupyter notebook exploration_interactive.ipynb
```

---

## 📖 Documentation

### 1. Data Exploration (`step1_exploration.py`)

Explores and prepares the CORD-19 dataset:

```python
from step1_exploration import CORD19Explorer

# Initialize explorer
explorer = CORD19Explorer("data/raw")

# Load and explore
df = explorer.load_metadata()
explorer.explore_structure()

# Clean and filter
clean_df = explorer.filter_quality_articles(min_abstract_length=100)
```

**Key Functions:**
- `load_metadata()` - Load CORD-19 dataset
- `explore_structure()` - Display statistics
- `filter_quality_articles()` - Clean and filter data
- `visualize_statistics()` - Create visualizations

---

### 2. Embeddings Creation (`step2_embeddings.py`)

Generate semantic embeddings using SciBERT:

```python
from step2_embeddings import ScientificEmbedder

# Initialize model
embedder = ScientificEmbedder(model_name='allenai/scibert_scivocab_uncased')

# Create embeddings
embeddings = embedder.embed_dataframe(
    df, 
    text_column='full_text',
    save_path='data/processed/embeddings.npy'
)

# Visualize
embedder.visualize_embeddings(embeddings, method='umap')
```

**Key Functions:**
- `embed_dataframe()` - Batch embedding creation
- `reduce_dimensions()` - Dimensionality reduction (UMAP/t-SNE/PCA)
- `visualize_embeddings()` - 2D/3D visualization

**Supported Models:**
- SciBERT (recommended)
- PubMedBERT
- BioBERT
- SPECTER
- See model comparison for details

---

### 3. Graph Construction (`step3_graph.py`)

Build and analyze the article graph:

```python
from step3_graph import ArticleGraphBuilder

# Initialize builder
builder = ArticleGraphBuilder(df, embeddings)

# Build graph
graph = builder.build_similarity_graph(
    threshold=0.75,
    max_edges_per_node=15
)

# Detect communities
communities = builder.detect_communities(algorithm='louvain')

# Analyze
stats = builder.analyze_communities(top_n=10)

# Visualize
builder.visualize_graph(max_nodes=500)
```

**Key Functions:**
- `build_similarity_graph()` - Construct graph from embeddings
- `detect_communities()` - Find thematic clusters
- `analyze_communities()` - Analyze cluster statistics
- `get_graph_statistics()` - Network metrics
- `export_for_gephi()` - Export for visualization

**Parameters:**
- `threshold` - Minimum similarity for edges (0.7-0.8 recommended)
- `max_edges_per_node` - Maximum connections per article (10-20 recommended)

---

### 4. GNN Classification (`step4_gnn.py`)

Train Graph Neural Network for classification:

```python
from step4_gnn import GNNClassifier, ArticleClassifierTrainer

# Create model
model = GNNClassifier(
    input_dim=768,
    hidden_dim=256,
    output_dim=num_classes,
    model_type='GAT',
    num_layers=3
)

# Train
trainer = ArticleClassifierTrainer(model)
data = trainer.prepare_data(graph, embeddings, labels)
trainer.train(data, epochs=100, lr=0.001)

# Evaluate
test_acc, predictions, true_labels = trainer.test(data)
```

**Key Functions:**
- `GNNClassifier` - GAT/GCN/GraphSAGE models
- `prepare_data()` - Convert to PyTorch Geometric format
- `train()` - Train with early stopping
- `test()` - Evaluate performance
- `plot_training_history()` - Visualize training

**Model Options:**
- `GCN` - Graph Convolutional Network
- `GAT` - Graph Attention Network (recommended)
- `GraphSAGE` - Inductive learning

---

### 5. Graph-RAG System (`step5_rag.py`)

Semantic search and recommendations:

```python
from step5_rag import GraphRAG

# Initialize system
rag = GraphRAG(df, graph, embeddings, communities)

# Semantic search
results = rag.search("COVID-19 vaccine effectiveness", top_k=10)

# Graph-enhanced search
results = rag.graph_enhanced_search(
    query="vaccine effectiveness",
    top_k=10,
    expansion_hops=1,
    alpha=0.7
)

# Find similar articles
similar = rag.find_similar_articles(article_idx=42, top_k=10)

# Explore community
community_info = rag.explore_community(community_id=0, top_k=20)

# Question answering
answer = rag.answer_question("What are long-term COVID effects?")
```

**Key Functions:**
- `search()` - Semantic search
- `graph_enhanced_search()` - Search with graph context
- `find_similar_articles()` - Get recommendations
- `explore_community()` - Analyze thematic clusters
- `answer_question()` - Q&A with sources

---

### Utility Functions (`utils.py`)

Helper functions for analysis:

```python
from utils import CORD19Utils, quick_search, get_stats_summary

# Load all results
utils = CORD19Utils()
results = utils.load_results()

# Community analysis
info = utils.get_community_info(community_id=0)
utils.visualize_community(community_id=0)

# Compare communities
comparison = utils.compare_communities(comm_id1=0, comm_id2=1)

# Export
utils.export_community_to_csv(community_id=0)
utils.create_community_summary_report()

# Quick search (standalone)
results = quick_search("vaccine effectiveness", top_k=10)

# Statistics
get_stats_summary()
```

---

### Configuration

Adjust parameters in `main.py`:

```python
config = {
    # Data
    'sample_size': None,  # None = all data, or number for sample
    'min_abstract_length': 100,
    
    # Embeddings
    'model_name': 'allenai/scibert_scivocab_uncased',
    'batch_size': 16,
    'max_length': 512,
    
    # Graph
    'similarity_threshold': 0.75,
    'max_edges_per_node': 15,
    'community_algorithm': 'louvain',
    
    # GNN
    'gnn_type': 'GAT',
    'hidden_dim': 256,
    'num_layers': 3,
    'dropout': 0.3,
    'learning_rate': 0.001,
    'epochs': 100,
    'patience': 15,
}
```

---

## 📁 Project Structure

```
cord19-classification/
├── data/
│   ├── raw/                          # Original CORD-19 dataset
│   │   └── metadata.csv
│   └── processed/                    # Processed data
│       ├── cleaned_articles.csv
│       ├── embeddings.npy
│       ├── article_graph.gpickle
│       └── articles_with_communities.csv
├── models/
│   └── gnn_model.pth                 # Trained GNN model
├── outputs/
│   ├── graph_visualization.png       # Visualizations
│   └── training_history.png
├── notebooks/
│   ├── 00_Installation_Setup.ipynb   # Installation guide
│   └── exploration_interactive.ipynb # Interactive exploration
├── step1_exploration.py              # Data exploration
├── step2_embeddings.py               # Embedding creation
├── step3_graph.py                    # Graph construction
├── step4_gnn.py                      # GNN training
├── step5_rag.py                      # Graph-RAG system
├── main.py                           # Complete pipeline
├── utils.py                          # Utility functions
├── requirements.txt                  # Dependencies
├── README.md                         # This file
└── LICENSE                           # MIT License
```

---

## 🛠️ Technologies

### Core Libraries

- **[PyTorch](https://pytorch.org/)** - Deep learning framework
- **[Transformers](https://huggingface.co/transformers/)** - Pre-trained language models
- **[PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)** - Graph neural networks
- **[NetworkX](https://networkx.org/)** - Graph analysis and algorithms
- **[scikit-learn](https://scikit-learn.org/)** - Machine learning utilities

### Key Models

- **SciBERT** - Scientific text embeddings (default)
- **Graph Attention Network (GAT)** - Article classification
- **Louvain Algorithm** - Community detection

### Data Processing

- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical operations
- **UMAP** - Dimensionality reduction and visualization

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Add docstrings to all functions
- Include tests for new features
- Update documentation accordingly

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---


**Project Link:** [https://github.com/Dansoko22md/cord19-classification](https://github.com/Dansoko22md/cord19-classification)


---

<div align="center">

**⭐ If you find this project useful, please star it!**

Made with ❤️ for the research community

</div>