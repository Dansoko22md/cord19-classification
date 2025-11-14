import pandas as pd
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from pathlib import Path
from tqdm import tqdm
import json
import time
from datetime import datetime
import logging
from scipy.sparse import lil_matrix, csr_matrix
import sys

# Configure logging to handle Unicode on Windows
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('graph_construction.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

try:
    import community.community_louvain as community_louvain
    HAS_LOUVAIN = True
except ImportError:
    HAS_LOUVAIN = False
    logger.warning("python-louvain not installed, using greedy")


class FastGraphBuilder:
    """Ultra-fast graph builder using approximate nearest neighbors"""
    
    def __init__(self, df, embeddings):
        self.df = df.reset_index(drop=True)
        self.embeddings = embeddings
        self.graph = None
        self.communities = None
        self.stats = {}
        
        logger.info("="*70)
        logger.info("FAST GRAPH BUILDER")
        logger.info("="*70)
        logger.info(f"Articles: {len(df):,}")
        logger.info(f"Embeddings: {embeddings.shape}")
        logger.info(f"Memory: {embeddings.nbytes / 1024**3:.2f} GB")
        
        if len(df) != len(embeddings):
            raise ValueError(f"Mismatch: {len(df)} vs {len(embeddings)}")
    
    def build_graph_fast(self, threshold=0.75, max_edges_per_node=15):
        """
        Build graph using FAISS for fast nearest neighbor search
        OPTIMIZED for large datasets (800K+ nodes)
        """
        logger.info("="*70)
        logger.info("FAST GRAPH CONSTRUCTION WITH FAISS")
        logger.info("="*70)
        logger.info(f"Threshold: {threshold}")
        logger.info(f"Max edges per node: {max_edges_per_node}")
        
        try:
            import faiss
            HAS_FAISS = True
            logger.info("[OK] FAISS available - using optimized CPU index")
        except ImportError:
            HAS_FAISS = False
            logger.warning("[WARNING] FAISS not installed, falling back to sklearn")
        
        start_time = time.time()
        n_samples = len(self.embeddings)
        
        G = nx.Graph()
        
        # Add nodes (fast)
        logger.info("Adding nodes...")
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Nodes"):
            G.add_node(idx,
                      title=str(row['title'])[:100] if pd.notna(row['title']) else 'No title',
                      abstract=str(row['abstract'])[:200] if pd.notna(row['abstract']) else '',
                      source=str(row['source_x']) if 'source_x' in row else 'unknown')
        
        if HAS_FAISS:
            # CRITICAL FIX: Use batched search instead of single big search
            logger.info("Building FAISS index with optimized settings...")
            
            # Normalize embeddings
            embeddings_norm = self.embeddings.astype('float32')
            faiss.normalize_L2(embeddings_norm)
            
            dimension = embeddings_norm.shape[1]
            
            # Use flat index (fastest for exact search)
            index = faiss.IndexFlatIP(dimension)
            logger.info("Using CPU IndexFlatIP (optimized for cosine similarity)")
            
            # Add to index
            logger.info("Adding vectors to index...")
            index.add(embeddings_norm)
            logger.info(f"[OK] Index built with {index.ntotal:,} vectors")
            
            # CRITICAL: Search in batches to avoid memory issues
            k = max_edges_per_node + 1
            batch_size = 10000  # Process 10K queries at a time
            n_batches = (n_samples + batch_size - 1) // batch_size
            
            logger.info(f"Searching {k} nearest neighbors in {n_batches} batches...")
            logger.info(f"Estimated time: ~{n_batches * 0.5:.1f} minutes")
            
            weights = []
            edge_count = 0
            
            for batch_idx in tqdm(range(n_batches), desc="Searching batches"):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, n_samples)
                
                # Search this batch
                batch_embeddings = embeddings_norm[start_idx:end_idx]
                similarities, indices = index.search(batch_embeddings, k)
                
                # Add edges from results
                for i in range(len(batch_embeddings)):
                    global_i = start_idx + i
                    
                    for j_idx in range(1, k):  # Skip first (self)
                        j = indices[i, j_idx]
                        sim = similarities[i, j_idx]
                        
                        if sim >= threshold:
                            # Only add if not already exists (avoid duplicates)
                            if not G.has_edge(global_i, j):
                                G.add_edge(global_i, int(j), weight=float(sim))
                                weights.append(sim)
                                edge_count += 1
                
                # Progress update every 10 batches
                if (batch_idx + 1) % 10 == 0:
                    logger.info(f"Batch {batch_idx+1}/{n_batches} | "
                              f"Edges so far: {G.number_of_edges():,}")
        
        else:
            # Sklearn fallback with smart batching
            logger.info("Using optimized sklearn approach with batching...")
            
            batch_size = 2000
            n_batches = (n_samples + batch_size - 1) // batch_size
            
            logger.info(f"Processing {n_batches} batches")
            logger.info(f"Estimated time: ~{n_batches * 5 / 60:.1f} minutes")
            
            weights = []
            
            for i in tqdm(range(n_batches), desc="Batches", unit="batch"):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_samples)
                
                batch_embeddings = self.embeddings[start_idx:end_idx]
                remaining_embeddings = self.embeddings[end_idx:]
                
                if len(remaining_embeddings) > 0:
                    similarities = cosine_similarity(batch_embeddings, remaining_embeddings)
                    
                    for local_idx in range(len(batch_embeddings)):
                        global_idx = start_idx + local_idx
                        sims = similarities[local_idx]
                        
                        if len(sims) > 0:
                            k_to_get = min(max_edges_per_node, len(sims))
                            if k_to_get > 0:
                                top_k_indices = np.argpartition(sims, -k_to_get)[-k_to_get:]
                                top_k_indices = top_k_indices[np.argsort(-sims[top_k_indices])]
                                
                                added = 0
                                for local_j in top_k_indices:
                                    j = end_idx + local_j
                                    sim_score = sims[local_j]
                                    
                                    if sim_score >= threshold and added < max_edges_per_node:
                                        G.add_edge(global_idx, j, weight=float(sim_score))
                                        weights.append(sim_score)
                                        added += 1
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Batch {i+1}/{n_batches} | Edges: {G.number_of_edges():,}")
        
        elapsed = time.time() - start_time
        self.graph = G
        
        logger.info("="*70)
        logger.info("GRAPH STATISTICS")
        logger.info("="*70)
        logger.info(f"[TIME] Construction time: {elapsed/60:.1f} minutes")
        logger.info(f"[NODES] Nodes: {G.number_of_nodes():,}")
        logger.info(f"[EDGES] Edges: {G.number_of_edges():,}")
        logger.info(f"[DENSITY] Density: {nx.density(G):.6f}")
        logger.info(f"[DEGREE] Avg degree: {2*G.number_of_edges()/G.number_of_nodes():.2f}")
        
        if weights:
            logger.info(f"[WEIGHTS] Edge weights:")
            logger.info(f"  Mean: {np.mean(weights):.4f}")
            logger.info(f"  Median: {np.median(weights):.4f}")
            logger.info(f"  Min: {np.min(weights):.4f}")
            logger.info(f"  Max: {np.max(weights):.4f}")
        
        components = list(nx.connected_components(G))
        logger.info(f"[COMPONENTS] Connected components: {len(components)}")
        if len(components) > 1:
            largest = len(max(components, key=len))
            logger.info(f"  Largest: {largest:,} ({largest/G.number_of_nodes()*100:.1f}%)")
        
        return G
    
    def detect_communities_scalable(self, algorithm='louvain', resolution=1.0):
        """Detect communities on large graph"""
        logger.info("="*70)
        logger.info("COMMUNITY DETECTION")
        logger.info("="*70)
        
        if self.graph is None:
            raise ValueError("Build graph first")
        
        start_time = time.time()
        
        if nx.number_connected_components(self.graph) > 1:
            largest_cc = max(nx.connected_components(self.graph), key=len)
            G_connected = self.graph.subgraph(largest_cc).copy()
            logger.info(f"Using largest component: {len(G_connected):,} nodes")
        else:
            G_connected = self.graph
        
        if algorithm == 'louvain' and HAS_LOUVAIN:
            logger.info("Running Louvain community detection...")
            self.communities = community_louvain.best_partition(
                G_connected,
                resolution=resolution,
                random_state=42
            )
            modularity = community_louvain.modularity(self.communities, G_connected)
        else:
            logger.info("Running greedy modularity community detection...")
            communities_gen = nx.community.greedy_modularity_communities(G_connected)
            self.communities = {}
            for i, comm in enumerate(communities_gen):
                for node in comm:
                    self.communities[node] = i
            modularity = nx.community.modularity(G_connected, communities_gen)
        
        for node in self.graph.nodes():
            if node not in self.communities:
                self.communities[node] = -1
        
        elapsed = time.time() - start_time
        num_communities = len(set(self.communities.values()))
        
        comm_sizes = pd.Series(self.communities.values()).value_counts()
        
        logger.info(f"[TIME] Detection time: {elapsed/60:.1f} minutes")
        logger.info(f"[COMMUNITIES] Communities: {num_communities}")
        logger.info(f"[MODULARITY] Modularity: {modularity:.4f}")
        logger.info(f"[SIZES] Community sizes:")
        logger.info(f"  Mean: {comm_sizes.mean():.1f}")
        logger.info(f"  Median: {comm_sizes.median():.1f}")
        logger.info(f"  Min: {comm_sizes.min()}")
        logger.info(f"  Max: {comm_sizes.max()}")
        
        self.stats['modularity'] = modularity
        self.stats['num_communities'] = num_communities
        
        return self.communities
    
    def analyze_communities(self, top_n=10):
        """Analyze detected communities"""
        logger.info("="*70)
        logger.info("COMMUNITY ANALYSIS")
        logger.info("="*70)
        
        if self.communities is None:
            raise ValueError("Detect communities first")
        
        self.df['community'] = self.df.index.map(self.communities)
        
        comm_stats = []
        
        for comm_id in sorted(set(self.communities.values())):
            if comm_id == -1:
                continue
            
            articles = self.df[self.df['community'] == comm_id]
            node_indices = articles.index.tolist()
            
            degrees = [self.graph.degree(node) for node in node_indices 
                      if node in self.graph]
            avg_degree = np.mean(degrees) if degrees else 0
            
            subgraph = self.graph.subgraph(node_indices)
            internal_edges = subgraph.number_of_edges()
            
            comm_stats.append({
                'community_id': comm_id,
                'size': len(articles),
                'avg_degree': avg_degree,
                'internal_edges': internal_edges,
                'density': nx.density(subgraph) if len(node_indices) > 1 else 0,
                'sample_titles': articles['title'].head(3).tolist()
            })
        
        stats_df = pd.DataFrame(comm_stats).sort_values('size', ascending=False)
        
        logger.info(f"Top {top_n} communities:")
        for idx, row in stats_df.head(top_n).iterrows():
            logger.info(f"\n[COMMUNITY] Community {row['community_id']}:")
            logger.info(f"  Size: {row['size']}")
            logger.info(f"  Avg degree: {row['avg_degree']:.1f}")
            logger.info(f"  Internal edges: {row['internal_edges']}")
            logger.info(f"  Density: {row['density']:.4f}")
            logger.info(f"  Sample titles:")
            for i, title in enumerate(row['sample_titles'], 1):
                title_str = str(title)[:70] if pd.notna(title) else 'No title'
                logger.info(f"    {i}. {title_str}...")
        
        return stats_df
    
    def export_graph(self, output_dir):
        """Export graph in multiple formats"""
        logger.info("="*70)
        logger.info("EXPORTING GRAPH")
        logger.info("="*70)
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # NetworkX pickle
        pickle_path = output_dir / "article_graph.gpickle"
        logger.info(f"Saving NetworkX graph...")
        with open(pickle_path, 'wb') as f:
            pickle.dump(self.graph, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"[OK] NetworkX: {pickle_path} ({pickle_path.stat().st_size / 1024**2:.1f} MB)")
        
        # GEXF for Gephi
        gexf_path = output_dir / "article_graph.gexf"
        logger.info(f"Saving GEXF format...")
        if self.communities is not None:
            for node in self.graph.nodes():
                self.graph.nodes[node]['community'] = self.communities.get(node, -1)
        nx.write_gexf(self.graph, gexf_path)
        logger.info(f"[OK] GEXF: {gexf_path} ({gexf_path.stat().st_size / 1024**2:.1f} MB)")
        
        # GraphML format
        graphml_path = output_dir / "article_graph.graphml"
        logger.info(f"Saving GraphML format...")
        nx.write_graphml(self.graph, graphml_path)
        logger.info(f"[OK] GraphML: {graphml_path} ({graphml_path.stat().st_size / 1024**2:.1f} MB)")
        
        # Edge list (lightweight)
        edgelist_path = output_dir / "article_graph_edges.csv"
        logger.info(f"Saving edge list...")
        with open(edgelist_path, 'w') as f:
            f.write("source,target,weight\n")
            for u, v, data in self.graph.edges(data=True):
                f.write(f"{u},{v},{data.get('weight', 1.0)}\n")
        logger.info(f"[OK] Edge list: {edgelist_path} ({edgelist_path.stat().st_size / 1024**2:.1f} MB)")
        
        # Communities
        if self.communities is not None:
            comm_path = output_dir / "communities.pkl"
            with open(comm_path, 'wb') as f:
                pickle.dump(self.communities, f)
            logger.info(f"[OK] Communities: {comm_path}")
            
            comm_csv_path = output_dir / "node_communities.csv"
            pd.DataFrame({
                'node_id': list(self.communities.keys()),
                'community': list(self.communities.values())
            }).to_csv(comm_csv_path, index=False)
            logger.info(f"[OK] Community CSV: {comm_csv_path}")
        
        # CSV with communities
        df_path = output_dir / "articles_with_communities.csv"
        logger.info(f"Saving articles CSV...")
        self.df.to_csv(df_path, index=False)
        logger.info(f"[OK] Articles: {df_path}")
        
        # Metadata
        meta_path = output_dir / "graph_metadata.json"
        metadata = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'density': float(nx.density(self.graph)),
            'num_communities': len(set(self.communities.values())) if self.communities else None,
            'modularity': float(self.stats.get('modularity', 0)),
            'created_at': datetime.now().isoformat(),
            'builder_version': '3.1_batched_faiss',
            'formats_available': ['gpickle', 'gexf', 'graphml', 'edgelist']
        }
        
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"[OK] Metadata: {meta_path}")
        
        logger.info("="*70)
        logger.info("[SUCCESS] EXPORT COMPLETE")
        logger.info("="*70)


def main():
    print("\n" + "="*70)
    print("[START] CORD-19 ULTRA-FAST GRAPH CONSTRUCTION")
    print("="*70 + "\n")
    
    DATA_PATH = Path("S1_CORD19_Classification/data/processed/cleaned_articles.csv")
    EMBEDDINGS_PATH = Path("S1_CORD19_Classification/data/processed/embeddings.npy")
    OUTPUT_DIR = Path("S1_CORD19_Classification/data/processed")
    
    SIMILARITY_THRESHOLD = 0.75
    MAX_EDGES_PER_NODE = 15
    
    # Load data
    logger.info(f"Loading CSV...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    logger.info(f"[OK] Loaded: {len(df):,} articles")
    
    logger.info(f"Loading embeddings...")
    embeddings = np.load(EMBEDDINGS_PATH)
    logger.info(f"[OK] Loaded: {embeddings.shape}")
    
    # Alignment check
    if len(df) != len(embeddings):
        logger.warning(f"[WARNING] Mismatch: {len(df)} vs {len(embeddings)}")
        min_size = min(len(df), len(embeddings))
        logger.info(f"Aligning to {min_size}")
        df = df.iloc[:min_size].reset_index(drop=True)
        embeddings = embeddings[:min_size]
    
    # Build graph
    builder = FastGraphBuilder(df, embeddings)
    
    graph = builder.build_graph_fast(
        threshold=SIMILARITY_THRESHOLD,
        max_edges_per_node=MAX_EDGES_PER_NODE
    )
    
    # Detect communities
    communities = builder.detect_communities_scalable(
        algorithm='louvain',
        resolution=1.0
    )
    
    # Analyze
    comm_stats = builder.analyze_communities(top_n=10)
    
    # Export
    builder.export_graph(OUTPUT_DIR)
    
    print("\n" + "="*70)
    print("[SUCCESS] PIPELINE COMPLETE")
    print("="*70)
    print(f"\n[SUMMARY] Graph Summary:")
    print(f"  Nodes: {graph.number_of_nodes():,}")
    print(f"  Edges: {graph.number_of_edges():,}")
    print(f"  Communities: {len(set(communities.values()))}")
    print(f"  Modularity: {builder.stats.get('modularity', 0):.4f}")
    print(f"\n[OUTPUT] Output: {OUTPUT_DIR}")
    print(f"\n[FILES] Available formats:")
    print(f"  * .gpickle (NetworkX)")
    print(f"  * .gexf (Gephi)")
    print(f"  * .graphml (General)")
    print(f"  * .csv (Edge list)")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()