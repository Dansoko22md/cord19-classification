import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Essayer d'importer community louvain
try:
    import community.community_louvain as community_louvain
    HAS_LOUVAIN = True
except ImportError:
    print("⚠️  python-louvain non installé, utilisation de l'algorithme greedy")
    HAS_LOUVAIN = False

class ArticleGraphBuilder:
    """Classe pour construire un graphe d'articles scientifiques"""
    
    def __init__(self, df, embeddings):
        """
        Initialise le constructeur de graphe
        
        Args:
            df: DataFrame contenant les métadonnées des articles
            embeddings: Array numpy des embeddings
        """
        self.df = df.reset_index(drop=True)
        self.embeddings = embeddings
        self.graph = None
        self.communities = None
        
        print(f"🔧 Initialisation du constructeur de graphe")
        print(f"   - {len(df)} articles")
        print(f"   - Embeddings: {embeddings.shape}")
    
    def build_similarity_graph(self, threshold=0.7, max_edges_per_node=10):
        """
        Construit un graphe basé sur la similarité cosinus
        
        Args:
            threshold: Seuil de similarité minimum pour créer une arête
            max_edges_per_node: Nombre maximum de connexions par nœud
            
        Returns:
            NetworkX Graph
        """
        print(f"\n🏗️  Construction du graphe de similarité...")
        print(f"   - Seuil: {threshold}")
        print(f"   - Max arêtes/nœud: {max_edges_per_node}")
        
        # Créer le graphe
        G = nx.Graph()
        
        # Ajouter les nœuds avec attributs
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Ajout des nœuds"):
            G.add_node(idx, 
                      title=str(row['title'])[:100] if pd.notna(row['title']) else 'No title',
                      abstract=str(row['abstract'])[:200] if pd.notna(row['abstract']) else '',
                      source=str(row['source_x']) if 'source_x' in row else 'unknown')
        
        # Calculer la similarité cosinus
        print("\n   📊 Calcul de la matrice de similarité...")
        similarity_matrix = cosine_similarity(self.embeddings)
        
        # Ajouter les arêtes
        print("   🔗 Ajout des arêtes...")
        edge_count = 0
        
        for i in tqdm(range(len(similarity_matrix)), desc="Création des liens"):
            # Obtenir les indices triés par similarité décroissante
            similarities = similarity_matrix[i]
            sorted_indices = np.argsort(similarities)[::-1]
            
            # Prendre les top k voisins (en excluant le nœud lui-même)
            count = 0
            for j in sorted_indices[1:]:  # Exclure l'article lui-même
                sim_score = similarities[j]
                
                if sim_score >= threshold and count < max_edges_per_node:
                    if not G.has_edge(i, j):
                        G.add_edge(i, j, weight=float(sim_score))
                        edge_count += 1
                        count += 1
                elif sim_score < threshold:
                    break  # Arrêter si en dessous du seuil
        
        self.graph = G
        
        print(f"\n✅ Graphe construit:")
        print(f"   - Nœuds: {G.number_of_nodes():,}")
        print(f"   - Arêtes: {G.number_of_edges():,}")
        print(f"   - Densité: {nx.density(G):.6f}")
        print(f"   - Composantes connexes: {nx.number_connected_components(G)}")
        
        return G
    
    def detect_communities(self, algorithm='louvain'):
        """
        Détecte les communautés (clusters thématiques)
        
        Args:
            algorithm: 'louvain' ou 'greedy'
            
        Returns:
            Dictionnaire {node: community_id}
        """
        print(f"\n🔍 Détection de communautés ({algorithm})...")
        
        if self.graph is None:
            raise ValueError("Le graphe doit être construit d'abord")
        
        # Utiliser le plus grand composant connecté
        if nx.number_connected_components(self.graph) > 1:
            largest_cc = max(nx.connected_components(self.graph), key=len)
            G_connected = self.graph.subgraph(largest_cc).copy()
            print(f"   ⚠️  Utilisation du plus grand composant: {len(G_connected)} nœuds")
        else:
            G_connected = self.graph
        
        if algorithm == 'louvain' and HAS_LOUVAIN:
            self.communities = community_louvain.best_partition(G_connected)
            modularity = community_louvain.modularity(self.communities, G_connected)
        else:
            # Algorithme greedy (backup)
            if not HAS_LOUVAIN:
                print("   ⚠️  Louvain non disponible, utilisation de greedy")
            communities_gen = nx.community.greedy_modularity_communities(G_connected)
            self.communities = {}
            for i, comm in enumerate(communities_gen):
                for node in comm:
                    self.communities[node] = i
            modularity = nx.community.modularity(G_connected, communities_gen)
        
        # Ajouter les nœuds isolés (communauté -1)
        for node in self.graph.nodes():
            if node not in self.communities:
                self.communities[node] = -1
        
        # Statistiques
        num_communities = len(set(self.communities.values()))
        
        print(f"✅ Communautés détectées:")
        print(f"   - Nombre: {num_communities}")
        print(f"   - Modularité: {modularity:.4f}")
        
        # Distribution des tailles
        comm_sizes = pd.Series(self.communities.values()).value_counts().sort_index()
        print(f"\n📊 Distribution des tailles:")
        print(f"   - Moyenne: {comm_sizes.mean():.1f} articles/cluster")
        print(f"   - Médiane: {comm_sizes.median():.1f} articles/cluster")
        print(f"   - Min: {comm_sizes.min()} articles/cluster")
        print(f"   - Max: {comm_sizes.max()} articles/cluster")
        
        return self.communities
    
    def analyze_communities(self, top_n=5):
        """
        Analyse les communautés détectées
        
        Args:
            top_n: Nombre de top communautés à analyser
            
        Returns:
            DataFrame avec les statistiques des communautés
        """
        print(f"\n📊 ANALYSE DES COMMUNAUTÉS")
        print("="*60)
        
        if self.communities is None:
            raise ValueError("Les communautés doivent être détectées d'abord")
        
        # Ajouter les communautés au DataFrame
        self.df['community'] = self.df.index.map(self.communities)
        
        # Statistiques par communauté
        comm_stats = []
        
        for comm_id in sorted(set(self.communities.values())):
            if comm_id == -1:  # Ignorer les isolés
                continue
                
            articles = self.df[self.df['community'] == comm_id]
            
            comm_stats.append({
                'community_id': comm_id,
                'size': len(articles),
                'avg_connections': np.mean([self.graph.degree(node) 
                                           for node in articles.index if node in self.graph]),
                'sample_titles': articles['title'].head(3).tolist()
            })
        
        stats_df = pd.DataFrame(comm_stats).sort_values('size', ascending=False)
        
        # Afficher les top communautés
        print(f"\n🏆 Top {top_n} communautés par taille:")
        for i, row in stats_df.head(top_n).iterrows():
            print(f"\n   Communauté {row['community_id']}:")
            print(f"   - Taille: {row['size']} articles")
            print(f"   - Connexions moyennes: {row['avg_connections']:.1f}")
            print(f"   - Exemples de titres:")
            for j, title in enumerate(row['sample_titles'], 1):
                title_str = str(title)[:80] if pd.notna(title) else 'No title'
                print(f"      {j}. {title_str}...")
        
        return stats_df
    
    def visualize_graph(self, max_nodes=500, layout='spring'):
        """
        Visualise le graphe avec les communautés
        
        Args:
            max_nodes: Nombre maximum de nœuds à afficher
            layout: Type de layout ('spring', 'kamada_kawai', 'circular')
        """
        print(f"\n🎨 Visualisation du graphe...")
        
        if self.graph is None:
            raise ValueError("Le graphe doit être construit d'abord")
        
        # Prendre un sous-graphe si trop grand
        if self.graph.number_of_nodes() > max_nodes:
            print(f"   ⚠️  Graphe trop grand, affichage d'un échantillon de {max_nodes} nœuds")
            # Prendre les nœuds avec le plus de connexions
            degrees = dict(self.graph.degree())
            top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:max_nodes]
            G_viz = self.graph.subgraph(top_nodes).copy()
        else:
            G_viz = self.graph
        
        # Configuration de la figure
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        # Layout
        print(f"   🎯 Calcul du layout ({layout})...")
        if layout == 'spring':
            pos = nx.spring_layout(G_viz, k=0.5, iterations=50, seed=42)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G_viz)
        elif layout == 'circular':
            pos = nx.circular_layout(G_viz)
        else:
            pos = nx.spring_layout(G_viz, seed=42)
        
        # Graphe 1: Sans couleurs de communautés
        ax1 = axes[0]
        nx.draw_networkx_nodes(G_viz, pos, node_size=30, alpha=0.6, 
                              node_color='steelblue', ax=ax1)
        nx.draw_networkx_edges(G_viz, pos, alpha=0.2, width=0.5, ax=ax1)
        ax1.set_title(f"Graphe d'Articles (Structure)\n{G_viz.number_of_nodes()} nœuds, "
                     f"{G_viz.number_of_edges()} arêtes", fontsize=14)
        ax1.axis('off')
        
        # Graphe 2: Avec couleurs de communautés
        if self.communities is not None:
            ax2 = axes[1]
            
            # Couleurs des communautés
            node_colors = [self.communities.get(node, 0) for node in G_viz.nodes()]
            
            nx.draw_networkx_nodes(G_viz, pos, node_size=30, alpha=0.7,
                                  node_color=node_colors, cmap='tab20', ax=ax2)
            nx.draw_networkx_edges(G_viz, pos, alpha=0.2, width=0.5, ax=ax2)
            
            ax2.set_title(f"Graphe avec Communautés Thématiques\n"
                         f"{len(set(self.communities.values()))} clusters détectés",
                         fontsize=14)
            ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig('graph_visualization.png', dpi=300, bbox_inches='tight')
        print(f"   📊 Visualisation sauvegardée: graph_visualization.png")
        plt.show()
    
    def get_graph_statistics(self):
        """Calcule et affiche les statistiques du graphe"""
        print(f"\n📈 STATISTIQUES DU GRAPHE")
        print("="*60)
        
        if self.graph is None:
            raise ValueError("Le graphe doit être construit d'abord")
        
        G = self.graph
        
        # Statistiques de base
        print(f"\n🔢 Métriques de base:")
        print(f"   - Nœuds: {G.number_of_nodes():,}")
        print(f"   - Arêtes: {G.number_of_edges():,}")
        print(f"   - Densité: {nx.density(G):.6f}")
        
        # Degrés
        degrees = [d for n, d in G.degree()]
        print(f"\n📊 Distribution des degrés:")
        print(f"   - Degré moyen: {np.mean(degrees):.2f}")
        print(f"   - Degré médian: {np.median(degrees):.0f}")
        print(f"   - Degré max: {max(degrees)}")
        print(f"   - Degré min: {min(degrees)}")
        
        # Composantes connexes
        components = list(nx.connected_components(G))
        print(f"\n🔗 Connectivité:")
        print(f"   - Composantes connexes: {len(components)}")
        print(f"   - Taille plus grande composante: {len(max(components, key=len))}")
        
        # Centralité (sur un échantillon si trop grand)
        if G.number_of_nodes() < 1000:
            print(f"\n⭐ Centralité:")
            degree_cent = nx.degree_centrality(G)
            top_central = sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"   Top 5 nœuds centraux:")
            for node, cent in top_central:
                title = str(self.df.iloc[node]['title'])[:60] if pd.notna(self.df.iloc[node]['title']) else 'No title'
                print(f"      - {title}... (centralité: {cent:.4f})")
        
        # Modularité
        if self.communities is not None and HAS_LOUVAIN:
            # Calculer sur le plus grand composant
            largest_cc = max(nx.connected_components(G), key=len)
            G_connected = G.subgraph(largest_cc).copy()
            communities_connected = {k: v for k, v in self.communities.items() if k in G_connected}
            modularity = community_louvain.modularity(communities_connected, G_connected)
            print(f"\n🎯 Qualité des clusters:")
            print(f"   - Modularité: {modularity:.4f}")
        
        return {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'density': nx.density(G),
            'avg_degree': np.mean(degrees),
            'num_communities': len(set(self.communities.values())) if self.communities else None
        }
    
    def save_graph(self, path):
        """Sauvegarde le graphe et les communautés"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarder le graphe
        nx.write_gpickle(self.graph, path)
        
        # Sauvegarder les communautés
        if self.communities is not None:
            comm_path = path.parent / (path.stem + '_communities.pkl')
            with open(comm_path, 'wb') as f:
                pickle.dump(self.communities, f)
        
        print(f"\n💾 Graphe sauvegardé: {path}")
    
    def export_for_gephi(self, path):
        """Exporte le graphe au format GEXF pour Gephi"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Ajouter les attributs de communauté
        if self.communities is not None:
            for node in self.graph.nodes():
                self.graph.nodes[node]['community'] = self.communities.get(node, -1)
        
        nx.write_gexf(self.graph, path)
        print(f"\n💾 Graphe exporté pour Gephi: {path}")


# EXEMPLE D'UTILISATION
if __name__ == "__main__":
    print("="*60)
    print("PHASE 3: CONSTRUCTION DU GRAPHE D'ARTICLES")
    print("="*60)
    
    # 1. Charger les données et embeddings
    data_path = Path("S1_CORD19_Classification/data/processed/cleaned_articles.csv")
    embeddings_path = Path("S1_CORD19_Classification/data/processed/embeddings.npy")
    
    print(f"\n📂 Chargement des données...")
    df = pd.read_csv(data_path)
    embeddings = np.load(embeddings_path)
    
    print(f"   ✅ Articles CSV: {len(df):,}")
    print(f"   ✅ Embeddings: {embeddings.shape}")
    
    # Vérifier la cohérence
    if len(df) != len(embeddings):
        print(f"\n⚠️  ATTENTION: Incohérence détectée!")
        print(f"   Articles: {len(df)}, Embeddings: {len(embeddings)}")
        print(f"   Alignement sur le minimum...")
        min_size = min(len(df), len(embeddings))
        df = df.iloc[:min_size].reset_index(drop=True)
        embeddings = embeddings[:min_size]
        print(f"   ✅ Aligné sur {min_size} articles")
    
    # Échantillon pour test rapide
    SAMPLE_SIZE = 5000  # Mettre None pour traiter tout
    if SAMPLE_SIZE and len(df) > SAMPLE_SIZE:
        print(f"\n⚠️  Mode échantillon: {SAMPLE_SIZE} articles")
        indices = np.random.choice(len(df), SAMPLE_SIZE, replace=False)
        df = df.iloc[indices].reset_index(drop=True)
        embeddings = embeddings[indices]
    
    # 2. Construire le graphe
    builder = ArticleGraphBuilder(df, embeddings)
    
    # Paramètres du graphe
    graph = builder.build_similarity_graph(
        threshold=0.75,  # Seuil de similarité
        max_edges_per_node=15  # Connexions max par article
    )
    
    # 3. Détecter les communautés
    communities = builder.detect_communities(algorithm='louvain')
    
    # 4. Analyser les communautés
    stats = builder.analyze_communities(top_n=10)
    
    # 5. Obtenir les statistiques
    graph_stats = builder.get_graph_statistics()
    
    # 6. Visualiser
    builder.visualize_graph(max_nodes=500, layout='spring')
    
    # 7. Sauvegarder
    output_dir = Path("S1_CORD19_Classification/data/processed")
    builder.save_graph(output_dir / "article_graph.gpickle")
    builder.export_for_gephi(output_dir / "article_graph.gexf")
    
    # 8. Sauvegarder le DataFrame avec communautés
    df_with_comm = builder.df.copy()
    df_with_comm.to_csv(output_dir / "articles_with_communities.csv", index=False)
    
    print("\n" + "="*60)
    print("✅ PHASE 3 TERMINÉE!")
    print("="*60)
    print(f"\n📦 Résumé:")
    print(f"   - Graphe: {graph.number_of_nodes():,} nœuds, {graph.number_of_edges():,} arêtes")
    print(f"   - Communautés: {len(set(communities.values()))}")
    print(f"   - Fichiers sauvegardés dans: {output_dir}")
    print(f"\n🎯 Fichiers générés:")
    print(f"   - article_graph.gpickle : Graphe NetworkX")
    print(f"   - article_graph.gexf : Pour Gephi")
    print(f"   - articles_with_communities.csv : Articles avec clusters")
    print(f"   - graph_visualization.png : Visualisation")
    print(f"\n➡️  Prochaine étape: Analyse approfondie ou Graph-RAG")