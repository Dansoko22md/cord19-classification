import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
import pickle
from typing import List, Dict
from collections import defaultdict
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class GraphRAGSimplified:
    """
    Graph-RAG simplifié sans rechargement du modèle
    Utilise les embeddings pré-calculés pour la recherche
    """
    
    def __init__(self, graph_path, embeddings_path, df_path):
        print("🚀 Initialisation du système Graph-RAG (version simplifiée)...")
        
        # Charger les données
        print("   📂 Chargement des données...")
        with open(graph_path, 'rb') as f:
            self.G = pickle.load(f)
        self.embeddings = np.load(embeddings_path)
        self.df = pd.read_csv(df_path)
        
        print(f"   ✅ Graphe: {self.G.number_of_nodes():,} nœuds, {self.G.number_of_edges():,} arêtes")
        print(f"   ✅ Embeddings: {self.embeddings.shape}")
        print(f"   ✅ Articles: {len(self.df):,}")
        
        # Normaliser les embeddings
        self.embeddings_norm = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        
        # Construire l'index des communautés
        self._build_community_index()
        
        # Créer un index de recherche textuelle simple
        self._build_text_index()
        
        print("   ✅ Graph-RAG prêt!")
    
    def _build_community_index(self):
        """Construit un index des articles par communauté"""
        self.community_index = defaultdict(list)
        
        if 'community' in self.df.columns:
            for idx, comm in enumerate(self.df['community']):
                if pd.notna(comm) and comm != -1:
                    self.community_index[comm].append(idx)
            
            print(f"   📊 {len(self.community_index)} communautés indexées")
        else:
            print("   ⚠️  Pas de communautés détectées")
    
    def _build_text_index(self):
        """Construit un index de recherche textuelle simple"""
        print("   📝 Construction de l'index textuel...")
        
        # Créer un dictionnaire de mots -> indices d'articles
        self.text_index = defaultdict(list)
        
        for idx, row in self.df.iterrows():
            # Extraire les mots du titre et de l'abstract
            text = str(row['title']).lower()
            if pd.notna(row['abstract']):
                text += " " + str(row['abstract']).lower()
            
            # Tokenisation simple
            words = set(text.split())
            for word in words:
                if len(word) > 3:  # Ignorer les mots trop courts
                    self.text_index[word].append(idx)
        
        print(f"   📚 {len(self.text_index):,} mots indexés")
    
    def keyword_search(self, query: str, top_k: int = 100) -> List[int]:
        """
        Recherche par mots-clés pour trouver des candidats
        
        Args:
            query: Requête textuelle
            top_k: Nombre maximum de candidats
            
        Returns:
            Liste d'indices d'articles candidats
        """
        query_words = set(query.lower().split())
        query_words = {w for w in query_words if len(w) > 3}
        
        # Compter les occurrences
        article_scores = defaultdict(int)
        for word in query_words:
            if word in self.text_index:
                for idx in self.text_index[word]:
                    article_scores[idx] += 1
        
        # Trier par score
        sorted_articles = sorted(article_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [idx for idx, score in sorted_articles[:top_k]]
    
    def semantic_search(self, query_text: str = None, query_embedding: np.ndarray = None, 
                       top_k: int = 10, candidate_indices: List[int] = None) -> List[Dict]:
        """
        Recherche sémantique basée sur la similarité cosinus
        
        Args:
            query_text: Texte de la requête (utilisé pour keyword search)
            query_embedding: Embedding de la requête (ou calculé depuis un article)
            top_k: Nombre de résultats
            candidate_indices: Liste d'indices candidats (None = tous)
            
        Returns:
            Liste de résultats avec scores
        """
        # Si pas d'embedding fourni, utiliser keyword search + moyenne
        if query_embedding is None:
            if query_text is None:
                raise ValueError("Fournir query_text ou query_embedding")
            
            # Trouver des articles pertinents par mots-clés
            candidates = self.keyword_search(query_text, top_k=50)
            
            if len(candidates) == 0:
                return []
            
            # Utiliser la moyenne des embeddings des candidats comme requête
            query_embedding = np.mean(self.embeddings_norm[candidates[:5]], axis=0)
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Déterminer sur quels articles chercher
        if candidate_indices is not None:
            search_embeddings = self.embeddings_norm[candidate_indices]
            search_indices = candidate_indices
        else:
            search_embeddings = self.embeddings_norm
            search_indices = list(range(len(self.embeddings_norm)))
        
        # Calculer les similarités
        similarities = np.dot(search_embeddings, query_embedding)
        
        # Obtenir les top-k
        top_local_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for local_idx in top_local_indices:
            idx = search_indices[local_idx]
            results.append({
                'index': int(idx),
                'score': float(similarities[local_idx]),
                'title': self.df.iloc[idx]['title'],
                'abstract': self.df.iloc[idx]['abstract'][:300] + '...' if pd.notna(self.df.iloc[idx]['abstract']) else '',
                'source': self.df.iloc[idx]['source_x'] if 'source_x' in self.df.columns else 'Unknown',
                'community': int(self.df.iloc[idx]['community']) if 'community' in self.df.columns else None
            })
        
        return results
    
    def graph_expansion(self, seed_indices: List[int], max_depth: int = 2, max_neighbors: int = 5) -> List[int]:
        """
        Expansion dans le graphe à partir de nœuds seeds
        """
        visited = set(seed_indices)
        to_explore = list(seed_indices)
        
        for depth in range(max_depth):
            next_level = []
            
            for node in to_explore:
                if node not in self.G:
                    continue
                
                neighbors = list(self.G.neighbors(node))
                
                # Trier par poids si disponible
                if len(neighbors) > 0 and self.G.has_edge(node, neighbors[0]):
                    if 'weight' in self.G[node][neighbors[0]]:
                        neighbor_weights = [(n, self.G[node][n]['weight']) for n in neighbors]
                        neighbor_weights.sort(key=lambda x: x[1], reverse=True)
                        neighbors = [n for n, w in neighbor_weights[:max_neighbors]]
                    else:
                        neighbors = neighbors[:max_neighbors]
                
                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_level.append(neighbor)
            
            to_explore = next_level
            
            if not to_explore:
                break
        
        return list(visited)
    
    def hybrid_search(self, query: str, top_k: int = 10, expansion_depth: int = 1) -> List[Dict]:
        """
        Recherche hybride: keyword + sémantique + expansion graphe
        """
        print(f"\n🔍 Recherche hybride: '{query}'")
        
        # 1. Recherche par mots-clés pour avoir des candidats
        print("   1️⃣ Recherche par mots-clés...")
        candidates = self.keyword_search(query, top_k=100)
        
        if len(candidates) == 0:
            print("   ⚠️  Aucun candidat trouvé par mots-clés")
            return []
        
        print(f"   📊 {len(candidates)} candidats trouvés")
        
        # 2. Recherche sémantique sur les candidats
        print("   2️⃣ Recherche sémantique...")
        initial_results = self.semantic_search(
            query_text=query,
            top_k=min(top_k, len(candidates)),
            candidate_indices=candidates
        )
        
        seed_indices = [r['index'] for r in initial_results]
        
        # 3. Expansion dans le graphe
        print(f"   3️⃣ Expansion graphe (profondeur {expansion_depth})...")
        expanded_indices = self.graph_expansion(seed_indices, max_depth=expansion_depth, max_neighbors=3)
        
        print(f"   📊 {len(seed_indices)} → {len(expanded_indices)} articles")
        
        # 4. Re-scorer tous les articles découverts
        print("   4️⃣ Re-scoring...")
        query_embedding = np.mean(self.embeddings_norm[seed_indices[:3]], axis=0)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        final_results = self.semantic_search(
            query_embedding=query_embedding,
            top_k=top_k,
            candidate_indices=expanded_indices
        )
        
        # Marquer les résultats directs
        for result in final_results:
            result['direct_match'] = result['index'] in seed_indices
        
        return final_results
    
    def community_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Recherche au niveau des communautés
        """
        print(f"\n🔍 Recherche par communautés: '{query}'")
        
        # 1. Trouver les articles pertinents
        candidates = self.keyword_search(query, top_k=50)
        initial_results = self.semantic_search(
            query_text=query,
            top_k=5,
            candidate_indices=candidates if candidates else None
        )
        
        # 2. Identifier les communautés pertinentes
        relevant_communities = set()
        for result in initial_results:
            if result['community'] is not None:
                relevant_communities.add(result['community'])
        
        print(f"   📊 {len(relevant_communities)} communautés identifiées")
        
        # 3. Récupérer plus d'articles de ces communautés
        community_articles = []
        for comm in relevant_communities:
            comm_indices = self.community_index.get(comm, [])
            community_articles.extend(comm_indices)
        
        # 4. Re-scorer
        if len(community_articles) > 0:
            query_embedding = np.mean(self.embeddings_norm[[r['index'] for r in initial_results[:3]]], axis=0)
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            
            results = self.semantic_search(
                query_embedding=query_embedding,
                top_k=top_k,
                candidate_indices=community_articles
            )
            
            return results
        
        return initial_results
    
    def find_similar_articles(self, article_index: int, top_k: int = 10, method: str = 'hybrid') -> List[Dict]:
        """
        Trouve des articles similaires à un article donné
        """
        if article_index < 0 or article_index >= len(self.df):
            raise ValueError(f"Index invalide: {article_index}")
        
        article = self.df.iloc[article_index]
        
        if method == 'embedding':
            # Similarité pure
            similarities = np.dot(self.embeddings_norm, self.embeddings_norm[article_index])
            top_indices = np.argsort(similarities)[::-1][1:top_k+1]
            
            results = []
            for idx in top_indices:
                results.append({
                    'index': int(idx),
                    'score': float(similarities[idx]),
                    'title': self.df.iloc[idx]['title'],
                    'abstract': self.df.iloc[idx]['abstract'][:200] + '...' if pd.notna(self.df.iloc[idx]['abstract']) else '',
                })
        else:
            # Utiliser le graphe
            if article_index in self.G:
                neighbors = list(self.G.neighbors(article_index))
                
                if len(neighbors) > 0:
                    neighbor_embeddings = self.embeddings_norm[neighbors]
                    similarities = np.dot(neighbor_embeddings, self.embeddings_norm[article_index])
                    
                    sorted_indices = np.argsort(similarities)[::-1][:top_k]
                    
                    results = []
                    for idx in sorted_indices:
                        neighbor_idx = neighbors[idx]
                        results.append({
                            'index': int(neighbor_idx),
                            'score': float(similarities[idx]),
                            'title': self.df.iloc[neighbor_idx]['title'],
                            'abstract': self.df.iloc[neighbor_idx]['abstract'][:200] + '...' if pd.notna(self.df.iloc[neighbor_idx]['abstract']) else '',
                        })
                else:
                    results = []
            else:
                results = []
        
        return results
    
    def generate_context(self, results: List[Dict], max_articles: int = 5) -> str:
        """Génère un contexte textuel pour RAG"""
        context_parts = []
        
        for i, result in enumerate(results[:max_articles], 1):
            context_parts.append(f"Article {i}: {result['title']}")
            context_parts.append(f"Abstract: {result['abstract']}")
            context_parts.append(f"Relevance Score: {result['score']:.3f}")
            if result.get('community') is not None:
                context_parts.append(f"Topic Cluster: {result['community']}")
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def answer_question(self, question: str, method: str = 'hybrid', top_k: int = 5) -> Dict:
        """
        Répond à une question en utilisant Graph-RAG
        """
        print(f"\n❓ Question: {question}")
        print(f"   Méthode: {method}")
        
        # Rechercher selon la méthode
        if method == 'semantic':
            candidates = self.keyword_search(question, top_k=100)
            results = self.semantic_search(query_text=question, top_k=top_k, candidate_indices=candidates)
        elif method == 'community':
            results = self.community_search(question, top_k=top_k)
        elif method == 'hybrid':
            results = self.hybrid_search(question, top_k=top_k, expansion_depth=1)
        else:
            raise ValueError(f"Méthode inconnue: {method}")
        
        # Générer le contexte
        context = self.generate_context(results, max_articles=5)
        
        # Analyser les communautés
        communities = [r['community'] for r in results if r.get('community') is not None]
        unique_communities = len(set(communities)) if communities else 0
        
        return {
            'question': question,
            'method': method,
            'num_results': len(results),
            'results': results,
            'context': context,
            'communities_covered': unique_communities,
            'top_community': max(set(communities), key=communities.count) if communities else None
        }


# INTERFACE INTERACTIVE SIMPLIFIÉE
class SimpleRAGInterface:
    """Interface en ligne de commande simplifiée"""
    
    def __init__(self, rag_system: GraphRAGSimplified):
        self.rag = rag_system
    
    def run(self):
        print("\n" + "="*60)
        print("🤖 GRAPH-RAG INTERFACE INTERACTIVE")
        print("="*60)
        print("\nCommandes:")
        print("  search <query>     - Recherche hybride")
        print("  semantic <query>   - Recherche sémantique")
        print("  community <query>  - Recherche par communauté")
        print("  similar <index>    - Articles similaires")
        print("  stats              - Statistiques")
        print("  quit               - Quitter")
        print()
        
        while True:
            try:
                user_input = input("🔍 > ").strip()
                
                if not user_input:
                    continue
                
                parts = user_input.split(maxsplit=1)
                command = parts[0].lower()
                
                if command == 'quit':
                    print("👋 Au revoir!")
                    break
                
                elif command == 'stats':
                    self._show_stats()
                
                elif command in ['search', 'semantic', 'community']:
                    if len(parts) < 2:
                        print("❌ Veuillez fournir une requête")
                        continue
                    
                    query = parts[1]
                    method = 'hybrid' if command == 'search' else command
                    self._search(query, method)
                
                elif command == 'similar':
                    if len(parts) < 2:
                        print("❌ Veuillez fournir un index")
                        continue
                    
                    try:
                        index = int(parts[1])
                        self._find_similar(index)
                    except ValueError:
                        print("❌ Index invalide")
                
                else:
                    print(f"❌ Commande inconnue: {command}")
            
            except KeyboardInterrupt:
                print("\n👋 Au revoir!")
                break
            except Exception as e:
                print(f"❌ Erreur: {e}")
    
    def _show_stats(self):
        print("\n📊 STATISTIQUES")
        print("="*60)
        print(f"Articles: {len(self.rag.df):,}")
        print(f"Nœuds: {self.rag.G.number_of_nodes():,}")
        print(f"Arêtes: {self.rag.G.number_of_edges():,}")
        print(f"Communautés: {len(self.rag.community_index)}")
        print(f"Mots indexés: {len(self.rag.text_index):,}")
    
    def _search(self, query: str, method: str):
        answer = self.rag.answer_question(query, method=method, top_k=5)
        
        print(f"\n✅ {answer['num_results']} résultats")
        print(f"📊 {answer['communities_covered']} communautés\n")
        
        for i, result in enumerate(answer['results'], 1):
            print(f"{i}. [{result['score']:.3f}] {result['title']}")
            print(f"   {result['abstract'][:120]}...")
            if result.get('direct_match'):
                print("   🎯 Match direct")
            print()
    
    def _find_similar(self, index: int):
        if index < 0 or index >= len(self.rag.df):
            print(f"❌ Index invalide (0-{len(self.rag.df)-1})")
            return
        
        article = self.rag.df.iloc[index]
        print(f"\n📄 Référence #{index}:")
        print(f"   {article['title']}\n")
        
        results = self.rag.find_similar_articles(index, top_k=5, method='hybrid')
        
        if results:
            print(f"✅ {len(results)} articles similaires:")
            for i, result in enumerate(results, 1):
                print(f"{i}. [{result['score']:.3f}] {result['title']}")
                print(f"   {result['abstract'][:100]}...")
                print()
        else:
            print("❌ Aucun article similaire")


# EXÉCUTION PRINCIPALE
if __name__ == "__main__":
    print("="*80)
    print(" "*25 + "PHASE 5: GRAPH-RAG SYSTEM")
    print("="*80)
    
    base_path = Path("S1_CORD19_Classification/data/processed")
    
    # Initialiser le système
    rag = GraphRAGSimplified(
        graph_path=base_path / "article_graph.gpickle",
        embeddings_path=base_path / "embeddings.npy",
        df_path=base_path / "articles_with_communities.csv"
    )
    
    # Exemples de requêtes
    example_queries = [
        "COVID-19 vaccine efficacy",
        "coronavirus transmission",
        "clinical trials treatment",
    ]
    
    print("\n" + "="*80)
    print(" "*25 + "📝 EXEMPLES DE REQUÊTES")
    print("="*80)
    
    for query in example_queries:
        answer = rag.answer_question(query, method='hybrid', top_k=3)
        
        print(f"\n{'='*80}")
        print(f"✅ '{query}' → {answer['num_results']} résultats")
        print(f"📊 Communautés: {answer['communities_covered']}\n")
        
        for i, result in enumerate(answer['results'], 1):
            print(f"{i}. [Score: {result['score']:.3f}]")
            print(f"   {result['title']}")
            print(f"   {result['abstract'][:120]}...")
            print()
    
    # Mode interactif
    print("\n" + "="*80)
    print(" "*25 + "🎯 MODE INTERACTIF")
    print("="*80)
    
    interface = SimpleRAGInterface(rag)
    interface.run()