import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
from tqdm import tqdm
import torch
import pickle
import json

class EmbeddingsGenerator:
    """Classe pour générer des embeddings d'articles scientifiques"""
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialise le générateur d'embeddings
        
        Args:
            model_name: Nom du modèle Sentence-BERT à utiliser
                       - 'all-MiniLM-L6-v2': Rapide, 384 dimensions (RECOMMANDÉ)
                       - 'all-mpnet-base-v2': Plus précis, 768 dimensions
                       - 'allenai-specter': Spécialisé scientifique, 768 dimensions
        """
        print("🤖 GÉNÉRATION DES EMBEDDINGS")
        print("="*60)
        
        self.model_name = model_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"\n📦 Chargement du modèle: {model_name}")
        print(f"   Device: {self.device}")
        
        self.model = SentenceTransformer(model_name)
        self.model.to(self.device)
        
        print(f"   ✅ Modèle chargé")
        print(f"   Dimensions: {self.model.get_sentence_embedding_dimension()}")
        
    def prepare_texts(self, df, max_length=512):
        """
        Prépare les textes pour l'embedding
        
        Args:
            df: DataFrame avec colonnes 'title' et 'abstract'
            max_length: Longueur maximale en tokens
            
        Returns:
            Liste de textes préparés
        """
        print(f"\n📝 Préparation des textes...")
        
        texts = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Textes"):
            # Combiner titre + abstract
            title = str(row['title']) if pd.notna(row['title']) else ''
            abstract = str(row['abstract']) if pd.notna(row['abstract']) else ''
            
            # Format: "Title. Abstract"
            text = f"{title}. {abstract}".strip()
            
            # Tronquer si trop long (approximation: 4 chars = 1 token)
            if len(text) > max_length * 4:
                text = text[:max_length * 4]
            
            texts.append(text)
        
        print(f"   ✅ {len(texts)} textes préparés")
        print(f"   Longueur moyenne: {np.mean([len(t) for t in texts]):.0f} caractères")
        
        return texts
    
    def generate_embeddings(self, texts, batch_size=32, show_progress=True):
        """
        Génère les embeddings pour une liste de textes
        
        Args:
            texts: Liste de textes
            batch_size: Taille des batches
            show_progress: Afficher la barre de progression
            
        Returns:
            Array numpy d'embeddings (n_texts, embedding_dim)
        """
        print(f"\n🔄 Génération des embeddings...")
        print(f"   Batch size: {batch_size}")
        print(f"   Total: {len(texts)} textes")
        
        # Générer les embeddings
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normaliser pour cosine similarity
        )
        
        print(f"\n   ✅ Embeddings générés")
        print(f"   Shape: {embeddings.shape}")
        print(f"   Type: {embeddings.dtype}")
        print(f"   Taille mémoire: {embeddings.nbytes / 1024**2:.2f} MB")
        
        return embeddings
    
    def analyze_embeddings(self, embeddings, sample_size=1000):
        """
        Analyse la qualité des embeddings
        
        Args:
            embeddings: Array d'embeddings
            sample_size: Nombre d'échantillons pour l'analyse
        """
        print(f"\n📊 ANALYSE DES EMBEDDINGS")
        print("="*60)
        
        # Statistiques de base
        print(f"\n📏 Statistiques:")
        print(f"   - Moyenne: {embeddings.mean():.4f}")
        print(f"   - Std: {embeddings.std():.4f}")
        print(f"   - Min: {embeddings.min():.4f}")
        print(f"   - Max: {embeddings.max():.4f}")
        
        # Norme (devrait être ~1 si normalisé)
        norms = np.linalg.norm(embeddings, axis=1)
        print(f"\n📐 Normes (devrait être ~1):")
        print(f"   - Moyenne: {norms.mean():.4f}")
        print(f"   - Min: {norms.min():.4f}")
        print(f"   - Max: {norms.max():.4f}")
        
        # Similarité entre échantillons aléatoires
        if len(embeddings) > sample_size:
            print(f"\n🔍 Analyse de similarité (échantillon de {sample_size})...")
            indices = np.random.choice(len(embeddings), sample_size, replace=False)
            sample_emb = embeddings[indices]
            
            # Calculer matrice de similarité
            from sklearn.metrics.pairwise import cosine_similarity
            sim_matrix = cosine_similarity(sample_emb)
            
            # Exclure la diagonale (similarité avec soi-même)
            mask = ~np.eye(sample_size, dtype=bool)
            similarities = sim_matrix[mask]
            
            print(f"\n   Similarités cosinus:")
            print(f"   - Moyenne: {similarities.mean():.4f}")
            print(f"   - Médiane: {np.median(similarities):.4f}")
            print(f"   - Std: {similarities.std():.4f}")
            print(f"   - Min: {similarities.min():.4f}")
            print(f"   - Max: {similarities.max():.4f}")
            
            # Distribution
            print(f"\n   Distribution:")
            bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
            hist, _ = np.histogram(similarities, bins=bins)
            for i in range(len(bins)-1):
                print(f"   - [{bins[i]:.1f}-{bins[i+1]:.1f}]: {hist[i]:,} paires ({hist[i]/len(similarities)*100:.1f}%)")
    
    def visualize_embeddings(self, embeddings, df, sample_size=2000):
        """
        Visualise les embeddings en 2D avec UMAP/t-SNE
        
        Args:
            embeddings: Array d'embeddings
            df: DataFrame avec métadonnées
            sample_size: Nombre d'échantillons à visualiser
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.manifold import TSNE
        
        print(f"\n🎨 Visualisation des embeddings...")
        
        # Échantillonner si trop grand
        if len(embeddings) > sample_size:
            indices = np.random.choice(len(embeddings), sample_size, replace=False)
            emb_sample = embeddings[indices]
            df_sample = df.iloc[indices]
        else:
            emb_sample = embeddings
            df_sample = df
        
        # Réduction de dimension avec t-SNE
        print(f"   Réduction de dimension (t-SNE)...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        emb_2d = tsne.fit_transform(emb_sample)
        
        # Créer la visualisation
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Plot 1: Points simples
        ax1 = axes[0]
        scatter1 = ax1.scatter(emb_2d[:, 0], emb_2d[:, 1], 
                              alpha=0.5, s=10, c='steelblue')
        ax1.set_title(f'Embeddings en 2D (t-SNE)\n{len(emb_sample)} articles', 
                     fontsize=14)
        ax1.set_xlabel('t-SNE dimension 1')
        ax1.set_ylabel('t-SNE dimension 2')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Coloré par source si disponible
        ax2 = axes[1]
        if 'source_x' in df_sample.columns:
            # Prendre les top sources
            top_sources = df_sample['source_x'].value_counts().head(10).index
            colors = df_sample['source_x'].apply(
                lambda x: list(top_sources).index(x) if x in top_sources else -1
            )
            
            scatter2 = ax2.scatter(emb_2d[:, 0], emb_2d[:, 1], 
                                  c=colors, alpha=0.6, s=10, 
                                  cmap='tab10', vmin=0, vmax=9)
            ax2.set_title('Embeddings colorés par source\n(Top 10 sources)', 
                         fontsize=14)
            
            # Légende
            handles = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=plt.cm.tab10(i/10), 
                                 markersize=8, label=source[:20]) 
                      for i, source in enumerate(top_sources)]
            ax2.legend(handles=handles, loc='best', fontsize=8)
        else:
            ax2.scatter(emb_2d[:, 0], emb_2d[:, 1], 
                       alpha=0.5, s=10, c='steelblue')
            ax2.set_title('Embeddings en 2D (t-SNE)', fontsize=14)
        
        ax2.set_xlabel('t-SNE dimension 1')
        ax2.set_ylabel('t-SNE dimension 2')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('embeddings_visualization.png', dpi=300, bbox_inches='tight')
        print(f"   📊 Visualisation sauvegardée: embeddings_visualization.png")
        plt.show()
    
    def save_embeddings(self, embeddings, df, output_dir):
        """
        Sauvegarde les embeddings et métadonnées
        
        Args:
            embeddings: Array d'embeddings
            df: DataFrame avec métadonnées
            output_dir: Répertoire de sortie
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n💾 Sauvegarde des embeddings...")
        
        # Sauvegarder les embeddings
        emb_path = output_dir / 'embeddings.npy'
        np.save(emb_path, embeddings)
        print(f"   ✅ Embeddings: {emb_path}")
        print(f"      Taille: {emb_path.stat().st_size / 1024**2:.2f} MB")
        
        # Sauvegarder les métadonnées
        meta_path = output_dir / 'embeddings_metadata.json'
        metadata = {
            'model_name': self.model_name,
            'embedding_dim': int(embeddings.shape[1]),
            'num_articles': int(embeddings.shape[0]),
            'normalized': True,
            'device': self.device
        }
        
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"   ✅ Métadonnées: {meta_path}")
        
        # Sauvegarder le DataFrame
        df_path = output_dir / 'articles_for_embeddings.csv'
        df.to_csv(df_path, index=False)
        print(f"   ✅ Articles: {df_path}")
    
    def load_embeddings(self, embeddings_path):
        """
        Charge des embeddings sauvegardés
        
        Args:
            embeddings_path: Chemin vers le fichier .npy
            
        Returns:
            Array d'embeddings
        """
        print(f"\n📂 Chargement des embeddings...")
        embeddings = np.load(embeddings_path)
        print(f"   ✅ Shape: {embeddings.shape}")
        return embeddings


# INSTALLATION REQUISE:
# pip install sentence-transformers
# ou ajoutez dans requirements.txt: sentence-transformers>=2.2.0

# EXEMPLE D'UTILISATION
if __name__ == "__main__":
    print("\n" + "="*60)
    print("PHASE 2: GÉNÉRATION DES EMBEDDINGS")
    print("="*60)
    
    # 1. Charger les données nettoyées
    data_path = Path("S1_CORD19_Classification/data/processed/cleaned_articles.csv")
    
    print(f"\n📂 Chargement des données...")
    df = pd.read_csv(data_path)
    print(f"   ✅ {len(df):,} articles chargés")
    
    # Échantillon pour test rapide (commenter pour traiter tout)
    SAMPLE_SIZE = None  # Mettre None pour traiter tout le dataset
    if SAMPLE_SIZE and len(df) > SAMPLE_SIZE:
        print(f"\n⚠️  Mode échantillon: {SAMPLE_SIZE} articles")
        df = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
    
    # 2. Initialiser le générateur d'embeddings
    # Options de modèles:
    # - 'all-MiniLM-L6-v2': Rapide, léger (RECOMMANDÉ pour gros datasets)
    # - 'all-mpnet-base-v2': Plus précis mais plus lent
    # - 'allenai-specter': Spécialisé pour articles scientifiques
    
    generator = EmbeddingsGenerator(model_name='all-MiniLM-L6-v2')
    
    # 3. Préparer les textes
    texts = generator.prepare_texts(df, max_length=512)
    
    # 4. Générer les embeddings
    embeddings = generator.generate_embeddings(
        texts, 
        batch_size=64,  # Ajuster selon la RAM/VRAM disponible
        show_progress=True
    )
    
    # 5. Analyser les embeddings
    generator.analyze_embeddings(embeddings, sample_size=1000)
    
    # 6. Visualiser (optionnel, peut prendre du temps)
    generator.visualize_embeddings(embeddings, df, sample_size=2000)
    
    # 7. Sauvegarder
    output_dir = Path("S1_CORD19_Classification/data/processed")
    generator.save_embeddings(embeddings, df, output_dir)
    
    print("\n" + "="*60)
    print("✅ PHASE 2 TERMINÉE!")
    print("="*60)
    print(f"\n📦 Résumé:")
    print(f"   - Articles traités: {len(df):,}")
    print(f"   - Dimensions: {embeddings.shape}")
    print(f"   - Modèle: {generator.model_name}")
    print(f"   - Fichiers sauvegardés dans: {output_dir}")
    print(f"\n➡️  Prochaine étape: Construction du graphe (Phase 3)")
    print(f"   Utilisez article_graph_builder.py avec ces embeddings")