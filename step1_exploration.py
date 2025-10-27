import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Configuration
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

class CORD19Explorer:
    """Classe pour explorer le dataset CORD-19"""
    
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.metadata = None
        
    def load_metadata(self):
        """Charge le fichier metadata.csv"""
        print("📂 Chargement de metadata.csv...")
        self.metadata = pd.read_csv(self.data_path / 'metadata.csv', low_memory=False)
        print(f"✅ {len(self.metadata)} articles chargés")
        return self.metadata
    
    def explore_structure(self):
        """Explore la structure du dataset"""
        if self.metadata is None:
            self.load_metadata()
        
        print("\n" + "="*60)
        print("📊 STRUCTURE DU DATASET")
        print("="*60)
        
        # Dimensions
        print(f"\n🔢 Dimensions: {self.metadata.shape}")
        print(f"   - Nombre d'articles: {self.metadata.shape[0]:,}")
        print(f"   - Nombre de colonnes: {self.metadata.shape[1]}")
        
        # Colonnes disponibles
        print("\n📋 Colonnes disponibles:")
        for i, col in enumerate(self.metadata.columns, 1):
            print(f"   {i}. {col}")
        
        # Informations sur les données
        print("\n📈 Informations par colonne:")
        info_df = pd.DataFrame({
            'Non-Null': self.metadata.count(),
            'Null': self.metadata.isnull().sum(),
            '% Null': (self.metadata.isnull().sum() / len(self.metadata) * 100).round(2),
            'Type': self.metadata.dtypes
        })
        print(info_df)
        
        return info_df
    
    def analyze_content(self):
        """Analyse le contenu des articles"""
        print("\n" + "="*60)
        print("📝 ANALYSE DU CONTENU")
        print("="*60)
        
        # Articles avec abstract
        has_abstract = self.metadata['abstract'].notna().sum()
        print(f"\n✍️  Articles avec abstract: {has_abstract:,} ({has_abstract/len(self.metadata)*100:.1f}%)")
        
        # Articles avec titre
        has_title = self.metadata['title'].notna().sum()
        print(f"📄 Articles avec titre: {has_title:,} ({has_title/len(self.metadata)*100:.1f}%)")
        
        # Sources
        print(f"\n🌐 Sources principales:")
        top_sources = self.metadata['source_x'].value_counts().head(10)
        for source, count in top_sources.items():
            print(f"   - {source}: {count:,} articles")
        
        # Années de publication
        if 'publish_time' in self.metadata.columns:
            self.metadata['year'] = pd.to_datetime(self.metadata['publish_time'], errors='coerce').dt.year
            print(f"\n📅 Distribution par année:")
            year_dist = self.metadata['year'].value_counts().sort_index().tail(10)
            for year, count in year_dist.items():
                if pd.notna(year):
                    print(f"   - {int(year)}: {count:,} articles")
    
    def filter_quality_articles(self, min_abstract_length=100):
        """Filtre les articles de qualité"""
        print("\n" + "="*60)
        print("🔍 FILTRAGE DES ARTICLES DE QUALITÉ")
        print("="*60)
        
        # Critères de qualité
        quality_mask = (
            self.metadata['title'].notna() &
            self.metadata['abstract'].notna() &
            (self.metadata['abstract'].str.len() > min_abstract_length)
        )
        
        quality_articles = self.metadata[quality_mask].copy()
        
        print(f"\n✅ Articles de qualité: {len(quality_articles):,}")
        print(f"   (avec titre + abstract > {min_abstract_length} caractères)")
        print(f"\n📉 Réduction: {len(self.metadata) - len(quality_articles):,} articles filtrés")
        print(f"   ({(len(self.metadata) - len(quality_articles))/len(self.metadata)*100:.1f}% du total)")
        
        return quality_articles
    
    def extract_text_for_nlp(self, articles_df):
        """Prépare les textes pour le NLP"""
        print("\n" + "="*60)
        print("📚 PRÉPARATION POUR NLP")
        print("="*60)
        
        # Combiner titre + abstract
        articles_df['full_text'] = (
            articles_df['title'].fillna('') + ' ' + 
            articles_df['abstract'].fillna('')
        ).str.strip()
        
        # Statistiques de longueur
        articles_df['text_length'] = articles_df['full_text'].str.len()
        
        print(f"\n📏 Statistiques de longueur des textes:")
        print(f"   - Moyenne: {articles_df['text_length'].mean():.0f} caractères")
        print(f"   - Médiane: {articles_df['text_length'].median():.0f} caractères")
        print(f"   - Min: {articles_df['text_length'].min():.0f} caractères")
        print(f"   - Max: {articles_df['text_length'].max():.0f} caractères")
        
        return articles_df
    
    def visualize_statistics(self, articles_df):
        """Crée des visualisations"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Distribution des longueurs de texte
        axes[0, 0].hist(articles_df['text_length'], bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('Longueur du texte (caractères)')
        axes[0, 0].set_ylabel('Nombre d\'articles')
        axes[0, 0].set_title('Distribution des longueurs de texte')
        axes[0, 0].axvline(articles_df['text_length'].mean(), color='red', 
                           linestyle='--', label='Moyenne')
        axes[0, 0].legend()
        
        # 2. Top sources
        top_sources = articles_df['source_x'].value_counts().head(10)
        axes[0, 1].barh(range(len(top_sources)), top_sources.values)
        axes[0, 1].set_yticks(range(len(top_sources)))
        axes[0, 1].set_yticklabels(top_sources.index)
        axes[0, 1].set_xlabel('Nombre d\'articles')
        axes[0, 1].set_title('Top 10 des sources')
        axes[0, 1].invert_yaxis()
        
        # 3. Articles par année
        if 'year' in articles_df.columns:
            year_counts = articles_df['year'].value_counts().sort_index()
            axes[1, 0].plot(year_counts.index, year_counts.values, marker='o')
            axes[1, 0].set_xlabel('Année')
            axes[1, 0].set_ylabel('Nombre d\'articles')
            axes[1, 0].set_title('Évolution temporelle des publications')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Statistiques globales
        stats_text = f"""
        STATISTIQUES GLOBALES
        
        📊 Total d'articles: {len(articles_df):,}
        
        📝 Avec abstract: {articles_df['abstract'].notna().sum():,}
        
        📄 Avec titre: {articles_df['title'].notna().sum():,}
        
        📏 Longueur moyenne: {articles_df['text_length'].mean():.0f} car.
        
        🌐 Nombre de sources: {articles_df['source_x'].nunique()}
        """
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, 
                       verticalalignment='center', family='monospace')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig('cord19_exploration.png', dpi=300, bbox_inches='tight')
        print("\n📊 Visualisation sauvegardée: cord19_exploration.png")
        plt.show()
        
        return fig


# EXEMPLE D'UTILISATION
if __name__ == "__main__":
    # Chemin vers vos données
    DATA_PATH = "S1_CORD19_Classification/data/raw"
    
    # Initialiser l'explorateur
    explorer = CORD19Explorer(DATA_PATH)
    
    # 1. Charger et explorer
    metadata = explorer.load_metadata()
    info = explorer.explore_structure()
    
    # 2. Analyser le contenu
    explorer.analyze_content()
    
    # 3. Filtrer les articles de qualité
    quality_articles = explorer.filter_quality_articles(min_abstract_length=100)
    
    # 4. Préparer pour NLP
    nlp_ready = explorer.extract_text_for_nlp(quality_articles)
    
    # 5. Visualiser
    explorer.visualize_statistics(nlp_ready)
    
    # 6. Sauvegarder les données nettoyées
    output_path = Path("S1_CORD19_Classification/data/processed")
    output_path.mkdir(parents=True, exist_ok=True)
    
    nlp_ready.to_csv(output_path / "cleaned_articles.csv", index=False)
    print(f"\n💾 Données nettoyées sauvegardées: {output_path / 'cleaned_articles.csv'}")
    print(f"   - {len(nlp_ready):,} articles prêts pour le NLP")