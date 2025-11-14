import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION POUR DATASET COMPLET
# ============================================================================
USE_FULL_DATA = True  # Utiliser TOUT le dataset
HIDDEN_DIM = 256
MAX_EPOCHS = 100
PATIENCE = 15
MIN_CLASS_SIZE = 100  # Augmenté pour avoir des classes robustes
REMOVE_OUTLIERS = True
BATCH_TRAINING = True  # Training par mini-batch pour économiser la mémoire

print(f"🚀 Configuration FULL DATASET:")
print(f"  Full data: {USE_FULL_DATA}")
print(f"  Hidden dim: {HIDDEN_DIM}")
print(f"  Max epochs: {MAX_EPOCHS}")
print(f"  Patience: {PATIENCE}")
print(f"  Min class size: {MIN_CLASS_SIZE}")
print(f"  Remove outliers: {REMOVE_OUTLIERS}")
print(f"  Batch training: {BATCH_TRAINING}")


class FullDatasetBuilder:
    """Builder pour le dataset complet"""
    
    def __init__(self, graph_path, embeddings_path, df_path):
        print("\n" + "="*60)
        print("CHARGEMENT DU DATASET COMPLET")
        print("="*60)
        
        with open(graph_path, 'rb') as f:
            self.G = pickle.load(f)
        print(f"✓ Graphe: {self.G.number_of_nodes():,} nœuds, {self.G.number_of_edges():,} arêtes")
        
        self.embeddings = np.load(embeddings_path)
        print(f"✓ Embeddings: {self.embeddings.shape}")
        
        self.df = pd.read_csv(df_path)
        print(f"✓ DataFrame: {len(self.df):,} articles")
        
        self.pyg_data = None
        self.label_encoder = None
        self.class_weights = None
    
    def create_labels_from_communities(self, min_size=MIN_CLASS_SIZE, remove_outliers=REMOVE_OUTLIERS):
        """Crée les labels avec filtrage intelligent"""
        print("\n" + "="*60)
        print("PRÉPARATION DES LABELS")
        print("="*60)
        
        if 'community' not in self.df.columns:
            raise ValueError("Colonne 'community' manquante")
        
        # Retirer les outliers
        if remove_outliers:
            outlier_mask = self.df['community'] == -1
            n_outliers = outlier_mask.sum()
            print(f"✓ Retrait des outliers (-1): {n_outliers:,} articles ({n_outliers/len(self.df)*100:.1f}%)")
            self.df = self.df[~outlier_mask].copy()
            self.embeddings = self.embeddings[~outlier_mask]
        
        # Analyser les communautés
        community_counts = self.df['community'].value_counts()
        print(f"\n✓ Statistiques des communautés:")
        print(f"  Total: {len(community_counts)}")
        print(f"  Top 10:\n{community_counts.head(10)}")
        
        # Filtrer par taille
        valid_communities = community_counts[community_counts >= min_size].index
        mask = self.df['community'].isin(valid_communities)
        
        n_before = len(self.df)
        self.df = self.df[mask].copy().reset_index(drop=True)
        self.embeddings = self.embeddings[mask]
        n_after = len(self.df)
        
        print(f"\n✓ Filtrage (min {min_size} membres):")
        print(f"  Communautés valides: {len(valid_communities)}")
        print(f"  Articles conservés: {n_after:,}/{n_before:,} ({n_after/n_before*100:.1f}%)")
        
        # Créer les labels
        self.label_encoder = LabelEncoder()
        labels = self.label_encoder.fit_transform(self.df['community'])
        
        # Calculer les poids de classe
        print(f"\n✓ Calcul des poids de classe...")
        self.class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(labels),
            y=labels
        )
        self.class_weights = torch.FloatTensor(self.class_weights)
        
        label_counts = pd.Series(labels).value_counts()
        print(f"  Classes finales: {len(self.label_encoder.classes_)}")
        print(f"  Taille des classes:")
        print(f"    Min: {label_counts.min()}, Max: {label_counts.max()}, Médiane: {label_counts.median():.0f}")
        print(f"  Poids de classe: min={self.class_weights.min():.3f}, max={self.class_weights.max():.3f}")
        
        return labels
    
    def update_graph(self):
        """Met à jour le graphe après filtrage"""
        print(f"\n✓ Mise à jour du graphe...")
        n_nodes_before = self.G.number_of_nodes()
        
        # Garder seulement les nœuds présents dans le DataFrame
        valid_nodes = set(range(len(self.df)))
        self.G = self.G.subgraph(valid_nodes).copy()
        
        # Réindexer les nœuds
        mapping = {old: new for new, old in enumerate(sorted(valid_nodes))}
        self.G = nx.relabel_nodes(self.G, mapping)
        
        print(f"  Nœuds: {n_nodes_before:,} → {self.G.number_of_nodes():,}")
        print(f"  Arêtes: {self.G.number_of_edges():,}")
    
    def build_pyg_data(self, labels, train_mask, val_mask, test_mask):
        print("\n" + "="*60)
        print("CONSTRUCTION DU GRAPHE PYTORCH GEOMETRIC")
        print("="*60)
        
        edge_index = torch.tensor(list(self.G.edges())).t().contiguous()
        print(f"✓ Edges: {edge_index.shape}")
        
        x = torch.FloatTensor(self.embeddings)
        print(f"✓ Features: {x.shape}")
        
        y = torch.LongTensor(labels)
        print(f"✓ Labels: {y.shape}")
        
        data = Data(
            x=x, 
            edge_index=edge_index, 
            y=y,
            train_mask=torch.BoolTensor(train_mask),
            val_mask=torch.BoolTensor(val_mask),
            test_mask=torch.BoolTensor(test_mask)
        )
        
        self.pyg_data = data
        
        print(f"\n✓ Graphe PyG complet:")
        print(f"  Nœuds: {data.num_nodes:,}")
        print(f"  Arêtes: {data.num_edges:,}")
        print(f"  Features: {data.num_node_features}")
        print(f"  Classes: {len(torch.unique(data.y))}")
        
        return data
    
    def create_train_val_test_split(self, labels, train_ratio=0.7, val_ratio=0.15):
        print("\n" + "="*60)
        print("CRÉATION DU SPLIT TRAIN/VAL/TEST")
        print("="*60)
        
        n = len(labels)
        indices = np.arange(n)
        
        unique, counts = np.unique(labels, return_counts=True)
        min_count = counts.min()
        print(f"✓ Taille minimale de classe: {min_count}")
        
        if min_count < 3:
            raise ValueError(f"Certaines classes ont moins de 3 échantillons.")
        
        # Split train / temp
        train_idx, temp_idx = train_test_split(
            indices, train_size=train_ratio, stratify=labels, random_state=42
        )
        
        # Split val / test
        temp_labels = labels[temp_idx]
        val_size = val_ratio / (1 - train_ratio)
        val_idx, test_idx = train_test_split(
            temp_idx, train_size=val_size, stratify=temp_labels, random_state=42
        )
        
        train_mask = np.zeros(n, dtype=bool)
        val_mask = np.zeros(n, dtype=bool)
        test_mask = np.zeros(n, dtype=bool)
        
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True
        
        print(f"\n✓ Split créé:")
        print(f"  Train: {train_mask.sum():,} ({train_mask.sum()/n*100:.1f}%)")
        print(f"  Val: {val_mask.sum():,} ({val_mask.sum()/n*100:.1f}%)")
        print(f"  Test: {test_mask.sum():,} ({test_mask.sum()/n*100:.1f}%)")
        
        return train_mask, val_mask, test_mask


class EfficientGNNClassifier(nn.Module):
    """Modèle GNN optimisé pour gros datasets"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super(EfficientGNNClassifier, self).__init__()
        
        # GraphSAGE est plus efficace pour les gros graphes
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, hidden_dim // 2)
        
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        self.fc1 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.fc2 = nn.Linear(hidden_dim // 4, output_dim)
        
        self.dropout = dropout
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)


class FullDataTrainer:
    """Entraîneur optimisé pour gros datasets"""
    
    def __init__(self, model, data, class_weights, device='cpu'):
        self.model = model.to(device)
        self.data = data.to(device)
        self.device = device
        self.class_weights = class_weights.to(device) if class_weights is not None else None
        self.history = {
            'train_loss': [], 'val_loss': [], 
            'train_acc': [], 'val_acc': [],
            'train_f1': [], 'val_f1': []
        }
    
    def train_epoch(self, optimizer):
        self.model.train()
        optimizer.zero_grad()
        
        out = self.model(self.data)
        loss = F.nll_loss(
            out[self.data.train_mask], 
            self.data.y[self.data.train_mask],
            weight=self.class_weights
        )
        
        loss.backward()
        optimizer.step()
        
        pred = out[self.data.train_mask].max(1)[1]
        acc = pred.eq(self.data.y[self.data.train_mask]).sum().item() / self.data.train_mask.sum().item()
        
        return loss.item(), acc
    
    @torch.no_grad()
    def evaluate(self, mask):
        self.model.eval()
        
        out = self.model(self.data)
        loss = F.nll_loss(
            out[mask], 
            self.data.y[mask],
            weight=self.class_weights
        )
        
        pred = out[mask].max(1)[1]
        acc = pred.eq(self.data.y[mask]).sum().item() / mask.sum().item()
        
        return loss.item(), acc
    
    def train(self, epochs=MAX_EPOCHS, lr=0.005, weight_decay=5e-4, patience=PATIENCE):
        print("\n" + "="*60)
        print("ENTRAÎNEMENT DU MODÈLE")
        print("="*60)
        print(f"✓ Configuration:")
        print(f"  Epochs: {epochs}")
        print(f"  Learning rate: {lr}")
        print(f"  Weight decay: {weight_decay}")
        print(f"  Patience: {patience}")
        print(f"  Class weights: {'Activés' if self.class_weights is not None else 'Désactivés'}")
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )
        
        best_val_acc = 0
        patience_counter = 0
        
        print(f"\n✓ Début de l'entraînement...")
        pbar = tqdm(range(epochs), desc="Training")
        
        for epoch in pbar:
            train_loss, train_acc = self.train_epoch(optimizer)
            val_loss, val_acc = self.evaluate(self.data.val_mask)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            scheduler.step(val_acc)
            
            pbar.set_postfix({
                'tr_loss': f'{train_loss:.3f}',
                'tr_acc': f'{train_acc:.3f}',
                'val_acc': f'{val_acc:.3f}'
            })
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_model_full.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n✓ Early stopping à l'epoch {epoch+1}")
                    break
        
        self.model.load_state_dict(torch.load('best_model_full.pt'))
        
        print(f"\n✓ Entraînement terminé!")
        print(f"  Meilleure val accuracy: {best_val_acc:.4f}")
        print(f"  Epochs effectués: {epoch+1}")
        
        return self.history
    
    @torch.no_grad()
    def test(self):
        print("\n" + "="*60)
        print("ÉVALUATION SUR LE TEST SET")
        print("="*60)
        
        self.model.eval()
        out = self.model(self.data)
        
        pred = out[self.data.test_mask].max(1)[1].cpu().numpy()
        true = self.data.y[self.data.test_mask].cpu().numpy()
        
        acc = (pred == true).sum() / len(true)
        print(f"✓ Test accuracy: {acc:.4f}")
        
        return pred, true, acc
    
    def plot_training_history(self):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        epochs = range(len(self.history['train_loss']))
        
        axes[0].plot(epochs, self.history['train_loss'], label='Train Loss', linewidth=2)
        axes[0].plot(epochs, self.history['val_loss'], label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(epochs, self.history['train_acc'], label='Train Accuracy', linewidth=2)
        axes[1].plot(epochs, self.history['val_acc'], label='Val Accuracy', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history_full.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Historique sauvegardé: training_history_full.png")
        plt.close()


class FullDataAnalyzer:
    """Analyse complète des résultats"""
    
    def __init__(self, model, data, label_encoder, df):
        self.model = model
        self.data = data
        self.label_encoder = label_encoder
        self.df = df
    
    def visualize_confusion_matrix(self, pred, true):
        cm = confusion_matrix(true, pred)
        
        n_classes = len(self.label_encoder.classes_)
        figsize = (max(12, n_classes * 0.3), max(10, n_classes * 0.25))
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, cmap='Blues', cbar=True, square=True, 
                   cbar_kws={'label': 'Count'})
        plt.title(f'Matrice de Confusion (FULL DATA - {n_classes} classes)', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('Vrai label', fontsize=12)
        plt.xlabel('Prédiction', fontsize=12)
        plt.tight_layout()
        plt.savefig('confusion_matrix_full.png', dpi=300, bbox_inches='tight')
        print(f"✓ Matrice sauvegardée: confusion_matrix_full.png")
        plt.close()
    
    def get_classification_report(self, pred, true):
        print("\n" + "="*60)
        print("RAPPORT DE CLASSIFICATION DÉTAILLÉ")
        print("="*60)
        
        report = classification_report(
            true, pred, 
            target_names=[str(c) for c in self.label_encoder.classes_],
            digits=4,
            zero_division=0
        )
        print(report)
        
        report_dict = classification_report(
            true, pred, 
            target_names=[str(c) for c in self.label_encoder.classes_],
            output_dict=True,
            zero_division=0
        )
        
        print("\n" + "="*60)
        print("MÉTRIQUES GLOBALES")
        print("="*60)
        print(f"✓ Accuracy: {report_dict['accuracy']:.4f}")
        print(f"✓ Macro avg:")
        print(f"  - Precision: {report_dict['macro avg']['precision']:.4f}")
        print(f"  - Recall: {report_dict['macro avg']['recall']:.4f}")
        print(f"  - F1-score: {report_dict['macro avg']['f1-score']:.4f}")
        print(f"✓ Weighted avg:")
        print(f"  - Precision: {report_dict['weighted avg']['precision']:.4f}")
        print(f"  - Recall: {report_dict['weighted avg']['recall']:.4f}")
        print(f"  - F1-score: {report_dict['weighted avg']['f1-score']:.4f}")
        
        return report_dict
    
    def analyze_top_bottom_classes(self, pred, true):
        print("\n" + "="*60)
        print("ANALYSE PAR CLASSE")
        print("="*60)
        
        report_dict = classification_report(
            true, pred,
            target_names=[str(c) for c in self.label_encoder.classes_],
            output_dict=True,
            zero_division=0
        )
        
        f1_scores = []
        for label in self.label_encoder.classes_:
            f1 = report_dict[str(label)]['f1-score']
            support = report_dict[str(label)]['support']
            precision = report_dict[str(label)]['precision']
            recall = report_dict[str(label)]['recall']
            f1_scores.append((label, f1, precision, recall, support))
        
        f1_scores.sort(key=lambda x: x[1], reverse=True)
        
        print("\n✓ Top 10 classes (meilleur F1-score):")
        for i, (label, f1, prec, rec, sup) in enumerate(f1_scores[:10], 1):
            print(f"  {i:2d}. Classe {label:4d}: F1={f1:.4f} | Prec={prec:.4f} | Rec={rec:.4f} | Support={int(sup):5d}")
        
        print("\n✓ Bottom 10 classes (pire F1-score):")
        for i, (label, f1, prec, rec, sup) in enumerate(f1_scores[-10:], 1):
            print(f"  {i:2d}. Classe {label:4d}: F1={f1:.4f} | Prec={prec:.4f} | Rec={rec:.4f} | Support={int(sup):5d}")
        
        # Distribution F1
        f1_values = [x[1] for x in f1_scores]
        print(f"\n✓ Distribution des F1-scores:")
        print(f"  Min: {min(f1_values):.4f}")
        print(f"  Q1: {np.percentile(f1_values, 25):.4f}")
        print(f"  Médiane: {np.percentile(f1_values, 50):.4f}")
        print(f"  Q3: {np.percentile(f1_values, 75):.4f}")
        print(f"  Max: {max(f1_values):.4f}")
        print(f"  Classes avec F1 > 0.5: {sum(1 for f1 in f1_values if f1 > 0.5)} / {len(f1_values)}")


def run_full_dataset_pipeline():
    """Pipeline sur le dataset complet"""
    
    print("\n" + "="*80)
    print(" "*20 + "🚀 FULL DATASET GNN PIPELINE 🚀")
    print("="*80)
    
    base_path = Path("S1_CORD19_Classification/data/processed")
    
    # 1. Charger toutes les données
    builder = FullDatasetBuilder(
        base_path / "article_graph.gpickle",
        base_path / "embeddings.npy",
        base_path / "articles_with_communities.csv"
    )
    
    # 2. Créer les labels avec filtrage
    labels = builder.create_labels_from_communities(
        min_size=MIN_CLASS_SIZE,
        remove_outliers=REMOVE_OUTLIERS
    )
    
    # 3. Mettre à jour le graphe
    builder.update_graph()
    
    # 4. Split train/val/test
    train_mask, val_mask, test_mask = builder.create_train_val_test_split(labels)
    
    # 5. Construire le graphe PyG
    data = builder.build_pyg_data(labels, train_mask, val_mask, test_mask)
    
    # 6. Créer le modèle
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ Device: {device}")
    
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    model = EfficientGNNClassifier(
        input_dim=data.num_node_features,
        hidden_dim=HIDDEN_DIM,
        output_dim=len(builder.label_encoder.classes_),
        dropout=0.5
    )
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n✓ Modèle GraphSAGE créé:")
    print(f"  Paramètres: {n_params:,}")
    print(f"  Taille estimée: {n_params * 4 / 1e6:.1f} MB")
    
    # 7. Entraîner
    trainer = FullDataTrainer(model, data, builder.class_weights, device)
    history = trainer.train(epochs=MAX_EPOCHS, lr=0.005, patience=PATIENCE)
    
    # 8. Visualiser l'historique
    trainer.plot_training_history()
    
    # 9. Tester
    pred, true, test_acc = trainer.test()
    
    # 10. Analyser en détail
    analyzer = FullDataAnalyzer(model, data, builder.label_encoder, builder.df)
    analyzer.visualize_confusion_matrix(pred, true)
    report_dict = analyzer.get_classification_report(pred, true)
    analyzer.analyze_top_bottom_classes(pred, true)
    
    # 11. Sauvegarder le modèle final
    output_path = Path("S1_CORD19_Classification/models")
    output_path.mkdir(parents=True, exist_ok=True)
    
    model_data = {
        'model_state_dict': model.state_dict(),
        'label_encoder': builder.label_encoder,
        'class_weights': builder.class_weights,
        'history': history,
        'test_acc': test_acc,
        'report': report_dict,
        'config': {
            'n_samples': len(builder.df),
            'n_classes': len(builder.label_encoder.classes_),
            'hidden_dim': HIDDEN_DIM,
            'min_class_size': MIN_CLASS_SIZE,
            'remove_outliers': REMOVE_OUTLIERS
        }
    }
    
    save_path = output_path / "gnn_full_dataset.pt"
    torch.save(model_data, save_path)
    print(f"\n✓ Modèle sauvegardé: {save_path}")
    
    # 12. Résumé final
    print("\n" + "="*80)
    print(" "*25 + "📊 RÉSULTATS FINAUX 📊")
    print("="*80)
    print(f"\n✓ Dataset:")
    print(f"  Articles traités: {len(builder.df):,}")
    print(f"  Classes: {len(builder.label_encoder.classes_)}")
    print(f"  Arêtes du graphe: {data.num_edges:,}")
    print(f"\n✓ Performance:")
    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  Macro F1: {report_dict['macro avg']['f1-score']:.4f}")
    print(f"  Weighted F1: {report_dict['weighted avg']['f1-score']:.4f}")
    print(f"\n✓ Fichiers générés:")
    print(f"  - {save_path}")
    print(f"  - training_history_full.png")
    print(f"  - confusion_matrix_full.png")
    print("\n" + "="*80)
    
    return model, trainer, analyzer


if __name__ == "__main__":
    model, trainer, analyzer = run_full_dataset_pipeline()
    print("\n✅ Pipeline terminé! Prochaine étape: Graph-RAG 🚀")