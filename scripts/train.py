import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader, random_split


# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataset import MultiOmicsDataset
from src.models.classifier import MultiOmicsGraphClassifier
from src.training.trainer import MultiOmicsTrainer
from src.data.dynamic_graph import DynamicDropEdgeRWPE


def main(args):
    # 1. Device Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=====================================================")
    print(f"  Starting Training on Device: {device}")
    print(f"=====================================================")
    
    # 2. Load Dataset
    print("\n[1/5] Initializing Multi-Omics Dataset...")
    dataset = MultiOmicsDataset(
        clinical_path=os.path.join(args.data_dir, "processed_clinical.csv"),
        rna_path=os.path.join(args.data_dir, "processed_rna.csv"),
        cnv_path=os.path.join(args.data_dir, "processed_cnv.csv"),
        methy_path=os.path.join(args.data_dir, "processed_methy.csv"),
        label_col="SUBTYPE"
    )
    
    label_map = dataset.get_label_mapping()
    num_classes = len(label_map)
    print(f"Total Patients: {len(dataset)} | Classes ({num_classes}): {label_map}")
    
    # 3. Train/Validation Split
    print(f"\n[2/5] Splitting Data ({args.train_split*100}% Train / {(1-args.train_split)*100}% Val)...")
    train_size = int(args.train_split * len(dataset))
    val_size = len(dataset) - train_size
    
    # Using a fixed generator seed ensures reproducibility across runs
    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 4. Load Structural Priors (Graph PE and Base Adjacency)
    print("\n[3/5] Loading Graph Positional Encodings and Topology...")
    pe_path = os.path.join(args.data_dir, f"graph_pe_{args.pe_method}.pt")
    adj_path = os.path.join(args.data_dir, "base_adj.pt")
    
    if not os.path.exists(pe_path) or not os.path.exists(adj_path):
        raise FileNotFoundError("Graph PE or Base Adjacency not found. Run run_preprocessing.py first.")
    
    graph_pe = torch.load(pe_path)
    base_adj = torch.load(adj_path) 
    print(f"Graph PE loaded with shape: {graph_pe.shape}")
    
    # 5. Initialize Model
    print("\n[4/5] Building Multi-Omics Graph Transformer...")
    model = MultiOmicsGraphClassifier(
        num_classes=num_classes,
        d=args.d,
        pe_dim=graph_pe.shape[1],  
        mini_heads=args.mini_heads,
        global_heads=args.global_heads,
        global_layers=args.global_layers,
        dropout=args.dropout,
        rna_dropout_prob=args.rna_dropout
    )
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model instantiated with {total_params:,} trainable parameters.")
    
    # ---> NEW: Instantiate the Dynamic DropEdge Module <---
    dynamic_pe = DynamicDropEdgeRWPE(
        base_adj=base_adj,
        pe_dim=graph_pe.shape[1],
        drop_prob=0.10 # Drops 10% of edges
    )
    
    # 6. Initialize Trainer and Start
    print("\n[5/5] Initializing Trainer...")
    trainer = MultiOmicsTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        dynamic_pe_module=dynamic_pe, 
        device=device,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        save_dir=args.save_dir,
        use_dropedge=args.use_dropedge  
    )
    
    trainer.fit(epochs=args.epochs)
    print("\n Training Complete! Best model saved to checkpoints.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Patient-Specific Multi-Omics Graph Transformer.")
    
    # Data & Paths
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Path to processed tensors")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save model weights")
    parser.add_argument("--pe_method", type=str, default="rwpe", choices=["laplacian", "rwpe"])
    
    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--train_split", type=float, default=0.8, help="Fraction of data for training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data splitting")
    
    # Architecture Hyperparameters
    parser.add_argument("--d", type=int, default=64, help="Token embedding dimension")
    parser.add_argument("--mini_heads", type=int, default=4, help="Attention heads in Mini-Transformer")
    parser.add_argument("--global_heads", type=int, default=8, help="Attention heads in Global Transformer")
    parser.add_argument("--global_layers", type=int, default=4, help="Number of layers in Global Transformer")
    parser.add_argument("--dropout", type=float, default=0.1, help="Network dropout rate")
    parser.add_argument("--rna_dropout", type=float, default=0.15, help="Modality dropout rate for mRNA")
    parser.add_argument("--use_dropedge", default=True, action="store_true", help="Enable dynamic DropEdge regularization")
    
    args = parser.parse_args()
    main(args)