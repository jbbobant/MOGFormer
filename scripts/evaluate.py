import os
import sys
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataset import MultiOmicsDataset
from src.models.classifier import MultiOmicsGraphClassifier

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=====================================================")
    print(f"  Evaluating Model on Device: {device}")
    print(f"=====================================================")

    # 1. Load Dataset & Recreate the exact Validation Split
    print("\n[1/4] Loading Data & Reconstructing Validation Set...")
    dataset = MultiOmicsDataset(
        clinical_path=os.path.join(args.data_dir, "processed_clinical.csv"),
        rna_path=os.path.join(args.data_dir, "processed_rna.csv"),
        cnv_path=os.path.join(args.data_dir, "processed_cnv.csv"),
        methy_path=os.path.join(args.data_dir, "processed_methy.csv"),
        label_col="SUBTYPE"
    )
    
    label_map = dataset.get_label_mapping()
    num_classes = len(label_map)
    inverse_label_map = {v: k for k, v in label_map.items()}
    
    # Recreate the exact split from training using the same seed
    train_size = int(args.train_split * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(args.seed)
    _, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Validation set contains {len(val_dataset)} patients.")

    # 2. Load Structural Priors (Graph PE)
    print("\n[2/4] Loading Graph Positional Encodings...")
    pe_path = os.path.join(args.data_dir, f"graph_pe_{args.pe_method}.pt")
    graph_pe = torch.load(pe_path).to(device)

    # 3. Initialize Model and Load Weights
    print("\n[3/4] Loading Model Weights...")
    checkpoint_path = os.path.join(args.checkpoint_dir, "best_model.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = MultiOmicsGraphClassifier(
        num_classes=num_classes,
        d=args.d,
        pe_dim=graph_pe.shape[1],
        mini_heads=args.mini_heads,
        global_heads=args.global_heads,
        global_layers=args.global_layers
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded weights from Best Epoch (Val F1: {checkpoint.get('best_val_f1', 'Unknown'):.4f})")

    # 4. Evaluation Loop
    print("\n[4/4] Running Inference...")
    all_preds = []
    all_labels = []
    
    # Interpretability hooks
    sample_patient_id = None
    sample_attn_weights = None

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            rna = batch['mRNA'].to(device)
            cnv = batch['CNV'].to(device)
            methy = batch['methy'].to(device)
            labels = batch['label'].to(device)
            
            # Pass return_attention=True to trigger the interpretability output
            outputs = model(rna, cnv, methy, graph_pe, return_attention=True)
            logits = outputs['logits']
            
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Capture attention weights for the first batch as a proof-of-concept
            if i == 0 and args.interpret:
                sample_patient_id = batch['patient_id'][0] # Grab first patient in batch
                sample_attn_weights = outputs['intra_attn_weights'][0].cpu() # Shape: (Num_Heads, 4, 4)

    # 5. Clinical Metrics Reporting
    print("\n=====================================================")
    print("                 CLINICAL EVALUATION                 ")
    print("=====================================================")
    
    target_names = [inverse_label_map[i] for i in range(num_classes)]
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=target_names, zero_division=0))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    
    # 6. Native Interpretability Output
    if args.interpret and sample_attn_weights is not None:
        print("\n=====================================================")
        print("          NATIVE INTERPRETABILITY STRATEGY           ")
        print("=====================================================")
        print(f"Extracted Intra-Gene Attention Matrix for Patient: {sample_patient_id}")
        print(f"Attention Weights Shape: {sample_attn_weights.shape} (Heads, Sequence, Sequence)")
        print("Sequence Index Map: 0:[z_cls], 1:[mRNA], 2:[CNV], 3:[Methylation]")
        print("\nNote: Pass this tensor to a heatmap visualizer to perform 'Deep Drill-Down' into modality drivers.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the Multi-Omics Graph Transformer.")
    
    # Data & Paths
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--pe_method", type=str, default="laplacian")
    
    # Evaluation configuration
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--train_split", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--interpret", action="store_true", help="Extract and print sample attention weights")
    
    # Architecture Hyperparameters (MUST match train.py)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--mini_heads", type=int, default=4)
    parser.add_argument("--global_heads", type=int, default=8)
    parser.add_argument("--global_layers", type=int, default=4)
    
    args = parser.parse_args()
    main(args)