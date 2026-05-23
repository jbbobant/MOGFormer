import os
import sys
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataset import MultiOmicsDataset
from src.utils.baseline_fusion import extract_numpy_from_loader, build_early_fusion_matrix
from src.models.baselines import MultiOmicsBaselineTrainer


def main(args):
    print("=====================================================")
    print("  Starting Baseline Model Training & Evaluation")
    print("=====================================================")
    
    # 1. Load Dataset
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
    # Extract ordered class names for the confusion matrix display
    class_names = [k for k, v in sorted(label_map.items(), key=lambda item: item[1])]
    
    print(f"Total Patients: {len(dataset)} | Classes ({num_classes}): {label_map}")
    
    # 2. Train/Validation Split (Identical to main Transformer pipeline)
    print(f"\n[2/5] Splitting Data ({args.train_split*100}% Train / {(1-args.train_split)*100}% Val)...")
    train_size = int(args.train_split * len(dataset))
    val_size = len(dataset) - train_size
    
    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    # Note: drop_last=False is strictly required here to ensure no patients are left behind 
    # during the NumPy extraction. Batch size doesn't matter for sklearn, so we use a large one.
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, drop_last=False)

    # 3. Extract to NumPy and Compute Sample Weights
    print("\n[3/5] Extracting tensors to NumPy arrays...")
    rna_train, cnv_train, methy_train, y_train = extract_numpy_from_loader(train_loader)
    rna_val, cnv_val, methy_val, y_val = extract_numpy_from_loader(val_loader)
    
    # Replicate the Focal Loss inverse frequency weighting dynamically for XGBoost
    class_counts = np.bincount(y_train)
    freqs = class_counts / len(y_train)
    # Using normalized square root of inverse frequency
    inv_sq_weights = 1.0 / np.sqrt(freqs)
    normalized_weights = inv_sq_weights / np.sum(inv_sq_weights) * num_classes
    
    # Map the class weights to the individual samples in the training set
    sample_weights = np.array([normalized_weights[label] for label in y_train])
    
    # 4. Execute Modality Fusion
    print("\n[4/5] Executing Early Integration (Concatenation)...")
    X_train = build_early_fusion_matrix(rna_train, cnv_train, methy_train)
    X_val = build_early_fusion_matrix(rna_val, cnv_val, methy_val)
    
    print(f"Fused Training Matrix Shape: {X_train.shape}")
    print(f"Fused Validation Matrix Shape: {X_val.shape}")

    # 5. Initialize Trainer and Execute
    print("\n[5/5] Initializing Baseline Models...")
    trainer = MultiOmicsBaselineTrainer(
        num_classes=num_classes,
        class_names=class_names,
        viz_dir=args.viz_dir
    )
    
    trainer.fit_and_evaluate(X_train, y_train, X_val, y_val, sample_weights)
    trainer.plot_best_confusion_matrix()
    
    print("\n=====================================================")
    print("  Baseline Evaluation Complete.")
    print(f"  Visualizations saved to: {args.viz_dir}/")
    print("=====================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Traditional Machine Learning Baselines.")
    
    # Data & Paths
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Path to processed tensors")
    parser.add_argument("--viz_dir", type=str, default="viz/baselines", help="Directory to save output plots")
    
    # Pipeline Settings
    parser.add_argument("--train_split", type=float, default=0.8, help="Fraction of data for training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data splitting (must match train.py)")
    
    args = parser.parse_args()
    main(args)