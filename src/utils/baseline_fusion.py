import torch
import numpy as np
from typing import Tuple, Dict
from torch.utils.data import DataLoader


def extract_numpy_from_loader(
    dataloader: DataLoader
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Iterates through a PyTorch DataLoader to extract the full dataset into memory.
    Required for feeding data into traditional ML baselines like SVM or XGBoost.
    
    Args:
        dataloader: PyTorch DataLoader containing the multi-omics dataset.
        
    Returns:
        Tuple containing full NumPy arrays for (RNA, CNV, Methylation, Labels).
    """
    rna_list, cnv_list, methy_list, labels_list = [], [], [], []
    
    # Temporarily disable gradients since we are only extracting data
    with torch.no_grad():
        for batch in dataloader:
            # Assuming your dataset yields: x_rna, x_cnv, x_methy, y
            x_rna, x_cnv, x_methy, y = batch
            
            rna_list.append(x_rna.cpu().numpy())
            cnv_list.append(x_cnv.cpu().numpy())
            methy_list.append(x_methy.cpu().numpy())
            labels_list.append(y.cpu().numpy())
            
    # Concatenate all mini-batches into full dataset arrays
    full_rna = np.vstack(rna_list)
    full_cnv = np.vstack(cnv_list)
    full_methy = np.vstack(methy_list)
    full_labels = np.concatenate(labels_list)
    
    return full_rna, full_cnv, full_methy, full_labels


def build_early_fusion_matrix(
    rna: np.ndarray, 
    cnv: np.ndarray, 
    methy: np.ndarray
) -> np.ndarray:
    """
    Executes Early Integration (Feature Concatenation).
    Combines the three modalities horizontally into a single feature matrix.
    
    Args:
        rna: Array of shape (n_samples, n_rna_features)
        cnv: Array of shape (n_samples, n_cnv_features)
        methy: Array of shape (n_samples, n_methy_features)
        
    Returns:
        np.ndarray of shape (n_samples, n_rna + n_cnv + n_methy)
    """
    # Strict ordering is critical to prevent data misalignment during inference
    early_fusion_matrix = np.hstack([rna, cnv, methy])
    return early_fusion_matrix


def prepare_late_fusion_data(
    rna: np.ndarray, 
    cnv: np.ndarray, 
    methy: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Prepares the data for Late Integration (Ensemble).
    Returns a dictionary mapping each modality name to its isolated feature matrix
    to facilitate iterating through independent baseline models.
    
    Args:
        rna: Array of shape (n_samples, n_rna_features)
        cnv: Array of shape (n_samples, n_cnv_features)
        methy: Array of shape (n_samples, n_methy_features)
        
    Returns:
        Dictionary mapping modality names to their respective data matrices.
    """
    return {
        "rna": rna,
        "cnv": cnv,
        "methy": methy
    }