import os
import sys
import argparse

# Add the project root to the Python path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.preprocess import MultiOmicsPreprocessor
from src.data.graph_utils import GraphPEGenerator

def main(args):
    print("=====================================================")
    print("  Step 1: Multi-Omics Alignment & Variance Filtering ")
    print("=====================================================")
    
    preprocessor = MultiOmicsPreprocessor(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        top_k_genes=args.top_k
    )
    
    preprocessor.run_pipeline(
        rna_file="data_rna_seq_v2_rsem.csv",
        cnv_file="data_cnv.csv",
        methy_file="data_methylation450.csv",
        clin_file="data_clinical.csv"
    )
    
    print("\n=====================================================")
    print("  Step 2: Structural Graph PE Generation (STRINGdb)  ")
    print("=====================================================")
    
    generator = GraphPEGenerator(
        ppi_path=os.path.join(args.raw_dir, "9606.protein.links.v12.0.txt"),
        alias_path=os.path.join(args.raw_dir, "9606.protein.aliases.v12.0.txt"),
        gene_list_path=os.path.join(args.processed_dir, "hvg_gene_list.txt"),
        output_dir=args.processed_dir
    )
    
    generator.generate(method=args.pe_method, pe_dim=args.pe_dim)
    
    print("\n All Multi-Omics & Graph Preprocessing Successfully Completed!")
    print(f"Data is ready for PyTorch in: {args.processed_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute the full multi-omics preprocessing pipeline.")
    
    # Directory arguments
    parser.add_argument("--raw_dir", type=str, default="data/raw", 
                        help="Directory containing the raw multi-omics and STRINGdb files.")
    parser.add_argument("--processed_dir", type=str, default="data/processed", 
                        help="Directory to output the PyTorch-ready processed files.")
    
    # Hyperparameters
    parser.add_argument("--top_k", type=int, default=500, 
                        help="Number of highly variable genes to extract via consensus ranking.")
    parser.add_argument("--pe_method", type=str, default="rwpe", choices=["laplacian", "rwpe"], 
                        help="Method to generate Graph Positional Encodings.")
    parser.add_argument("--pe_dim", type=int, default=16, 
                        help="Dimensionality of the generated Graph Positional Encodings.")
    
    args = parser.parse_args()
    main(args)