import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay

class MultiOmicsBaselineTrainer:
    def __init__(
        self, 
        num_classes: int, 
        class_names: List[str], 
        viz_dir: str = "viz/baselines"
    ):
        """
        Initializes the baseline models and ensures the visualization directory exists.
        """
        self.num_classes = num_classes
        self.class_names = class_names
        self.viz_dir = viz_dir
        os.makedirs(self.viz_dir, exist_ok=True)
        
        # Best model tracking
        self.best_model_name = None
        self.best_macro_f1 = 0.0
        self.best_y_pred = None
        self.best_y_true = None

    def _get_models(self) -> Dict:
        """Instantiates the baseline models."""
        return {
            "Logistic_Regression": LogisticRegression(
                max_iter=1000, 
                class_weight="balanced" # Approximates your inverse-freq weighting natively
            ),
            "Random_Forest": RandomForestClassifier(
                n_estimators=10, # Start small for warm-start tracking
                warm_start=True,
                class_weight="balanced",
                n_jobs=-1,
                random_state=42
            ),
            "SVM_RBF": SVC(
                kernel='rbf', 
                probability=True, 
                class_weight="balanced",
                random_state=42
            ),
            "XGBoost": XGBClassifier(
                objective='multi:softmax',
                num_class=self.num_classes,
                eval_metric='mlogloss',
                use_label_encoder=False,
                random_state=42
                # Sample weights will be passed during fit() for XGBoost
            )
        }

    def _plot_training_curves(
        self, 
        model_name: str, 
        macro_f1_history: List[float], 
        class_f1_history: Dict[int, List[float]]
    ):
        """Generates and saves the F1 learning curves for ensemble models."""
        plt.figure(figsize=(10, 6))
        
        # Plot Macro F1
        plt.plot(macro_f1_history, label="Macro F1 (Overall)", linewidth=3, color='black', linestyle='--')
        
        # Plot Per-Class F1
        colors = plt.cm.tab10(np.linspace(0, 1, self.num_classes))
        for cls_idx in range(self.num_classes):
            plt.plot(
                class_f1_history[cls_idx], 
                label=f"Class: {self.class_names[cls_idx]}", 
                color=colors[cls_idx]
            )
            
        plt.title(f"Validation F1-Score Trajectory: {model_name}")
        plt.xlabel("Iterations / Trees Added")
        plt.ylabel("F1 Score")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(self.viz_dir, f"{model_name}_learning_curve.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved learning curve to {save_path}")

    def plot_best_confusion_matrix(self):
        """Plots the confusion matrix for the overall best performing baseline."""
        if self.best_y_pred is None:
            print("No models have been trained yet.")
            return

        cm = confusion_matrix(self.best_y_true, self.best_y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.class_names)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation='vertical')
        plt.title(f"Best Baseline Confusion Matrix\n({self.best_model_name} | Macro F1: {self.best_macro_f1:.4f})")
        plt.tight_layout()
        
        save_path = os.path.join(self.viz_dir, "best_baseline_confusion_matrix.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"\nSaved Best Confusion Matrix ({self.best_model_name}) to {save_path}")

    def fit_and_evaluate(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray, 
        X_val: np.ndarray, 
        y_val: np.ndarray,
        sample_weights: np.ndarray
    ):
        """
        Executes the training and evaluation loop across all baselines,
        generating visualization curves where mathematically applicable.
        """
        models = self._get_models()
        
        for name, model in models.items():
            print(f"\n--- Training {name} ---")
            
            if name == "Random_Forest":
                # Simulate 'epochs' by adding trees iteratively via warm_start
                macro_history = []
                class_history = {i: [] for i in range(self.num_classes)}
                
                for step in range(1, 11): # Train up to 100 trees in increments of 10
                    model.n_estimators = step * 10
                    model.fit(X_train, y_train)
                    
                    y_pred = model.predict(X_val)
                    macro_f1 = f1_score(y_val, y_pred, average='macro')
                    class_f1s = f1_score(y_val, y_pred, average=None)
                    
                    macro_history.append(macro_f1)
                    for i, cf1 in enumerate(class_f1s):
                        class_history[i].append(cf1)
                        
                self._plot_training_curves(name, macro_history, class_history)
                final_pred = y_pred
                final_macro = macro_history[-1]

            elif name == "XGBoost":
                # XGBoost can track validation metrics natively per tree
                model.fit(
                    X_train, y_train,
                    sample_weight=sample_weights,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
                
                # To get F1 per tree, we have to iterate through the staged predictions
                macro_history = []
                class_history = {i: [] for i in range(self.num_classes)}
                
                # XGBoost doesn't return staged predict natively in sklearn API easily, 
                # so we calculate it on the final model to keep things efficient.
                final_pred = model.predict(X_val)
                final_macro = f1_score(y_val, final_pred, average='macro')
                
                # Note: Exact per-tree F1 for XGBoost requires extensive custom callbacks.
                # We log the final result for standard output.
                print(f"{name} completed training.")
                
            else:
                # Logistic Regression and SVM (Single Pass Optimization)
                model.fit(X_train, y_train)
                final_pred = model.predict(X_val)
                final_macro = f1_score(y_val, final_pred, average='macro')
                print(f"{name} completed convex optimization (No epoch curves generated).")

            # Final validation metrics for this model
            class_f1s = f1_score(y_val, final_pred, average=None)
            print(f"{name} Final Validation Macro F1: {final_macro:.4f}")
            
            # Update best model tracker
            if final_macro > self.best_macro_f1:
                self.best_macro_f1 = final_macro
                self.best_model_name = name
                self.best_y_pred = final_pred
                self.best_y_true = y_val