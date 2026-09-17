#!/usr/bin/env python3
"""
Cross-conflict concept vector analysis (Section 4.1, "Conflict Awareness via
Concept Vectors").

Given a causal LM and paired consistent / conflict prompts, this module
measures how linearly separable "a conflict is present" is in the residual
stream at every layer:

- Hidden states of the last prompt token are captured with a forward hook on
  model.model.layers[layer_idx].
- The concept vector is the normalized mean difference
  v = mean(H_conflict) - mean(H_consistent).
- Conflict awareness is reported as the ROC-AUC of projecting held-out hidden
  states onto v under 5-fold stratified cross-validation; the direction is
  fitted on the training folds only, so the estimate is not optimistic.
- A TF-IDF + logistic regression baseline (max_iter=1000, 80/20 split) checks
  whether shallow lexical cues alone could explain the separability.
- Cosine similarity between the concept vectors of different conflict types
  and the best-awareness layer per type are computed for cross-type comparison.

The analyzer is driven by analysis/run_cross_conflict_analysis.py and
analysis/run_subfolder_analysis.py; it does not load models itself.

Example:
    from analysis.concept_vector.cross_conflict_concept_vector import CrossConflictConceptVectorAnalyzer
    analyzer = CrossConflictConceptVectorAnalyzer(model, tokenizer, model_name, device="cuda")
    analyzer.analyze_all_conflict_types({"temporal_conflict": (consistent_texts, conflict_texts)})
    analyzer.save_results("results/analysis/cross_conflict/concept_vector")
"""

import os
import torch
import numpy as np
import logging
from typing import List, Dict, Tuple
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine

import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams

# =============================================================================
# Configuration for reproducibility
# =============================================================================
DEFAULT_TEST_SIZE = 0.2
DEFAULT_N_FOLDS = 5
DEFAULT_RANDOM_STATE = 42

logging.basicConfig(
    format="%(asctime)s - %(levelname)s %(name)s %(lineno)s: %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class CrossConflictConceptVectorAnalyzer:
    """
    cross-conflict concept vector analyzer
    compares representation space characteristics across conflict types
    """

    def __init__(self, model, tokenizer, model_name: str, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.device = device
        self.num_layers = len(model.model.layers)

        # store concept vectors per conflict type
        # {conflict_type: {layer_idx: concept_vector}}
        self.concept_vectors = {}

        # store auc scores per conflict type
        # {conflict_type: {layer_idx: auc_score}}
        self.auc_scores = {}

    @torch.no_grad()
    def extract_hidden_states(
        self,
        texts: List[str],
        layer_idx: int,
        batch_size: int = 1
    ) -> torch.Tensor:
        """
        extract hidden states from specified layer

        Args:
            texts: input text list
            layer_idx: layer index
            batch_size: batch size

        Returns:
            hidden_states: [num_samples, hidden_dim]
        """
        all_hidden_states = []

        # define hook
        captured_hidden_states = []

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                captured_hidden_states.append(output[0].detach())
            else:
                captured_hidden_states.append(output.detach())

        # register hook
        target_layer = self.model.model.layers[layer_idx]
        handle = target_layer.register_forward_hook(hook_fn)

        for i in tqdm(range(0, len(texts), batch_size), desc=f"Layer {layer_idx}", leave=False):
            batch_texts = texts[i:i+batch_size]

            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Forward
            captured_hidden_states.clear()
            _ = self.model(**inputs)

            # get last token hidden state
            hidden = captured_hidden_states[0][:, -1, :]
            all_hidden_states.append(hidden.cpu())

        handle.remove()

        all_hidden_states = torch.cat(all_hidden_states, dim=0)
        return all_hidden_states

    def compute_concept_vector(
        self,
        conflict_hiddens: torch.Tensor,
        consistent_hiddens: torch.Tensor
    ) -> torch.Tensor:
        """
        compute the concept vector: v_conflict = mean(H_conflict) - mean(H_consistent)

        Args:
            conflict_hiddens: conflict sample hidden states [num_conflict, hidden_dim]
            consistent_hiddens: consistent sample hidden states [num_consistent, hidden_dim]

        Returns:
            concept_vector: normalized concept vector [hidden_dim]
        """
        conflict_mean = conflict_hiddens.mean(dim=0)
        consistent_mean = consistent_hiddens.mean(dim=0)

        concept_vector = conflict_mean - consistent_mean
        concept_vector = concept_vector / (concept_vector.norm() + 1e-8)

        return concept_vector

    def compute_awareness_auc(
        self,
        conflict_hiddens: torch.Tensor,
        consistent_hiddens: torch.Tensor,
        concept_vector: torch.Tensor
    ) -> float:
        """
        [DEPRECATED] compute conflict awareness auc score using linear projection
        WARNING: This uses the same data for training and testing - results are optimistic!
        Use compute_awareness_auc_with_split instead.
        """
        # Linear projection method
        conflict_scores = torch.matmul(conflict_hiddens, concept_vector).float().cpu().numpy()
        consistent_scores = torch.matmul(consistent_hiddens, concept_vector).float().cpu().numpy()

        # prepare auc computation
        all_scores = np.concatenate([conflict_scores, consistent_scores])
        labels = np.array([1] * len(conflict_scores) + [0] * len(consistent_scores))

        auc = roc_auc_score(labels, all_scores)

        return auc

    def compute_awareness_auc_with_split(
        self,
        conflict_hiddens: torch.Tensor,
        consistent_hiddens: torch.Tensor,
        test_size: float = DEFAULT_TEST_SIZE,
        n_folds: int = DEFAULT_N_FOLDS,
        random_state: int = DEFAULT_RANDOM_STATE,
        use_cv: bool = True
    ) -> Dict[str, float]:
        """
        Compute conflict awareness AUC with proper train/test split.
        
        This method trains the concept vector on training data only and
        evaluates on held-out test data to avoid optimistic AUC estimates.

        Args:
            conflict_hiddens: conflict sample hidden states [n_conflict, hidden_dim]
            consistent_hiddens: consistent sample hidden states [n_consistent, hidden_dim]
            test_size: fraction of data for testing (if not using CV)
            n_folds: number of folds for cross-validation
            random_state: random seed for reproducibility
            use_cv: if True, use k-fold CV; if False, use single train/test split

        Returns:
            Dict with 'auc_mean', 'auc_std', 'auc_folds' (if CV)
        """
        # Combine data
        X = torch.cat([conflict_hiddens, consistent_hiddens], dim=0).float().cpu().numpy()
        y = np.array([1] * len(conflict_hiddens) + [0] * len(consistent_hiddens))

        if use_cv:
            # K-Fold Cross-Validation
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
            fold_aucs = []

            for train_idx, test_idx in skf.split(X, y):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # Compute concept vector on TRAIN set only
                train_conflict = X_train[y_train == 1]
                train_consistent = X_train[y_train == 0]
                
                concept_vector = train_conflict.mean(axis=0) - train_consistent.mean(axis=0)
                concept_vector = concept_vector / (np.linalg.norm(concept_vector) + 1e-8)

                # Evaluate on TEST set
                test_scores = X_test @ concept_vector
                fold_auc = roc_auc_score(y_test, test_scores)
                fold_aucs.append(fold_auc)

            return {
                'auc_mean': np.mean(fold_aucs),
                'auc_std': np.std(fold_aucs),
                'auc_folds': fold_aucs
            }
        else:
            # Single train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )

            # Compute concept vector on TRAIN set only
            train_conflict = X_train[y_train == 1]
            train_consistent = X_train[y_train == 0]
            
            concept_vector = train_conflict.mean(axis=0) - train_consistent.mean(axis=0)
            concept_vector = concept_vector / (np.linalg.norm(concept_vector) + 1e-8)

            # Evaluate on TEST set
            test_scores = X_test @ concept_vector
            auc = roc_auc_score(y_test, test_scores)

            return {
                'auc_mean': auc,
                'auc_std': 0.0,
                'auc_folds': [auc]
            }

    def compute_lexical_baseline_auc(
        self,
        conflict_texts: List[str],
        consistent_texts: List[str],
        test_size: float = DEFAULT_TEST_SIZE,
        random_state: int = DEFAULT_RANDOM_STATE
    ) -> Dict[str, float]:
        """
        Compute BOW/TF-IDF baseline AUC to check for lexical cues.
        
        If this baseline AUC is high, it suggests shallow lexical features
        may be dominating the concept vector analysis.

        Args:
            conflict_texts: list of conflict sample texts
            consistent_texts: list of consistent sample texts
            test_size: fraction for testing
            random_state: random seed

        Returns:
            Dict with 'bow_auc', 'tfidf_auc'
        """
        all_texts = conflict_texts + consistent_texts
        y = np.array([1] * len(conflict_texts) + [0] * len(consistent_texts))

        # TF-IDF vectorization
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        X = vectorizer.fit_transform(all_texts)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Train logistic regression
        clf = LogisticRegression(max_iter=1000, random_state=random_state)
        clf.fit(X_train, y_train)

        # Evaluate
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        tfidf_auc = roc_auc_score(y_test, y_pred_proba)

        return {
            'tfidf_auc': tfidf_auc,
            'warning': 'HIGH - lexical cues may dominate!' if tfidf_auc > 0.7 else 'OK'
        }

    def analyze_single_conflict_type(
        self,
        conflict_type: str,
        conflict_texts: List[str],
        consistent_texts: List[str],
        target_layers: List[int] = None,
        use_cv: bool = True,
        n_folds: int = DEFAULT_N_FOLDS,
        check_lexical_baseline: bool = True
    ) -> Dict[int, Dict[str, float]]:
        """
        Analyze single conflict type with proper train/test split.

        Args:
            conflict_type: conflict type name
            conflict_texts: conflict sample texts
            consistent_texts: consistent sample texts
            target_layers: layers to analyze
            use_cv: if True, use k-fold CV for robust AUC estimation
            n_folds: number of CV folds
            check_lexical_baseline: if True, compute BOW baseline to check for lexical cues

        Returns:
            layer_auc: {layer_idx: {'auc_mean': float, 'auc_std': float, ...}}
        """
        if target_layers is None:
            target_layers = list(range(self.num_layers))

        logger.info(f"\n{'='*60}")
        logger.info(f"Analyzing: {conflict_type}")
        logger.info(f"{'='*60}")
        logger.info(f"Samples: {len(conflict_texts)} conflict, {len(consistent_texts)} consistent")
        logger.info(f"Evaluation: {'5-fold CV' if use_cv else 'Single 80/20 split'}")

        # Check lexical baseline first
        if check_lexical_baseline:
            logger.info("\nChecking lexical baseline (TF-IDF)...")
            lexical_result = self.compute_lexical_baseline_auc(conflict_texts, consistent_texts)
            logger.info(f"  TF-IDF AUC: {lexical_result['tfidf_auc']:.4f} [{lexical_result['warning']}]")
            self.lexical_baselines = getattr(self, 'lexical_baselines', {})
            self.lexical_baselines[conflict_type] = lexical_result

        self.concept_vectors[conflict_type] = {}
        self.auc_scores[conflict_type] = {}
        self.auc_details = getattr(self, 'auc_details', {})
        self.auc_details[conflict_type] = {}

        for layer_idx in target_layers:
            logger.info(f"Layer {layer_idx}...")

            # extract hidden states
            conflict_hiddens = self.extract_hidden_states(conflict_texts, layer_idx)
            consistent_hiddens = self.extract_hidden_states(consistent_texts, layer_idx)

            # compute concept vector (on all data for storage, but AUC uses split)
            concept_vector = self.compute_concept_vector(conflict_hiddens, consistent_hiddens)
            self.concept_vectors[conflict_type][layer_idx] = concept_vector

            # compute AUC with proper train/test split
            auc_result = self.compute_awareness_auc_with_split(
                conflict_hiddens,
                consistent_hiddens,
                use_cv=use_cv,
                n_folds=n_folds
            )
            
            # Store mean AUC for backward compatibility
            self.auc_scores[conflict_type][layer_idx] = auc_result['auc_mean']
            # Store full details
            self.auc_details[conflict_type][layer_idx] = auc_result

            if use_cv:
                logger.info(f"  AUC: {auc_result['auc_mean']:.4f} ± {auc_result['auc_std']:.4f}")
            else:
                logger.info(f"  AUC: {auc_result['auc_mean']:.4f}")

        # find best layer
        best_layer = max(self.auc_scores[conflict_type], key=self.auc_scores[conflict_type].get)
        best_auc = self.auc_scores[conflict_type][best_layer]
        best_std = self.auc_details[conflict_type][best_layer].get('auc_std', 0)
        
        logger.info(f"\nBest Awareness Layer: {best_layer}")
        logger.info(f"  AUC = {best_auc:.4f} ± {best_std:.4f}")

        return self.auc_scores[conflict_type]

    def analyze_all_conflict_types(
        self,
        conflict_samples_dict: Dict[str, Tuple[List[str], List[str]]],
        target_layers: List[int] = None
    ) -> Dict[str, Dict[int, float]]:
        """
        analyze all conflict types

        Args:
            conflict_samples_dict: {conflict_type: (consistent_texts, conflict_texts)}
            target_layers: layers to analyze

        Returns:
            auc_scores: {conflict_type: {layer_idx: auc_score}}
        """
        for conflict_type, (consistent_texts, conflict_texts) in conflict_samples_dict.items():
            if not consistent_texts or not conflict_texts:
                logger.warning(f"Skipping {conflict_type}: no samples")
                continue

            self.analyze_single_conflict_type(
                conflict_type,
                conflict_texts,
                consistent_texts,
                target_layers
            )

        return self.auc_scores

    
    # cross-conflict comparison analysis
    

    def compute_concept_vector_similarity(self) -> Dict[str, Dict[str, float]]:
        """
        compute concept vector cosine similarity between conflict types (averaged over layers)

        Returns:
            similarity_matrix: {conflict_type_1: {conflict_type_2: cosine_similarity}}
        """
        conflict_types = list(self.concept_vectors.keys())
        similarity_matrix = {ct: {} for ct in conflict_types}

        logger.info(f"\n{'='*60}")
        logger.info("Computing Concept Vector Similarities")
        logger.info(f"{'='*60}")

        for ct1 in conflict_types:
            for ct2 in conflict_types:
                if ct1 == ct2:
                    similarity_matrix[ct1][ct2] = 1.0
                    continue

                # compute similarity per layer then average
                layer_similarities = []

                common_layers = set(self.concept_vectors[ct1].keys()) & set(self.concept_vectors[ct2].keys())

                for layer_idx in common_layers:
                    v1 = self.concept_vectors[ct1][layer_idx].float().cpu().numpy()
                    v2 = self.concept_vectors[ct2][layer_idx].float().cpu().numpy()

                    # cosine similarity = 1 - cosine distance
                    similarity = 1 - cosine(v1, v2)
                    layer_similarities.append(similarity)

                avg_similarity = np.mean(layer_similarities) if layer_similarities else 0.0
                similarity_matrix[ct1][ct2] = avg_similarity

                logger.info(f"  {ct1} <-> {ct2}: {avg_similarity:.4f}")

        return similarity_matrix

    def get_best_awareness_layers(self) -> Dict[str, int]:
        """
        get best awareness layer for each conflict type

        Returns:
            best_layers: {conflict_type: best_layer_idx}
        """
        best_layers = {}

        for conflict_type, layer_auc in self.auc_scores.items():
            if layer_auc:
                best_layer = max(layer_auc, key=layer_auc.get)
                best_layers[conflict_type] = best_layer

        return best_layers

    def get_auc_strength_summary(self) -> Dict[str, Dict[str, float]]:
        """
        get auc strength summary for each conflict type

        Returns:
            summary: {
                conflict_type: {
                    'max_auc': float,
                    'mean_auc': float,
                    'best_layer': int
                }
            }
        """
        summary = {}

        for conflict_type, layer_auc in self.auc_scores.items():
            if not layer_auc:
                continue

            auc_values = list(layer_auc.values())
            best_layer = max(layer_auc, key=layer_auc.get)

            summary[conflict_type] = {
                'max_auc': max(auc_values),
                'mean_auc': np.mean(auc_values),
                'best_layer': best_layer
            }

        return summary

    
    # visualization
    

    def plot_auc_comparison(self, save_path: str = None):
        """
        plot auc comparison across all conflict types

        Args:
            save_path: save path
        """
        rcParams['axes.labelsize'] = 14
        rcParams['xtick.labelsize'] = 12
        rcParams['ytick.labelsize'] = 12
        rcParams['legend.fontsize'] = 10

        plt.figure(figsize=(14, 8))

        colors = plt.cm.tab10(np.linspace(0, 1, len(self.auc_scores)))

        for i, (conflict_type, layer_auc) in enumerate(self.auc_scores.items()):
            if not layer_auc:
                continue

            layers = sorted(layer_auc.keys())
            aucs = [layer_auc[l] for l in layers]

            plt.plot(
                layers,
                aucs,
                linewidth=2,
                color=colors[i],
                label=conflict_type.replace('_', ' ').title()
            )

        plt.axhline(y=0.5, color='red', linestyle='--', label='Random Baseline', alpha=0.5, linewidth=1.5)

        plt.xlabel('Layer Index', fontweight='bold')
        plt.ylabel('AUC Score', fontweight='bold')
        plt.title('Cross-Conflict Concept Vector Awareness', fontweight='bold', fontsize=16)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.ylim([0.4, 1.0])

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"AUC comparison plot saved to {save_path}")

        plt.show()
        plt.close()

    def plot_similarity_heatmap(self, similarity_matrix: Dict[str, Dict[str, float]], save_path: str = None):
        """
        plot concept vector similarity heatmap

        Args:
            similarity_matrix: similarity matrix
            save_path: save path
        """
        conflict_types = list(similarity_matrix.keys())

        # convert to numpy matrix
        matrix = np.zeros((len(conflict_types), len(conflict_types)))
        for i, ct1 in enumerate(conflict_types):
            for j, ct2 in enumerate(conflict_types):
                matrix[i, j] = similarity_matrix[ct1][ct2]

        # shorten labels
        labels = [ct.replace('_conflict', '').replace('_', ' ').title() for ct in conflict_types]

        rcParams['axes.labelsize'] = 12
        rcParams['xtick.labelsize'] = 10
        rcParams['ytick.labelsize'] = 10

        plt.figure(figsize=(10, 8))

        sns.heatmap(
            matrix,
            cmap='RdYlGn',
            xticklabels=labels,
            yticklabels=labels,
            annot=True,
            fmt='.3f',
            cbar_kws={'label': 'Cosine Similarity'},
            vmin=-1,
            vmax=1,
            center=0
        )

        plt.title('Concept Vector Similarity Across Conflict Types', fontweight='bold', fontsize=14)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Similarity heatmap saved to {save_path}")

        plt.show()
        plt.close()

    def plot_best_layers_comparison(self, save_path: str = None):
        """
        plot the best awareness layer and the max AUC for each conflict type

        Args:
            save_path: save path
        """
        best_layers = self.get_best_awareness_layers()
        summary = self.get_auc_strength_summary()

        if not best_layers:
            logger.warning("No best layers to plot")
            return

        conflict_types = list(best_layers.keys())
        layers = [best_layers[ct] for ct in conflict_types]
        max_aucs = [summary[ct]['max_auc'] for ct in conflict_types]

        # shorten labels
        labels = [ct.replace('_conflict', '').replace('_', ' ').title() for ct in conflict_types]

        rcParams['axes.labelsize'] = 12
        rcParams['xtick.labelsize'] = 10
        rcParams['ytick.labelsize'] = 10

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # subplot 1: best awareness layer
        colors1 = plt.cm.viridis(np.linspace(0, 1, len(layers)))
        bars1 = ax1.barh(range(len(labels)), layers, color=colors1, edgecolor='black', linewidth=0.8)

        for i, (layer, label) in enumerate(zip(layers, labels)):
            ax1.text(layer + 0.5, i, f'L{layer}', va='center', fontsize=10, fontweight='bold')

        ax1.set_yticks(range(len(labels)))
        ax1.set_yticklabels(labels)
        ax1.set_xlabel('Layer Index', fontweight='bold')
        ax1.set_title('Best Awareness Layer by Conflict Type', fontweight='bold', fontsize=13)
        ax1.grid(axis='x', alpha=0.3)

        # subplot 2: max auc strength
        colors2 = plt.cm.plasma(np.linspace(0, 1, len(max_aucs)))
        bars2 = ax2.barh(range(len(labels)), max_aucs, color=colors2, edgecolor='black', linewidth=0.8)

        for i, (auc, label) in enumerate(zip(max_aucs, labels)):
            ax2.text(auc + 0.01, i, f'{auc:.3f}', va='center', fontsize=10, fontweight='bold')

        ax2.set_yticks(range(len(labels)))
        ax2.set_yticklabels(labels)
        ax2.set_xlabel('Max AUC Score', fontweight='bold')
        ax2.set_title('Awareness Strength by Conflict Type', fontweight='bold', fontsize=13)
        ax2.set_xlim([0.5, 1.0])
        ax2.grid(axis='x', alpha=0.3)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Best layers comparison saved to {save_path}")

        plt.show()
        plt.close()

    
    # save and load results
    

    def save_results(self, save_dir: str):
        """Save analysis results with full methodology details."""
        import json
        
        os.makedirs(save_dir, exist_ok=True)

        # save concept vectors
        vectors_dir = os.path.join(save_dir, "concept_vectors")
        os.makedirs(vectors_dir, exist_ok=True)

        for conflict_type, vectors in self.concept_vectors.items():
            for layer_idx, vector in vectors.items():
                save_path = os.path.join(
                    vectors_dir,
                    f"{conflict_type}_layer{layer_idx}.pt"
                )
                torch.save(vector.float().cpu(), save_path)

        # save auc scores (mean only, for backward compatibility)
        auc_path = os.path.join(save_dir, "auc_scores.json")
        with open(auc_path, 'w') as f:
            json.dump(self.auc_scores, f, indent=2)

        # save detailed AUC results with CV folds
        if hasattr(self, 'auc_details'):
            # Convert numpy arrays to lists for JSON serialization
            auc_details_serializable = {}
            for ct, layers in self.auc_details.items():
                auc_details_serializable[ct] = {}
                for layer, details in layers.items():
                    auc_details_serializable[ct][str(layer)] = {
                        'auc_mean': float(details['auc_mean']),
                        'auc_std': float(details['auc_std']),
                        'auc_folds': [float(x) for x in details.get('auc_folds', [])]
                    }
            
            auc_details_path = os.path.join(save_dir, "auc_details_cv.json")
            with open(auc_details_path, 'w') as f:
                json.dump(auc_details_serializable, f, indent=2)

        # save lexical baseline results
        if hasattr(self, 'lexical_baselines'):
            lexical_path = os.path.join(save_dir, "lexical_baseline.json")
            with open(lexical_path, 'w') as f:
                json.dump(self.lexical_baselines, f, indent=2)

        # save summary statistics
        summary = self.get_auc_strength_summary()
        summary_path = os.path.join(save_dir, "auc_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        # save methodology metadata
        methodology = {
            'evaluation_method': '5-fold stratified cross-validation',
            'concept_vector_training': 'Computed on training fold only',
            'auc_evaluation': 'Computed on held-out test fold',
            'n_folds': DEFAULT_N_FOLDS,
            'random_state': DEFAULT_RANDOM_STATE,
            'test_size_if_single_split': DEFAULT_TEST_SIZE,
            'lexical_baseline': 'TF-IDF + Logistic Regression with 80/20 split',
            'note': 'AUC values are mean ± std across CV folds to avoid optimistic estimates'
        }
        methodology_path = os.path.join(save_dir, "methodology.json")
        with open(methodology_path, 'w') as f:
            json.dump(methodology, f, indent=2)

        logger.info(f"Results saved to {save_dir}")
        logger.info(f"  - auc_scores.json: Mean AUC per layer")
        logger.info(f"  - auc_details_cv.json: Full CV results with std")
        logger.info(f"  - lexical_baseline.json: TF-IDF baseline AUCs")
        logger.info(f"  - methodology.json: Evaluation methodology details")
