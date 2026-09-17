#!/usr/bin/env python3
"""
Cross-Conflict SEA (Spectral Energy Analysis)

Implements the Spectral Energy Analysis of Section 4.2 of
"Large Language Models in Resolving Contextual Knowledge Conflicts"
(EMNLP 2026). For every target layer, the last non-padding-token hidden
states of the conflict prompts and of the consistent prompts form two
matrices H_conf and H_cons. Each matrix is centered and its energy ratio

    ER = sum_{i<=k} sigma_i^2 / ||H_centered||_F^2        (k = 10 by default)

is computed with torch.svd_lowrank. The conflict-specific signal is
Delta-ER = ER_conf - ER_cons: a positive value means the conflict prompts
are compressed into a lower-rank subspace than the consistent prompts
(low-rank compression), a negative value means they are dispersed.
Bootstrap confidence intervals quantify the stability of each ER.

Implementation notes:
- Last-token extraction uses the attention mask, so PAD tokens are never read.
- All target layers are captured in a single forward pass with forward hooks.
- torch.svd_lowrank returns only the top-k singular values, which avoids a
  full SVD and a CPU transfer.

This module is a library. The runners are
analysis/spectral_energy/run_sea_cross_conflict.py (Figures delta_er and
delta_er_comparison) and analysis/spectral_energy/run_sea_subfolder.py
(per-subfolder energy comparison). Example:

    python analysis/spectral_energy/run_sea_cross_conflict.py \
        --model_path meta-llama/Llama-3.1-8B-Instruct --sample_limit 400 \
        --top_k 10 --compute_ci --n_bootstrap 500 --seed 42
"""

import os
import torch
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams

logging.basicConfig(
    format="%(asctime)s - %(levelname)s %(name)s %(lineno)s: %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class CrossConflictSEAAnalyzer:
    """
    Cross-conflict Spectral Energy Analysis (SEA) analyzer:
    - Proper last non-PAD token extraction
    - Single forward pass multi-layer activation collection
    - PyTorch-based low-rank SVD
    - Conflict vs consistent comparison with Delta-ER (ΔER = ER_conflict - ER_consistent)
    """

    def __init__(self, model, tokenizer, model_name: str, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.device = device
        self.num_layers = len(model.model.layers)

        # Store spectrum results per conflict type
        # {conflict_type: {layer_idx: {'conflict_er': ..., 'consistent_er': ..., 'delta_er': ...}}}
        self.spectrum_results = {}

    @torch.no_grad()
    def extract_activations_all_layers(
        self,
        texts: List[str],
        target_layers: List[int],
        batch_size: int = 1
    ) -> Dict[int, torch.Tensor]:
        """
        Extract activations from all target layers in a SINGLE forward pass.
        Uses attention_mask to get the last non-PAD token for each sample.

        Args:
            texts: input text list
            target_layers: list of layer indices to extract
            batch_size: batch size

        Returns:
            activations: {layer_idx: tensor[num_samples, hidden_dim]}
        """
        # Initialize storage for each layer
        layer_activations = {layer_idx: [] for layer_idx in target_layers}

        # Captured activations storage (shared across hooks)
        captured = {}

        # Create hooks for all target layers
        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    captured[layer_idx] = output[0].detach()
                else:
                    captured[layer_idx] = output.detach()
            return hook_fn

        # Register all hooks at once
        handles = []
        for layer_idx in target_layers:
            target_layer = self.model.model.layers[layer_idx]
            handle = target_layer.register_forward_hook(make_hook(layer_idx))
            handles.append(handle)

        try:
            for i in tqdm(range(0, len(texts), batch_size), desc="Extracting activations", leave=False):
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

                # Clear captured activations
                captured.clear()

                # Single forward pass - all hooks fire
                _ = self.model(**inputs)

                # Get last non-PAD token index for each sample using attention_mask
                attention_mask = inputs['attention_mask']
                # last_idx[i] = index of last non-pad token for sample i
                last_idx = attention_mask.sum(dim=1) - 1  # [batch_size]

                # Extract last non-PAD token activation for each layer
                for layer_idx in target_layers:
                    act = captured[layer_idx]  # [batch_size, seq_len, hidden_dim]
                    batch_size_actual = act.shape[0]

                    # Gather last non-PAD token for each sample
                    # act[i, last_idx[i], :] for each i
                    # NOTE: In multi-GPU setup (device_map="auto"), act may be on different GPU
                    # Ensure indices are on the same device as act
                    act_device = act.device
                    last_token_acts = act[
                        torch.arange(batch_size_actual, device=act_device),
                        last_idx.to(act_device),
                        :
                    ]  # [batch_size, hidden_dim]

                    layer_activations[layer_idx].append(last_token_acts.cpu())

        finally:
            # Remove all hooks
            for handle in handles:
                handle.remove()

        # Concatenate all batches for each layer
        result = {}
        for layer_idx in target_layers:
            if layer_activations[layer_idx]:
                result[layer_idx] = torch.cat(layer_activations[layer_idx], dim=0)
            else:
                result[layer_idx] = torch.empty(0)

        return result

    def compute_energy_ratio_torch(
        self,
        activations: torch.Tensor,
        top_k: int = 10
    ) -> Tuple[float, np.ndarray]:
        """
        Compute energy ratio using PyTorch low-rank SVD.
        Avoids CPU transfer and full SVD computation.

        Args:
            activations: [num_samples, hidden_dim]
            top_k: number of top components for energy ratio

        Returns:
            energy_ratio: float in [0, 1]
            singular_values: top-k singular values (numpy array)
        """
        if activations.numel() == 0 or activations.shape[0] < 2:
            return 0.0, np.array([])

        # Center the activations
        H = activations.float()
        H_centered = H - H.mean(dim=0, keepdim=True)

        # Total energy using Frobenius norm (no full SVD needed)
        total_energy = torch.sum(H_centered ** 2).item()

        if total_energy == 0:
            return 0.0, np.array([])

        # Low-rank SVD for top-k singular values only
        # Use min of dimensions to ensure valid k
        k = min(top_k, H_centered.shape[0] - 1, H_centered.shape[1])
        if k < 1:
            return 0.0, np.array([])

        try:
            # PyTorch low-rank SVD (much faster than full SVD)
            U, S, V = torch.svd_lowrank(H_centered, q=k)
            singular_values = S.cpu().numpy()

            # Top-k energy
            top_k_energy = np.sum(singular_values ** 2)
            energy_ratio = top_k_energy / total_energy

            return float(energy_ratio), singular_values

        except Exception as e:
            logger.warning(f"SVD failed: {e}, falling back to numpy")
            # Fallback to numpy if torch SVD fails
            H_np = H_centered.cpu().numpy()
            _, S_full, _ = np.linalg.svd(H_np, full_matrices=False)
            singular_values = S_full[:top_k]
            top_k_energy = np.sum(singular_values ** 2)
            energy_ratio = top_k_energy / total_energy
            return float(energy_ratio), singular_values

    def compute_bootstrap_ci(
        self,
        activations: torch.Tensor,
        top_k: int = 10,
        n_bootstrap: int = 100,
        ci_level: float = 0.95
    ) -> Tuple[float, float, float]:
        """
        Compute bootstrap confidence interval for energy ratio.

        Args:
            activations: [num_samples, hidden_dim]
            top_k: number of top components
            n_bootstrap: number of bootstrap samples
            ci_level: confidence interval level (e.g., 0.95 for 95% CI)

        Returns:
            mean_er: mean energy ratio
            ci_lower: lower bound of CI
            ci_upper: upper bound of CI
        """
        n_samples = activations.shape[0]
        if n_samples < 10:
            er, _ = self.compute_energy_ratio_torch(activations, top_k)
            return er, er, er

        bootstrap_ers = []

        for _ in range(n_bootstrap):
            # Resample with replacement
            indices = torch.randint(0, n_samples, (n_samples,))
            resampled = activations[indices]

            er, _ = self.compute_energy_ratio_torch(resampled, top_k)
            bootstrap_ers.append(er)

        bootstrap_ers = np.array(bootstrap_ers)
        mean_er = np.mean(bootstrap_ers)

        alpha = 1 - ci_level
        ci_lower = np.percentile(bootstrap_ers, alpha / 2 * 100)
        ci_upper = np.percentile(bootstrap_ers, (1 - alpha / 2) * 100)

        return float(mean_er), float(ci_lower), float(ci_upper)

    def analyze_single_conflict_type(
        self,
        conflict_type: str,
        conflict_texts: List[str],
        consistent_texts: List[str],
        target_layers: List[int] = None,
        top_k: int = 10,
        compute_ci: bool = True,
        n_bootstrap: int = 100
    ) -> Dict[int, Dict]:
        """
        Analyze spectral structure of single conflict type with optimizations.

        Args:
            conflict_type: conflict type name
            conflict_texts: conflict sample texts
            consistent_texts: consistent sample texts
            target_layers: layers to analyze (default: all)
            top_k: top-k components for energy ratio
            compute_ci: whether to compute bootstrap CI
            n_bootstrap: number of bootstrap samples

        Returns:
            layer_spectrum: {layer_idx: {
                'conflict_er', 'consistent_er', 'delta_er',
                'conflict_ci', 'consistent_ci', 'singular_values'
            }}
        """
        if target_layers is None:
            target_layers = list(range(self.num_layers))

        logger.info(f"\n{'='*60}")
        logger.info(f"SEA Analysis: {conflict_type}")
        logger.info(f"  Conflict samples: {len(conflict_texts)}")
        logger.info(f"  Consistent samples: {len(consistent_texts)}")
        logger.info(f"  Target layers: {len(target_layers)}")
        logger.info(f"{'='*60}")

        self.spectrum_results[conflict_type] = {}

        # Extract all layer activations in ONE forward pass
        logger.info("Extracting conflict activations (single forward pass)...")
        conflict_acts_all = self.extract_activations_all_layers(
            conflict_texts, target_layers, batch_size=4
        )

        logger.info("Extracting consistent activations (single forward pass)...")
        consistent_acts_all = self.extract_activations_all_layers(
            consistent_texts, target_layers, batch_size=4
        )

        # Analyze each layer
        for layer_idx in tqdm(target_layers, desc="Computing spectral metrics"):
            conflict_acts = conflict_acts_all[layer_idx]
            consistent_acts = consistent_acts_all[layer_idx]

            # Compute energy ratios
            conflict_er, conflict_sv = self.compute_energy_ratio_torch(conflict_acts, top_k)
            consistent_er, consistent_sv = self.compute_energy_ratio_torch(consistent_acts, top_k)

            # Delta ER: conflict-specific signal
            delta_er = conflict_er - consistent_er

            result = {
                'conflict_er': conflict_er,
                'consistent_er': consistent_er,
                'delta_er': delta_er,
                'conflict_singular_values': conflict_sv,
                'consistent_singular_values': consistent_sv,
                'top_k': top_k
            }

            # Optional: bootstrap CI
            if compute_ci:
                conflict_mean, conflict_ci_low, conflict_ci_high = self.compute_bootstrap_ci(
                    conflict_acts, top_k, n_bootstrap
                )
                consistent_mean, consistent_ci_low, consistent_ci_high = self.compute_bootstrap_ci(
                    consistent_acts, top_k, n_bootstrap
                )

                result['conflict_ci'] = (conflict_ci_low, conflict_ci_high)
                result['consistent_ci'] = (consistent_ci_low, consistent_ci_high)
                result['conflict_er_bootstrap'] = conflict_mean
                result['consistent_er_bootstrap'] = consistent_mean

            self.spectrum_results[conflict_type][layer_idx] = result

        # Summary
        delta_ers = [self.spectrum_results[conflict_type][l]['delta_er'] for l in target_layers]
        best_layer = target_layers[np.argmax(delta_ers)]
        best_delta = max(delta_ers)

        logger.info(f"\nHighest ΔER Layer: {best_layer} (ΔER = {best_delta:.4f})")

        return self.spectrum_results[conflict_type]

    def analyze_all_conflict_types(
        self,
        conflict_samples_dict: Dict[str, Tuple[List[str], List[str]]],
        target_layers: List[int] = None,
        top_k: int = 10,
        compute_ci: bool = True,
        n_bootstrap: int = 100
    ) -> Dict[str, Dict[int, Dict]]:
        """
        Analyze spectral structures of all conflict types.

        Args:
            conflict_samples_dict: {conflict_type: (consistent_texts, conflict_texts)}
            target_layers: layers to analyze
            top_k: top-k components for energy ratio
            compute_ci: whether to compute bootstrap CI
            n_bootstrap: number of bootstrap samples

        Returns:
            all_spectrum_results: {conflict_type: {layer_idx: {...}}}
        """
        for conflict_type, (consistent_texts, conflict_texts) in conflict_samples_dict.items():
            if not consistent_texts or not conflict_texts:
                logger.warning(f"Skipping {conflict_type}: no samples")
                continue

            self.analyze_single_conflict_type(
                conflict_type,
                conflict_texts,
                consistent_texts,
                target_layers,
                top_k,
                compute_ci,
                n_bootstrap
            )

        return self.spectrum_results

    # ========================================
    # Cross-conflict comparison analysis
    # ========================================

    def get_highest_delta_er_layers(self) -> Dict[str, int]:
        """
        Get layer with highest ΔER for each conflict type.

        Returns:
            highest_layers: {conflict_type: layer_idx}
        """
        highest_layers = {}

        for conflict_type, layer_spectrum in self.spectrum_results.items():
            if not layer_spectrum:
                continue

            delta_by_layer = {l: layer_spectrum[l]['delta_er'] for l in layer_spectrum}
            highest_layer = max(delta_by_layer, key=delta_by_layer.get)
            highest_layers[conflict_type] = highest_layer

        return highest_layers

    def get_energy_strength_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get comprehensive energy strength summary for each conflict type.

        Returns:
            summary: {conflict_type: {
                'max_delta_er', 'mean_delta_er', 'highest_layer',
                'max_conflict_er', 'max_consistent_er'
            }}
        """
        summary = {}

        for conflict_type, layer_spectrum in self.spectrum_results.items():
            if not layer_spectrum:
                continue

            delta_ers = [layer_spectrum[l]['delta_er'] for l in layer_spectrum]
            conflict_ers = [layer_spectrum[l]['conflict_er'] for l in layer_spectrum]
            consistent_ers = [layer_spectrum[l]['consistent_er'] for l in layer_spectrum]

            highest_layer = self.get_highest_delta_er_layers().get(conflict_type, -1)

            summary[conflict_type] = {
                'max_delta_er': float(max(delta_ers)),
                'mean_delta_er': float(np.mean(delta_ers)),
                'std_delta_er': float(np.std(delta_ers)),
                'highest_layer': int(highest_layer),
                'max_conflict_er': float(max(conflict_ers)),
                'mean_conflict_er': float(np.mean(conflict_ers)),
                'max_consistent_er': float(max(consistent_ers)),
                'mean_consistent_er': float(np.mean(consistent_ers))
            }

        return summary

    # ========================================
    # Visualization
    # ========================================

    def plot_delta_er_comparison(self, save_path: str = None, show_ci: bool = True):
        """
        Plot ΔER (Delta Energy Ratio) comparison across all conflict types.
        This is the key metric showing conflict-specific spectral changes.

        Args:
            save_path: save path
            show_ci: whether to show confidence intervals
        """
        rcParams['axes.labelsize'] = 14
        rcParams['xtick.labelsize'] = 12
        rcParams['ytick.labelsize'] = 12
        rcParams['legend.fontsize'] = 10

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        colors = plt.cm.tab10(np.linspace(0, 1, len(self.spectrum_results)))

        # Plot 1: Delta ER (conflict-specific signal)
        ax1 = axes[0]
        for i, (conflict_type, layer_spectrum) in enumerate(self.spectrum_results.items()):
            if not layer_spectrum:
                continue

            layers = sorted(layer_spectrum.keys())
            delta_ers = [layer_spectrum[l]['delta_er'] for l in layers]

            ax1.plot(
                layers,
                delta_ers,
                linewidth=2,
                color=colors[i],
                label=conflict_type.replace('_', ' ').title()
            )

        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Layer Index', fontweight='bold')
        ax1.set_ylabel('ΔER (Conflict - Consistent)', fontweight='bold')
        ax1.set_title('Delta Energy Ratio by Layer', fontweight='bold', fontsize=14)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Conflict vs Consistent ER
        ax2 = axes[1]
        for i, (conflict_type, layer_spectrum) in enumerate(self.spectrum_results.items()):
            if not layer_spectrum:
                continue

            layers = sorted(layer_spectrum.keys())
            conflict_ers = [layer_spectrum[l]['conflict_er'] for l in layers]
            consistent_ers = [layer_spectrum[l]['consistent_er'] for l in layers]

            ax2.plot(
                layers,
                conflict_ers,
                linewidth=2,
                color=colors[i],
                linestyle='-',
                label=f"{conflict_type.replace('_', ' ').title()} (Conflict)"
            )
            ax2.plot(
                layers,
                consistent_ers,
                linewidth=2,
                color=colors[i],
                linestyle='--',
                alpha=0.6
            )

        ax2.set_ylim([0, 1.0])
        ax2.set_xlabel('Layer Index', fontweight='bold')
        ax2.set_ylabel('Energy Ratio', fontweight='bold')
        ax2.set_title('Conflict vs Consistent Energy Ratio', fontweight='bold', fontsize=14)
        ax2.legend(loc='best', fontsize=8)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Delta ER comparison plot saved to {save_path}")

        plt.show()
        plt.close()

    def plot_energy_comparison(self, save_path: str = None):
        """
        Plot the conflict energy ratio per layer for all conflict types (kept for backwards compatibility).

        Args:
            save_path: save path
        """
        rcParams['axes.labelsize'] = 14
        rcParams['xtick.labelsize'] = 12
        rcParams['ytick.labelsize'] = 12
        rcParams['legend.fontsize'] = 10

        plt.figure(figsize=(14, 8))

        colors = plt.cm.tab10(np.linspace(0, 1, len(self.spectrum_results)))

        for i, (conflict_type, layer_spectrum) in enumerate(self.spectrum_results.items()):
            if not layer_spectrum:
                continue

            layers = sorted(layer_spectrum.keys())
            # Plot the conflict ER only (kept for backwards compatibility)
            values = [layer_spectrum[l]['conflict_er'] for l in layers]

            plt.plot(
                layers,
                values,
                linewidth=2,
                color=colors[i],
                label=conflict_type.replace('_', ' ').title()
            )

        plt.ylim([0, 1.0])
        plt.xlabel('Layer Index', fontweight='bold')
        plt.ylabel('Energy Ratio (Top-k / Total)', fontweight='bold')
        plt.title('Cross-Conflict Spectral Energy Distribution', fontweight='bold', fontsize=16)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Energy comparison plot saved to {save_path}")

        plt.show()
        plt.close()

    def plot_summary_bar_chart(self, save_path: str = None):
        """
        Plot summary bar chart of max ΔER for each conflict type.

        Args:
            save_path: save path
        """
        summary = self.get_energy_strength_summary()

        conflict_types = list(summary.keys())
        max_delta_ers = [summary[ct]['max_delta_er'] for ct in conflict_types]
        highest_layers = [summary[ct]['highest_layer'] for ct in conflict_types]

        fig, ax = plt.subplots(figsize=(12, 6))

        colors = plt.cm.tab10(np.linspace(0, 1, len(conflict_types)))
        bars = ax.bar(
            range(len(conflict_types)),
            max_delta_ers,
            color=colors,
            edgecolor='black',
            linewidth=1
        )

        # Add layer labels on bars
        for bar, layer in zip(bars, highest_layers):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height + 0.01,
                f'L{layer}',
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold'
            )

        ax.set_xticks(range(len(conflict_types)))
        ax.set_xticklabels(
            [ct.replace('_', '\n') for ct in conflict_types],
            fontsize=10
        )
        ax.set_ylabel('Max ΔER', fontweight='bold')
        ax.set_title('Maximum Delta Energy Ratio by Conflict Type', fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Summary bar chart saved to {save_path}")

        plt.show()
        plt.close()

    # ========================================
    # Save results
    # ========================================

    def save_results(self, save_dir: str):
        """Save per-layer spectra (.npz per layer) and the energy summary (energy_summary.json)."""
        os.makedirs(save_dir, exist_ok=True)

        # Save spectrum results
        for conflict_type, layer_spectrum in self.spectrum_results.items():
            conflict_dir = os.path.join(save_dir, conflict_type)
            os.makedirs(conflict_dir, exist_ok=True)

            for layer_idx, spectrum in layer_spectrum.items():
                save_path = os.path.join(conflict_dir, f"spectrum_layer{layer_idx}.npz")

                save_dict = {
                    'conflict_er': spectrum['conflict_er'],
                    'consistent_er': spectrum['consistent_er'],
                    'delta_er': spectrum['delta_er'],
                    'conflict_singular_values': spectrum['conflict_singular_values'],
                    'consistent_singular_values': spectrum['consistent_singular_values'],
                    'top_k': spectrum['top_k']
                }

                # Add CI if available
                if 'conflict_ci' in spectrum:
                    save_dict['conflict_ci_low'] = spectrum['conflict_ci'][0]
                    save_dict['conflict_ci_high'] = spectrum['conflict_ci'][1]
                    save_dict['consistent_ci_low'] = spectrum['consistent_ci'][0]
                    save_dict['consistent_ci_high'] = spectrum['consistent_ci'][1]

                np.savez(save_path, **save_dict)

        # Save summary statistics
        import json
        summary = self.get_energy_strength_summary()
        summary_path = os.path.join(save_dir, "energy_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"SEA results saved to {save_dir}")
