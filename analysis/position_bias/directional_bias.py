"""
Directional position-bias measurement in internal representations.

Implements the measurement of Section 4.3 ("Evidence Position Bias in
Internal Representations") and the appendix "Bias Measurement Implementation
Notes" of "Large Language Models in Resolving Contextual Knowledge Conflicts"
(EMNLP 2026).

For a sample with K evidence pieces and a layer l, let c be the final-token
residual-stream activation of the combined-evidence prompt and a_i the
activation of the single-evidence prompt that contains only evidence i
(both collected by analysis/position_bias/evidence_collector.py). With the
neutral center mu = mean_i a_i, the directional bias score of evidence i is

    b_i = cos(c - mu, a_i - mu) = <c - mu, a_i - mu> / (|c - mu| |a_i - mu|)

with a small epsilon guard (norms below 1e-10 give b_i = 0). The evidence with
the largest b_i is the one the combined representation leans toward. Per layer,
the fraction of samples whose argmax_i b_i equals evidence i is the "bias
ratio" plotted as a stacked area over layers (one panel per conflict type:
ambiguity, granularity, perspective; figures
bias_simple_prompt_all_conflicts.png for Llama-3.1-8B with the simple prompt
and bias_stacked_area_all_conflicts_gpt20b.png for GPT-OSS-20B in the paper).

The code is extracted unchanged from the visualization module of the
activation-steering method; only the functions needed for the measurement are
kept. The distance-based variant (use_directional=False) is a legacy fallback
that was not used in the paper.

Functions:
    group_samples_by_conflict_type        group EvidenceActivations by task_type
    compute_directional_bias_attribution  b_i for one sample and layer
    compute_bias_metrics_per_layer        counts, ratios, mean b_i, balance score per layer
    compute_layer_evidence_bias_matrix    conflict_type -> layer -> evidence -> bias ratio
    visualize_bias_stacked_area_all_conflicts   three-panel stacked-area figure
    visualize_bias_stacked_area_combined        single panel averaged over conflict types

Example (after collecting activations with run_directional_bias.py):
    import pickle
    from analysis.position_bias.directional_bias import (
        compute_bias_metrics_per_layer, visualize_bias_stacked_area_all_conflicts)
    samples = pickle.load(open("results/analysis/position_bias/llama_8b/"
                               "evidence_activations_simple.pkl", "rb"))
    layers = sorted(samples[0].combined_evidence_acts.keys())
    visualize_bias_stacked_area_all_conflicts(samples, layers, "bias.png",
                                              title_suffix=" (Simple Prompt)")
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any
from collections import defaultdict


# Morandi color palette
MORANDI_COLORS = [
    '#9E8AAD', 
    '#B8847C', 
    '#7FA37A',  
    '#8BA3B8',  
    '#C9A07A', 
    '#7A9E9E',  
    '#B8A08A',  
    '#A89888',  
    '#8A9E86',  
    '#B8909E',  
]

# Color palette for different evidence sources (using Morandi)
SOURCE_COLORS = MORANDI_COLORS

# Display names for conflict types (task_type values of EvidenceActivations)
CONFLICT_DISPLAY_NAMES = {
    'ambiguity': 'Ambiguity Conflict',
    'granularity': 'Granularity Conflict',
    'perspective': 'Perspective Conflict',
}


# =============================================================================
# Helper Functions
# =============================================================================

def group_samples_by_conflict_type(samples: List[Any]) -> Dict[str, List[Any]]:
    """Group samples by their conflict type (task_type field)."""
    grouped = defaultdict(list)
    for sample in samples:
        conflict_type = sample.task_type
        grouped[conflict_type].append(sample)
    return dict(grouped)


def compute_directional_bias_attribution(
    combined_act: np.ndarray,
    single_acts: Dict[int, np.ndarray],
    centroid: np.ndarray,
    normalize_directions: bool = True
) -> Dict[int, float]:
    """
    Compute the directional bias score b_i of every evidence for one sample and layer.

    Compared with a distance-based nearest-neighbour rule this is more stable:
    1. it measures a directional projection, not an absolute distance,
    2. it is not affected by cluster density or by the evidence distribution,
    3. it has a clear geometric interpretation.

    Algorithm:
    1. Evidence characteristic direction (fingerprint): v_i = a_i - mu
    2. Combined activation offset: d = c - mu
    3. (Optional) normalize v_i and d to unit vectors
    4. Bias: b_i = d . v_i if normalized (cosine similarity, the paper's b_i),
       or b_i = (d . v_i) / |v_i|^2 if not normalized

    Normalization is applied AFTER computing v_i and d, not to the raw
    activations, so the geometry around the neutral center is preserved.
    Norms below 1e-10 (the epsilon guard) give a score of 0.

    Args:
        combined_act: Combined-evidence activation c, shape [hidden_dim]
        single_acts: Dict[source_idx, activation a_i] for the single-evidence prompts
        centroid: mu = mean of all single-evidence activations (neutral center)
        normalize_directions: Whether to normalize the direction vectors (paper: True)

    Returns:
        Dict[source_idx, bias_score]; a higher score means the combined
        representation leans more toward that evidence.
    """
    # Combined activation offset from neutral center
    d = combined_act - centroid
    
    # Normalize d AFTER subtraction
    if normalize_directions:
        d_norm = np.linalg.norm(d)
        if d_norm < 1e-10:
            return {idx: 0.0 for idx in single_acts}
        d = d / d_norm
    
    bias_scores = {}
    
    for source_idx, single_act in single_acts.items():
        # Evidence characteristic direction (fingerprint)
        v_i = single_act - centroid
        
        if normalize_directions:
            # Normalize v_i AFTER subtraction
            v_norm = np.linalg.norm(v_i)
            if v_norm < 1e-10:
                bias_scores[source_idx] = 0.0
                continue
            v_i = v_i / v_norm
            # With both normalized: bias_i = d . v_i (cosine similarity of directions)
            bias_i = np.dot(d, v_i)
        else:
            # Without normalization: bias_i = (d . v_i) / |v_i|^2
            v_norm_sq = np.dot(v_i, v_i)
            if v_norm_sq < 1e-10:
                bias_scores[source_idx] = 0.0
                continue
            bias_i = np.dot(d, v_i) / v_norm_sq
        
        bias_scores[source_idx] = float(bias_i)
    
    return bias_scores


def compute_bias_metrics_per_layer(
    samples: List[Any],
    target_layers: List[int],
    use_directional: bool = True,
    normalize_directions: bool = True
) -> Dict[int, Dict[str, Any]]:
    """
    Compute per-layer bias metrics with directional bias attribution.

    For every layer and sample, b_i = cos(c - mu, a_i - mu) is computed with
    compute_directional_bias_attribution and the evidence with the largest b_i
    is counted as the favored one.

    Metrics per layer:
    - bias_counts: number of samples favoring each evidence (argmax_i b_i)
    - bias_ratio: fraction of samples favoring each evidence
    - mean_bias_scores: mean b_i of each evidence
    - balance_score: normalized entropy of bias_ratio
      (1 = perfectly balanced, 0 = all samples favor one evidence)
    - total_samples, n_sources, method

    Samples with fewer than two single-evidence activations at a layer are
    skipped. Normalization happens AFTER computing the direction vectors
    (v_i = a_i - mu), not on the raw activations.

    Args:
        samples: List of EvidenceActivations
        target_layers: Layers to evaluate
        use_directional: True for the paper's directional measure; False uses the
            legacy distance-based nearest-neighbour rule
        normalize_directions: Normalize direction vectors (paper: True)

    Returns:
        Dict[layer_idx, metrics dict]; layers without valid samples are omitted.
    """
    layer_metrics = {}
    
    for layer_idx in target_layers:
        bias_counts = defaultdict(int)
        bias_scores_all = defaultdict(list)
        total_samples = 0
        
        for sample in samples:
            if layer_idx not in sample.combined_evidence_acts:
                continue
            
            combined = sample.combined_evidence_acts[layer_idx].flatten()
            
            # Collect single-evidence activations (RAW, no normalization here)
            single_acts = {}
            for source_idx in sample.single_evidence_acts:
                if layer_idx in sample.single_evidence_acts[source_idx]:
                    single_acts[source_idx] = sample.single_evidence_acts[source_idx][layer_idx].flatten()
            
            if len(single_acts) < 2:
                continue
            
            total_samples += 1
            
            if use_directional:
                # === Directional Bias Attribution (recommended) ===
                # Compute centroid from RAW activations
                all_single = np.array(list(single_acts.values()))
                centroid = np.mean(all_single, axis=0)
                
                # Get directional bias scores
                # Normalization happens INSIDE this function on direction vectors
                scores = compute_directional_bias_attribution(
                    combined, single_acts, centroid, 
                    normalize_directions=normalize_directions
                )
                
                # Record scores
                for source_idx, score in scores.items():
                    bias_scores_all[source_idx].append(score)
                
                # Find favored evidence (highest directional projection)
                if scores:
                    favored_idx = max(scores, key=scores.get)
                    bias_counts[favored_idx] += 1
            else:
                # === Distance-based (legacy fallback) ===
                min_dist = float('inf')
                closest_idx = -1
                for source_idx, single in single_acts.items():
                    dist = np.linalg.norm(combined - single)
                    bias_scores_all[source_idx].append(-dist)  # Negative for consistency
                    if dist < min_dist:
                        min_dist = dist
                        closest_idx = source_idx
                if closest_idx >= 0:
                    bias_counts[closest_idx] += 1
        
        if total_samples == 0:
            continue
        
        # Compute metrics
        n_sources = len(bias_scores_all)
        bias_ratio = {k: v / total_samples for k, v in bias_counts.items()}
        mean_bias_scores = {k: np.mean(v) for k, v in bias_scores_all.items()}
        
        # Balance score: entropy-based measure (1 = perfectly balanced)
        if n_sources > 1:
            probs = np.array([bias_counts.get(i, 0) / total_samples for i in range(n_sources)])
            probs = probs[probs > 0]  # Remove zeros for entropy calc
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            max_entropy = np.log(n_sources)
            balance_score = entropy / max_entropy if max_entropy > 0 else 0
        else:
            balance_score = 1.0
        
        layer_metrics[layer_idx] = {
            'bias_counts': dict(bias_counts),
            'bias_ratio': bias_ratio,
            'mean_bias_scores': mean_bias_scores,
            'balance_score': balance_score,
            'total_samples': total_samples,
            'n_sources': n_sources,
            'method': 'directional' if use_directional else 'distance'
        }
    
    return layer_metrics


# =============================================================================
# Layer x Evidence Bias Ratios
# =============================================================================

def compute_layer_evidence_bias_matrix(
    samples: List[Any],
    target_layers: List[int],
    use_directional: bool = True,
    normalize_directions: bool = True
) -> Dict[str, Dict[int, Dict[int, float]]]:
    """
    Compute the bias ratio of every evidence at every layer, per conflict type.

    For each sample and layer the combined activation is projected onto each
    evidence's characteristic direction, b_i = cos(c - mu, a_i - mu), and the
    evidence with the highest b_i is counted. The bias ratio of evidence i at a
    layer is the fraction of samples (of that conflict type) that favor i;
    this is the quantity plotted in the stacked-area figures.

    The directional measure is preferred over a distance-based nearest
    neighbour because it is not affected by cluster density, has a clear
    geometric interpretation, and matches the steering direction computation.
    Normalization happens AFTER computing the direction vectors
    (v_i = a_i - mu), not on the raw activations.

    Returns:
        Dict mapping conflict_type -> layer -> evidence_id -> bias_ratio
        (layers with no valid sample map to an empty dict)
    """
    samples_by_type = group_samples_by_conflict_type(samples)
    
    results = {}
    
    for conflict_type, ct_samples in samples_by_type.items():
        layer_evidence_counts = defaultdict(lambda: defaultdict(int))
        layer_totals = defaultdict(int)
        
        for sample in ct_samples:
            for layer_idx in target_layers:
                if layer_idx not in sample.combined_evidence_acts:
                    continue
                
                combined = sample.combined_evidence_acts[layer_idx].flatten()
                
                # Collect single-evidence activations (RAW, no normalization here)
                single_acts = {}
                for source_idx in sample.single_evidence_acts:
                    if layer_idx in sample.single_evidence_acts[source_idx]:
                        single_acts[source_idx] = sample.single_evidence_acts[source_idx][layer_idx].flatten()
                
                if len(single_acts) < 2:
                    continue
                
                if use_directional:
                    # === Directional Bias Attribution ===
                    # Compute centroid from RAW activations
                    all_single = np.array(list(single_acts.values()))
                    centroid = np.mean(all_single, axis=0)
                    
                    # Get directional bias scores
                    # Normalization happens INSIDE on direction vectors
                    scores = compute_directional_bias_attribution(
                        combined, single_acts, centroid,
                        normalize_directions=normalize_directions
                    )
                    
                    # Find favored evidence (highest directional projection)
                    if scores:
                        favored_idx = max(scores, key=scores.get)
                        layer_evidence_counts[layer_idx][favored_idx] += 1
                        layer_totals[layer_idx] += 1
                else:
                    # === Distance-based (legacy fallback) ===
                    min_dist = float('inf')
                    closest_idx = -1
                    
                    for source_idx, single in single_acts.items():
                        dist = np.linalg.norm(combined - single)
                        if dist < min_dist:
                            min_dist = dist
                            closest_idx = source_idx
                    
                    if closest_idx >= 0:
                        layer_evidence_counts[layer_idx][closest_idx] += 1
                        layer_totals[layer_idx] += 1
        
        # Convert to ratios
        results[conflict_type] = {}
        for layer_idx in target_layers:
            if layer_totals[layer_idx] > 0:
                results[conflict_type][layer_idx] = {
                    evi_id: count / layer_totals[layer_idx]
                    for evi_id, count in layer_evidence_counts[layer_idx].items()
                }
            else:
                results[conflict_type][layer_idx] = {}
    
    return results


# =============================================================================
# Stacked-Area Figures
# =============================================================================

def visualize_bias_stacked_area_all_conflicts(
    samples: List[Any],
    target_layers: List[int],
    output_path: str,
    steering_layers: Optional[List[int]] = None,
    title_suffix: str = ""
):
    """
    Stacked-area chart of the evidence bias ratio per layer, one panel per
    conflict type (ambiguity, granularity, perspective).

    X-axis: layers
    Y-axis: bias ratio, i.e. the fraction of samples whose argmax_i b_i is
            evidence i (stacked to 1)
    Colors: evidence sources (Evidence 0 is the first evidence in the prompt)

    This produces the figures bias_simple_prompt_all_conflicts.png and
    bias_stacked_area_all_conflicts_gpt20b.png of the paper.

    Args:
        samples: List of EvidenceActivations
        target_layers: Layers to visualize
        output_path: Where to save the figure
        steering_layers: Kept for interface compatibility with the steering code;
            not used by the plot
        title_suffix: Optional suffix added to the title (e.g., " (Simple Prompt)")
    """
    bias_data = compute_layer_evidence_bias_matrix(samples, target_layers)
    
    if not bias_data:
        print("  No data for stacked area chart")
        return
    
    conflict_types = ['ambiguity', 'granularity', 'perspective']
    
    # Find max evidence count
    max_evidence = 0
    for ct in conflict_types:
        if ct in bias_data:
            for layer_data in bias_data[ct].values():
                if layer_data:
                    max_evidence = max(max_evidence, max(layer_data.keys()) + 1)
    
    if max_evidence == 0:
        print("  No evidence data found")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    main_title = f'Evidence Bias Distribution by Layer (Stacked Area){title_suffix}\n(Shows which evidence dominates at each layer)'
    fig.suptitle(main_title, fontsize=14, fontweight='bold')
    
    for idx, conflict_type in enumerate(conflict_types):
        ax = axes[idx]
        
        if conflict_type not in bias_data:
            ax.set_title(f'{CONFLICT_DISPLAY_NAMES.get(conflict_type, conflict_type)}\n(No data)')
            ax.axis('off')
            continue
        
        ct_data = bias_data[conflict_type]
        layers = sorted(ct_data.keys())
        
        # Build data for stacked area
        evidence_series = {i: [] for i in range(max_evidence)}
        for layer in layers:
            for evi_id in range(max_evidence):
                evidence_series[evi_id].append(ct_data[layer].get(evi_id, 0))
        
        # Stack the areas
        x = np.array(layers)
        y_stack = np.zeros(len(layers))
        
        for evi_id in range(max_evidence):
            y = np.array(evidence_series[evi_id])
            color = SOURCE_COLORS[evi_id % len(SOURCE_COLORS)]
            ax.fill_between(x, y_stack, y_stack + y, 
                           color=color, alpha=0.8, label=f'Evidence {evi_id}')
            y_stack += y
        
        ax.set_xlabel('Layer', fontsize=11)
        ax.set_ylabel('Bias Ratio', fontsize=11)
        ax.set_title(f'{CONFLICT_DISPLAY_NAMES.get(conflict_type, conflict_type)}', 
                     fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.set_xlim(min(layers), max(layers))
        
        # X-axis: show every 2nd or 4th layer if too many
        tick_step = 4 if len(layers) > 20 else (2 if len(layers) > 10 else 1)
        ax.set_xticks(layers[::tick_step])
        
        # Add legend
        ax.legend(loc='upper right', fontsize=8, ncol=2)
        
        # Add grid
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved stacked area chart to {output_path}")


def visualize_bias_stacked_area_combined(
    samples: List[Any],
    target_layers: List[int],
    output_path: str,
    title_suffix: str = ""
):
    """
    Single stacked-area chart with the bias ratios averaged over all conflict
    types present in the samples.

    X-axis: layers
    Y-axis: average bias ratio per evidence (stacked to 1)
    Colors: evidence sources

    Args:
        samples: List of EvidenceActivations
        target_layers: Layers to visualize
        output_path: Where to save the figure
        title_suffix: Optional suffix added to the title (e.g., " (Simple Prompt)")
    """
    bias_data = compute_layer_evidence_bias_matrix(samples, target_layers)
    
    if not bias_data:
        print("  No data for combined stacked area chart")
        return
    
    conflict_types = ['ambiguity', 'granularity', 'perspective']
    available_types = [ct for ct in conflict_types if ct in bias_data]
    
    if not available_types:
        print("  No conflict types found")
        return
    
    # Find all layers and max evidence count
    all_layers = set()
    max_evidence = 0
    for ct in available_types:
        all_layers.update(bias_data[ct].keys())
        for layer_data in bias_data[ct].values():
            if layer_data:
                max_evidence = max(max_evidence, max(layer_data.keys()) + 1)
    
    layers = sorted(all_layers)
    
    if not layers or max_evidence == 0:
        print("  No layer/evidence data found")
        return
    
    # Compute average bias across conflict types for each layer and evidence
    avg_evidence_series = {i: [] for i in range(max_evidence)}
    
    for layer in layers:
        layer_avg = {i: [] for i in range(max_evidence)}
        
        for ct in available_types:
            if layer in bias_data[ct]:
                for evi_id in range(max_evidence):
                    ratio = bias_data[ct][layer].get(evi_id, 0)
                    layer_avg[evi_id].append(ratio)
        
        # Average across conflict types
        for evi_id in range(max_evidence):
            if layer_avg[evi_id]:
                avg_evidence_series[evi_id].append(np.mean(layer_avg[evi_id]))
            else:
                avg_evidence_series[evi_id].append(0)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.array(layers)
    y_stack = np.zeros(len(layers))
    
    for evi_id in range(max_evidence):
        y = np.array(avg_evidence_series[evi_id])
        color = SOURCE_COLORS[evi_id % len(SOURCE_COLORS)]
        ax.fill_between(x, y_stack, y_stack + y, 
                       color=color, alpha=0.8, label=f'Evidence {evi_id}')
        y_stack += y
    
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Average Bias Ratio', fontsize=12)
    ax.set_title(f'Combined Evidence Bias Distribution (Average of All Conflict Types){title_suffix}\n'
                 f'Layer {min(layers)}-{max(layers)}, {len(available_types)} conflict types averaged', 
                 fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.set_xlim(min(layers), max(layers))
    
    # X-axis ticks
    tick_step = 4 if len(layers) > 20 else (2 if len(layers) > 10 else 1)
    ax.set_xticks(layers[::tick_step])
    
    ax.legend(loc='upper right', fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved combined stacked area chart to {output_path}")
