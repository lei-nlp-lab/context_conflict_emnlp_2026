"""
Evidence activation collector for the position-bias measurement.

Given a ContextConflict sample with K evidence pieces (content.source_1 ...
content.source_K), this module builds one combined-evidence prompt and K
single-evidence prompts, runs a forward pass for each, and records the
residual-stream activation at the selected layers (last non-special token by
default). The result is an EvidenceActivations object holding

    combined_evidence_acts[l]          c^(l)   (combined prompt, layer l)
    single_evidence_acts[i][l]         a_i^(l) (single-evidence prompt i, layer l)

These activations are the input of the directional position-bias measurement
of Section 4.3 ("Evidence Position Bias in Internal Representations") and the
appendix "Bias Measurement Implementation Notes", implemented in
analysis/position_bias/directional_bias.py, and they are also reused by the
activation-steering method (synthesis direction = c - mean_i a_i).

The prompts (system prompts and user templates) are kept verbatim from the
experiments; "simple" and "detailed" combined-evidence prompts are supported
through prompt_type.

Typical use (see analysis/position_bias/run_directional_bias.py):
    collector = EvidenceSynthesisCollector(model, tokenizer, device="cuda",
                                           target_layers=list(range(n_layers)))
    sample = collector.collect_sample_with_prompt_type(
        sample_json, task_type="ambiguity", sample_id="ambiguity_12",
        prompt_type="simple")
"""

import json
import pickle
import numpy as np
import torch
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from tqdm import tqdm


# =============================================================================
# System Prompts for Evidence Synthesis
# =============================================================================

SIMPLE_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions based on provided evidence. "
    "Answer concisely in 1-2 sentences."
)

DETAILED_SYSTEM_PROMPT = (
    "You are a neutral information synthesis assistant. "
    "Analyze all sources with equal importance and provide unbiased, balanced answers."
)

SINGLE_EVIDENCE_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question based ONLY on the provided evidence. "
    "Be concise and factual."
)


@dataclass
class EvidenceActivations:
    """
    Container for activations from a single sample.
    
    Structure:
    - sample_id: Unique identifier (e.g., "ambiguity_sample123")
    - question: The question being answered
    - n_sources: Number of evidence sources
    - task_type: Type of task (ambiguity, granularity, perspective)
    - single_evidence_acts: {source_idx: {layer_idx: np.ndarray[hidden_dim]}}
      Activations when model sees only one evidence source
    - combined_evidence_acts: {layer_idx: np.ndarray[hidden_dim]}
      Activations when model sees all evidence sources together
    """
    sample_id: str
    question: str
    n_sources: int
    task_type: str = ""
    # Activations for each single-evidence prompt: {source_idx: {layer: activation}}
    single_evidence_acts: Dict[int, Dict[int, np.ndarray]] = field(default_factory=dict)
    # Activations for combined-evidence prompt: {layer: activation}
    combined_evidence_acts: Dict[int, np.ndarray] = field(default_factory=dict)


class EvidenceSynthesisCollector:
    """
    Collects single-evidence and combined-evidence activations.
    Used for the directional position-bias measurement and for activation steering.
    
    For each sample with n evidence sources:
    1. Generate n prompts, each with only one evidence
    2. Generate 1 prompt with all evidence combined
    3. Collect activations from specified layers for each prompt
    
    The synthesis direction is: combined_state - mean(single_states)
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        target_layers: Optional[List[int]] = None,
        activation_mode: str = "last_token",
    ):
        """
        Args:
            model: HuggingFace model
            tokenizer: HuggingFace tokenizer
            device: Device to use
            target_layers: Layers to collect activations from
            activation_mode: How to extract activation ('last_token', 'mean_token')
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # Validate activation_mode
        valid_modes = ["last_token", "mean_token"]
        if activation_mode not in valid_modes:
            raise ValueError(f"Invalid activation_mode: {activation_mode}. Must be one of {valid_modes}")
        self.activation_mode = activation_mode
        
        # Set default target layers (middle to late layers)
        if target_layers is None:
            n_layers = model.config.num_hidden_layers
            self.target_layers = list(range(n_layers // 3, n_layers - 2))
        else:
            self.target_layers = target_layers
            
        self.hidden_size = model.config.hidden_size
        
        # Auto-detect model layer structure
        self._layers = self._get_layers()
        
        # Storage for collected activations
        self.samples: List[EvidenceActivations] = []
        
        # Hooks (use list to ensure proper cleanup)
        self._hooks = []
        self._current_activations = {}
        
        # Check if tokenizer supports chat template
        self._use_chat_template = hasattr(tokenizer, 'apply_chat_template') and \
                                   tokenizer.chat_template is not None
        
        print(f"EvidenceSynthesisCollector initialized:")
        print(f"  Target layers: {self.target_layers}")
        print(f"  Activation mode: {self.activation_mode}")
        print(f"  Model layer structure: {type(self._layers).__name__}")
        print(f"  Use chat template: {self._use_chat_template}")
    
    def _get_layers(self):
        """Auto-detect model layer structure for different HuggingFace models."""
        m = self.model
        # LLaMA / Mistral / Qwen
        if hasattr(m, "model") and hasattr(m.model, "layers"):
            return m.model.layers
        # GPT-2 / GPT-Neo
        if hasattr(m, "transformer") and hasattr(m.transformer, "h"):
            return m.transformer.h
        # BERT / RoBERTa
        if hasattr(m, "encoder") and hasattr(m.encoder, "layer"):
            return m.encoder.layer
        # Fallback: try common patterns
        if hasattr(m, "layers"):
            return m.layers
        raise ValueError(
            f"Unsupported model architecture: {type(m).__name__}. "
            "Cannot find layer structure."
        )
        
    def _hook_fn(self, layer_idx: int, module, input, output):
        """
        Hook function to collect hidden states.
        Uses functools.partial to avoid closure bugs.
        """
        # Handle different output formats from HF models
        if hasattr(output, "last_hidden_state"):
            hidden = output.last_hidden_state
        elif isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        # Use clone().detach() to avoid in-place modification issues
        self._current_activations[layer_idx] = hidden.clone().detach()
    
    def _register_hooks(self):
        """Register forward hooks to collect hidden states."""
        self._hooks = []
        self._current_activations = {}
        
        for layer_idx in self.target_layers:
            layer = self._layers[layer_idx]
            # Use functools.partial to avoid closure bug
            handle = layer.register_forward_hook(
                partial(self._hook_fn, layer_idx)
            )
            self._hooks.append(handle)
            
    def _remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks = []
        
    def _get_last_meaningful_token_index(self, input_ids: torch.Tensor) -> int:
        """
        Get the index of the last meaningful (non-special) token.
        
        This avoids taking special tokens like [SEP], </s>, <pad> etc.
        
        Args:
            input_ids: [batch, seq_len] tensor of token ids
            
        Returns:
            Index of the last meaningful token
        """
        ids = input_ids[0].tolist()
        special_ids = set(self.tokenizer.all_special_ids)
        
        # Find last non-special token
        for i in range(len(ids) - 1, -1, -1):
            if ids[i] not in special_ids:
                return i
        
        # Fallback to last token if all are special
        return len(ids) - 1
        
    def _extract_activation(
        self, 
        hidden_states: torch.Tensor, 
        input_ids: torch.Tensor
    ) -> np.ndarray:
        """
        Extract activation vector from hidden states.
        
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            input_ids: [batch, seq_len] for finding last meaningful token
            
        Returns:
            Activation vector [hidden_dim]
        """
        if self.activation_mode == "last_token":
            # Get index of last meaningful (non-special) token
            last_idx = self._get_last_meaningful_token_index(input_ids)
            # Convert to float32 first (numpy doesn't support bfloat16)
            act = hidden_states[0, last_idx, :].float().cpu().numpy()
        elif self.activation_mode == "mean_token":
            # Use mean over all tokens
            # Convert to float32 first (numpy doesn't support bfloat16)
            act = hidden_states[0, :, :].mean(dim=0).float().cpu().numpy()
        else:
            raise ValueError(f"Unknown activation_mode: {self.activation_mode}")
            
        return act
    
    def _build_single_evidence_prompt(
        self,
        question: str,
        source_key: str,
        source_content: str,
        task_type: str
    ) -> str:
        """Build prompt with only one evidence source."""
        user_content = (
            f"Question: {question}\n\n"
            f"[Evidence]: {source_content}\n\n"
            "Based on this evidence, provide your answer inside <answer></answer> in 1-2 sentences."
        )
        
        # Apply chat template for instruction-tuned models
        if self._use_chat_template:
            messages = [
                {"role": "system", "content": SINGLE_EVIDENCE_SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            prompt = f"{SINGLE_EVIDENCE_SYSTEM_PROMPT}\n\n{user_content}"
            
        return prompt
    
    def _build_combined_evidence_prompt(
        self,
        question: str,
        sources: Dict[str, str],
        task_type: str,
        prompt_type: str = "detailed"
    ) -> str:
        """
        Build prompt with all evidence sources combined.
        
        Args:
            question: The question to answer
            sources: Dict of source_key -> source_content
            task_type: Type of task (ambiguity, granularity, perspective)
            prompt_type: Type of prompt to use:
                - "detailed": Full instructions emphasizing synthesis and neutrality
                - "simple": Minimal prompt without neutrality instructions
        """
        # Build sources text with consistent ordering
        sources_text = ""
        for i, key in enumerate(sorted(sources.keys()), 1):
            sources_text += f"\n[Evidence {i}]: {sources[key]}\n"
        
        if prompt_type == "simple":
            # Simple prompt: minimal instructions
            system_content = SIMPLE_SYSTEM_PROMPT
            user_content = (
                f"Based on the following evidence, answer the question in 1-2 sentences.\n\n"
                f"Question: {question}\n\n"
                f"Evidence:{sources_text}"
            )
        else:
            # Detailed prompt: emphasize neutrality and synthesis
            system_content = DETAILED_SYSTEM_PROMPT
            user_content = (
                "Instructions:\n"
                "First, briefly analyze all the evidence in 5-6 sentences. Then, provide a comprehensive and fair synthesized answer to the question inside <answer></answer> in 2-3 sentences.\n\n"
                f"Question: {question}\n\n"
                f"Evidence: {sources_text}"
            )
        
        # Apply chat template for instruction-tuned models
        if self._use_chat_template:
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            prompt = f"{system_content}\n\n{user_content}"
            
        return prompt
    
    def _collect_activation_for_prompt(self, prompt: str) -> Dict[int, np.ndarray]:
        """
        Run forward pass and collect activations for a prompt.
        
        Returns:
            Dict mapping layer_idx to activation vector (sorted by layer index)
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        
        # Clear previous activations before registering new hooks
        self._current_activations = {}
        self._register_hooks()
        
        activations = {}
        try:
            with torch.no_grad():
                _ = self.model(**inputs)
            
            # Extract activations (sorted by layer index for consistency)
            for layer_idx in sorted(self._current_activations.keys()):
                hidden = self._current_activations[layer_idx]
                activations[layer_idx] = self._extract_activation(hidden, input_ids)
                
        except Exception as e:
            print(f"    Error during forward pass: {e}")
            # Return empty dict on error
            activations = {}
        finally:
            # Always clean up hooks and clear activations
            self._remove_hooks()
            self._current_activations = {}
            
        return activations
    
    def collect_sample(
        self,
        sample_data: Dict[str, Any],
        task_type: str,
        sample_id: str
    ) -> EvidenceActivations:
        """
        Collect activations for a single sample.
        
        Args:
            sample_data: JSON data for the sample
            task_type: Type of task (ambiguity, granularity, perspective)
            sample_id: Unique identifier for the sample
            
        Returns:
            EvidenceActivations object
        """
        question = sample_data.get("question", "")
        content = sample_data.get("content", {})
        
        # Get all sources (sorted for consistent ordering)
        sources = {}
        for key, value in content.items():
            if key.startswith("source_"):
                sources[key] = value
                
        n_sources = len(sources)
        if n_sources == 0:
            return None
            
        # Create activation container
        evidence_acts = EvidenceActivations(
            sample_id=sample_id,
            question=question,
            n_sources=n_sources,
            task_type=task_type
        )
        
        # Collect single-evidence activations (sorted keys for reproducibility)
        for idx, source_key in enumerate(sorted(sources.keys())):
            source_content = sources[source_key]
            prompt = self._build_single_evidence_prompt(
                question, source_key, source_content, task_type
            )
            acts = self._collect_activation_for_prompt(prompt)
            evidence_acts.single_evidence_acts[idx] = acts
            
        # Collect combined-evidence activations (using detailed prompt by default)
        combined_prompt = self._build_combined_evidence_prompt(
            question, sources, task_type, prompt_type="detailed"
        )
        evidence_acts.combined_evidence_acts = self._collect_activation_for_prompt(combined_prompt)
        
        return evidence_acts
    
    def collect_sample_with_prompt_type(
        self,
        sample_data: Dict[str, Any],
        task_type: str,
        sample_id: str,
        prompt_type: str = "detailed"
    ) -> EvidenceActivations:
        """
        Collect activations for a single sample with specified prompt type.
        
        Args:
            sample_data: JSON data for the sample
            task_type: Type of task (ambiguity, granularity, perspective)
            sample_id: Unique identifier for the sample
            prompt_type: Type of prompt ("detailed" or "simple")
            
        Returns:
            EvidenceActivations object
        """
        question = sample_data.get("question", "")
        content = sample_data.get("content", {})
        
        # Get all sources (sorted for consistent ordering)
        sources = {}
        for key, value in content.items():
            if key.startswith("source_"):
                sources[key] = value
                
        n_sources = len(sources)
        if n_sources == 0:
            return None
            
        # Create activation container
        evidence_acts = EvidenceActivations(
            sample_id=f"{sample_id}_{prompt_type}",
            question=question,
            n_sources=n_sources,
            task_type=task_type
        )
        
        # Collect single-evidence activations (same for both prompt types)
        for idx, source_key in enumerate(sorted(sources.keys())):
            source_content = sources[source_key]
            prompt = self._build_single_evidence_prompt(
                question, source_key, source_content, task_type
            )
            acts = self._collect_activation_for_prompt(prompt)
            evidence_acts.single_evidence_acts[idx] = acts
            
        # Collect combined-evidence activations with specified prompt type
        combined_prompt = self._build_combined_evidence_prompt(
            question, sources, task_type, prompt_type=prompt_type
        )
        evidence_acts.combined_evidence_acts = self._collect_activation_for_prompt(combined_prompt)
        
        return evidence_acts
    
    def collect_from_directory(
        self,
        data_dir: str,
        task_type: str,
        train_ratio: float = 0.2,
        max_samples: int = 10000,
        seed: int = 42
    ) -> int:
        """
        Collect activations from training split of data in a directory.
        
        Uses the same split as the steering evaluation (test split):
        - Shuffle all files with seed=42
        - Take first train_ratio (20%) as training set
        - Remaining 80% is reserved for testing
        
        Args:
            data_dir: Directory containing JSON files
            task_type: Type of task
            train_ratio: Ratio of data for training (default: 0.2 = 20%)
            max_samples: Maximum number of samples to process (default: 10000)
            seed: Random seed for sampling (default: 42)
            
        Returns:
            Number of samples processed
        """
        import random
        
        data_path = Path(data_dir)
        if not data_path.exists():
            print(f"  Directory not found: {data_dir}")
            return 0
            
        # Find all JSON files recursively (using rglob for nested directories)
        all_files = list(data_path.rglob("*.json"))
                    
        if not all_files:
            print(f"  No JSON files found in {data_dir}")
            return 0
        
        # CRITICAL: Sort first for deterministic ordering across OS/runs
        # rglob order is not guaranteed to be consistent
        all_files = sorted(all_files, key=lambda p: str(p))
        
        total_files = len(all_files)
        
        # Same split as the steering evaluation: shuffle with seed, then take first train_ratio fraction
        random.seed(seed)
        random.shuffle(all_files)
        
        # Calculate train size, but don't exceed total files
        n_train = int(total_files * train_ratio)
        n_train = max(n_train, min(10, total_files))  # At least 10, but not more than total
        json_files = all_files[:n_train]
        
        # Apply max_samples cap if needed
        if len(json_files) > max_samples:
            # Use same seed for reproducibility
            np.random.seed(seed)
            indices = np.random.choice(len(json_files), max_samples, replace=False)
            json_files = [json_files[i] for i in sorted(indices)]
            
        print(f"  Processing {len(json_files)}/{total_files} samples ({train_ratio*100:.0f}% train split) from {task_type}...")
        
        for json_file in tqdm(json_files, desc=f"  {task_type}"):
            try:
                with open(json_file, 'r') as f:
                    sample_data = json.load(f)
                    
                sample_id = f"{task_type}_{json_file.stem}"
                evidence_acts = self.collect_sample(sample_data, task_type, sample_id)
                
                if evidence_acts is not None:
                    self.samples.append(evidence_acts)
                    
            except Exception as e:
                print(f"    Error processing {json_file}: {e}")
                continue
                
        return len(json_files)
    
    def save_activations(self, output_dir: str):
        """
        Save collected activations to disk.
        
        Output files:
        - evidence_activations.pkl: List[EvidenceActivations]
          Each EvidenceActivations contains:
            - single_evidence_acts: {source_idx: {layer_idx: activation}}
            - combined_evidence_acts: {layer_idx: activation}
        - collection_summary.json: Metadata about the collection
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save as pickle
        with open(output_path / "evidence_activations.pkl", 'wb') as f:
            pickle.dump(self.samples, f)
            
        # Save detailed summary
        summary = {
            "n_samples": len(self.samples),
            "target_layers": self.target_layers,
            "activation_mode": self.activation_mode,
            "hidden_size": self.hidden_size,
            "data_structure": {
                "file": "evidence_activations.pkl",
                "format": "List[EvidenceActivations]",
                "fields": {
                    "single_evidence_acts": "{source_idx: {layer_idx: np.ndarray[hidden_dim]}}",
                    "combined_evidence_acts": "{layer_idx: np.ndarray[hidden_dim]}"
                }
            },
            "samples_per_n_sources": {},
            "samples_per_task_type": {}
        }
        
        for sample in self.samples:
            # Count by number of sources
            n = sample.n_sources
            summary["samples_per_n_sources"][n] = summary["samples_per_n_sources"].get(n, 0) + 1
            # Count by task type
            tt = sample.task_type
            summary["samples_per_task_type"][tt] = summary["samples_per_task_type"].get(tt, 0) + 1
            
        with open(output_path / "collection_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
            
        print(f"Saved {len(self.samples)} samples to {output_path}")
        print(f"  Samples by task: {summary['samples_per_task_type']}")
        
    def clear(self):
        """Clear collected samples."""
        self.samples = []

