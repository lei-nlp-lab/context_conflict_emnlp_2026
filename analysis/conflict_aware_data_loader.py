#!/usr/bin/env python3
"""
Conflict-type-aware data loader for the ContextConflict dataset.

Builds paired consistent / conflict prompts for each of the six contextual
conflict types (Section 4.1, "Conflict Awareness via Concept Vectors").  Both
prompts of a pair share the same question and the same prompt template; they
differ only in which evidence sources are included, so that the difference in
hidden states isolates the presence of a contextual conflict.

Per-type pairing rules (consistent vs conflict):
1. ambiguity_conflict:      source_1 vs source_1 + source_2
2. granularity_conflict:    first source vs all sources
                            (subfolders: medical_qa, ROAST-ABSA)
3. inferential_conflict:
   - entailment_bank:       all but the last 3 sources vs all sources
   - folio:                 all but the last 2 sources vs all sources
   - medical_qa:            sources with accuracy_labels == true vs all sources
4. misinformation_conflict: sources with accuracy_labels == true vs all sources
                            (subfolders: cb_claim_evidence, sci_data)
5. perspective_conflict:
   - allsides:              one perspective vs all perspectives
   - perspectrum:           same-stance sources (stance_labels) vs all sources
6. temporal_conflict:       first source vs all sources

Sampling: when a limit is given, files are drawn with random.sample after
seeding with `seed` (default 42); the per-type limit is split evenly across
subfolders.

Example:
    from analysis.conflict_aware_data_loader import ConflictAwareDataLoader
    loader = ConflictAwareDataLoader(data_root="ContextConflict_Dataset/data")
    consistent, conflict = loader.load_conflict_type_samples("temporal_conflict", limit=100, seed=42)
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s %(name)s %(lineno)s: %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class ConflictAwareDataLoader:
    """
    conflict-type-aware data loader
    generates consistent/conflict sample pairs based on conflict type characteristics
    """

    def __init__(self, data_root: str = "ContextConflict_Dataset/data"):
        self.data_root = Path(data_root)
        self.conflict_types = [
            "ambiguity_conflict",
            "granularity_conflict",
            "inferential_conflict",
            "misinformation_conflict",
            "perspective_conflict",
            "temporal_conflict"
        ]

    def load_json_files(self, conflict_type: str, subfolder: Optional[str] = None, limit: int = None, seed: int = 42) -> List[Dict]:
        """
        load all json files for specified conflict type

        Args:
            conflict_type: conflict type
            subfolder: subfolder (e.g. medical_qa, folio)
            limit: max number of files to load (randomly sampled if fewer than total)
            seed: random seed for sampling

        Returns:
            data_list: list of json data dicts
        """
        if subfolder:
            conflict_dir = self.data_root / conflict_type / subfolder
        else:
            conflict_dir = self.data_root / conflict_type

        if not conflict_dir.exists():
            logger.warning(f"Directory not found: {conflict_dir}")
            return []

        # recursively find all json files
        json_files = sorted(conflict_dir.rglob("*.json"))

        # randomly sample if limit is specified and fewer than total files
        if limit and len(json_files) > limit:
            total_files = len(json_files)
            random.seed(seed)
            json_files = random.sample(json_files, limit)
            logger.info(f"Randomly sampled {limit} files from {total_files} total files (seed={seed})")
        elif limit:
            json_files = json_files[:limit]

        data_list = []
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    data['file_path'] = str(json_file)
                    data['conflict_type'] = conflict_type
                    if subfolder:
                        data['subfolder'] = subfolder
                    data_list.append(data)
            except Exception as e:
                logger.warning(f"Error loading {json_file}: {e}")

        logger.info(f"Loaded {len(data_list)} files from {conflict_dir}")
        return data_list

    def format_prompt(self, question: str, evidence: List[str]) -> str:
        """
        format input prompt

        Args:
            question: question text
            evidence: list of evidence strings

        Returns:
            formatted_prompt: formatted prompt string
        """
        # build evidence text
        evidence_dict = {f"source_{i+1}": ev for i, ev in enumerate(evidence)}
        evidence_json = json.dumps(evidence_dict, ensure_ascii=False, indent=2)

        prompt = f"""You are a helpful assistant. Please answer the following question based on the provided content.

INSTRUCTIONS:
1. If the provided content does not contain relevant information to answer the question, respond with "I don't know" or "The provided content does not contain enough information to answer this question."
2. Always provide a direct answer:
   - If asked yes/no: answer "yes" or "no"
   - If asked true/false/uncertain: answer "true", "false", or "uncertain"
   - For QA questions: provide a direct answer (e.g., a year, a name, a brief phrase)
   - For summarization: provide a concise summary (2-3 sentences max)
3. Do NOT include any sections labeled "reference", "references", "source", "sources", or any tags such as <reference>...</reference>.

QUESTION:
{question}

CONTENT:
{evidence_json}

REQUIRED RESPONSE FORMAT:

<answer>[your direct answer or summary]</answer>

[Your detailed explanation with source citations, max 200 words]

IMPORTANT:
- Put ONLY your direct answer/summary inside <answer></answer> tags.
- Do NOT output any additional sections like "reference:" or "<reference>".
- For summarization: keep the summary concise (2-3 sentences max).
- NO additional explanations inside the <answer> tags.
- After </answer> tag: provide reasoning and evidence citations (max 200 words), written naturally in text form (not as a reference list).

YOUR RESPONSE:"""

        return prompt

    
    # 1. Ambiguity Conflict
    
    def process_ambiguity_conflict(self, data_list: List[Dict]) -> Tuple[List[str], List[str]]:
        """
        process ambiguity_conflict
        - consistent: use source_1 only
        - conflict: use source_1 + source_2
        """
        consistent_samples = []
        conflict_samples = []

        for data in data_list:
            question = data.get('question', '')
            content = data.get('content', {})

            source_1 = content.get('source_1', '')
            source_2 = content.get('source_2', '')

            if not question or not source_1 or not source_2:
                continue

            # consistent: use source_1 only
            consistent_prompt = self.format_prompt(question, [source_1])
            consistent_samples.append(consistent_prompt)

            # conflict: use source_1 + source_2
            conflict_prompt = self.format_prompt(question, [source_1, source_2])
            conflict_samples.append(conflict_prompt)

        logger.info(f"Ambiguity conflict: {len(consistent_samples)} consistent, {len(conflict_samples)} conflict")
        return consistent_samples, conflict_samples

    
    # 2. Granularity Conflict
    
    def process_granularity_conflict(self, data_list: List[Dict]) -> Tuple[List[str], List[str]]:
        """
        process granularity_conflict (medical_qa, ROAST-ABSA)
        - consistent: use first source only
        - conflict: use all sources
        """
        consistent_samples = []
        conflict_samples = []

        for data in data_list:
            question = data.get('question', '')
            content = data.get('content', {})

            # extract all sources
            sources = []
            for key in sorted(content.keys()):
                if key.startswith('source_') and content[key]:
                    sources.append(content[key])

            if not question or len(sources) < 2:
                continue

            # consistent: use first source only
            consistent_prompt = self.format_prompt(question, [sources[0]])
            consistent_samples.append(consistent_prompt)

            # conflict: use all sources
            conflict_prompt = self.format_prompt(question, sources)
            conflict_samples.append(conflict_prompt)

        logger.info(f"Granularity conflict: {len(consistent_samples)} consistent, {len(conflict_samples)} conflict")
        return consistent_samples, conflict_samples

    
    # 3. Inferential Conflict

    def process_inferential_conflict_entailment_bank(self, data_list: List[Dict]) -> Tuple[List[str], List[str]]:
        """
        process inferential_conflict/entailment_bank
        - consistent: exclude last 3 sources
        - conflict: use all sources
        """
        consistent_samples = []
        conflict_samples = []

        for data in data_list:
            question = data.get('question', '')
            content = data.get('content', {})

            # extract all sources
            sources = []
            for key in sorted(content.keys()):
                if key.startswith('source_') and content[key]:
                    sources.append(content[key])

            if not question or len(sources) < 4:  # Need at least 4 sources (1 correct + 3 incorrect)
                continue

            # consistent: exclude last 3 sources
            consistent_sources = sources[:-3]
            consistent_prompt = self.format_prompt(question, consistent_sources)
            consistent_samples.append(consistent_prompt)

            # conflict: use all sources
            conflict_prompt = self.format_prompt(question, sources)
            conflict_samples.append(conflict_prompt)

        logger.info(f"Inferential conflict (entailment_bank): {len(consistent_samples)} consistent, {len(conflict_samples)} conflict")
        return consistent_samples, conflict_samples

    def process_inferential_conflict_folio(self, data_list: List[Dict]) -> Tuple[List[str], List[str]]:
        """
        process inferential_conflict/folio
        - consistent: exclude last 2 sources
        - conflict: use all sources
        """
        consistent_samples = []
        conflict_samples = []

        for data in data_list:
            question = data.get('question', '')
            content = data.get('content', {})

            # extract all sources
            sources = []
            for key in sorted(content.keys()):
                if key.startswith('source_') and content[key]:
                    sources.append(content[key])

            if not question or len(sources) < 3:  # Need at least 3 sources (1 correct + 2 incorrect)
                continue

            # consistent: exclude last 2 sources
            consistent_sources = sources[:-2]
            consistent_prompt = self.format_prompt(question, consistent_sources)
            consistent_samples.append(consistent_prompt)

            # conflict: use all sources
            conflict_prompt = self.format_prompt(question, sources)
            conflict_samples.append(conflict_prompt)

        logger.info(f"Inferential conflict (folio): {len(consistent_samples)} consistent, {len(conflict_samples)} conflict")
        return consistent_samples, conflict_samples

    def process_inferential_conflict_medical_qa(self, data_list: List[Dict]) -> Tuple[List[str], List[str]]:
        """
        process inferential_conflict/medical_qa
        - consistent: use sources with accuracy_labels=true only
        - conflict: use all sources (true + false)
        """
        consistent_samples = []
        conflict_samples = []

        for data in data_list:
            question = data.get('question', '')
            content = data.get('content', {})
            accuracy_labels = data.get('accuracy_labels', [])

            # extract all sources
            sources = []
            for key in sorted(content.keys()):
                if key.startswith('source_') and content[key]:
                    sources.append(content[key])

            if not question or not sources or len(accuracy_labels) != len(sources):
                continue

            # filter sources with true labels
            true_sources = [src for src, label in zip(sources, accuracy_labels) if label]

            if not true_sources:
                continue

            # consistent: use true sources only
            consistent_prompt = self.format_prompt(question, true_sources)
            consistent_samples.append(consistent_prompt)

            # conflict: use all sources (true + false)
            conflict_prompt = self.format_prompt(question, sources)
            conflict_samples.append(conflict_prompt)

        logger.info(f"Inferential conflict (medical_qa): {len(consistent_samples)} consistent, {len(conflict_samples)} conflict")
        return consistent_samples, conflict_samples

    
    # 4. Misinformation Conflict
    
    def process_misinformation_conflict(self, data_list: List[Dict]) -> Tuple[List[str], List[str]]:
        """
        process misinformation_conflict (cb_claim_evidence, sci_data)
        - consistent: use sources with accuracy_labels=true only
        - conflict: use all sources (true + false)
        """
        consistent_samples = []
        conflict_samples = []

        for data in data_list:
            question = data.get('question', '')
            content = data.get('content', {})
            accuracy_labels = data.get('accuracy_labels', [])

            # extract all sources
            sources = []
            for key in sorted(content.keys()):
                if key.startswith('source_') and content[key]:
                    sources.append(content[key])

            if not question or not sources or len(accuracy_labels) != len(sources):
                continue

            # filter sources with true labels
            true_sources = [src for src, label in zip(sources, accuracy_labels) if label]

            if not true_sources:
                continue

            # consistent: use true sources only
            consistent_prompt = self.format_prompt(question, true_sources)
            consistent_samples.append(consistent_prompt)

            # conflict: use all sources (true + false)
            conflict_prompt = self.format_prompt(question, sources)
            conflict_samples.append(conflict_prompt)

        logger.info(f"Misinformation conflict: {len(consistent_samples)} consistent, {len(conflict_samples)} conflict")
        return consistent_samples, conflict_samples

    
    # 5. Perspective Conflict
    
    def process_perspective_conflict_allsides(self, data_list: List[Dict]) -> Tuple[List[str], List[str]]:
        """
        process perspective_conflict/allsides
        - consistent: use one perspective only
        - conflict: use all three perspectives
        
        Supports two data formats:
        1. Dataset format (perspective_conflict/allsides/): a 'content' dict with source_1, source_2, ...
        2. Alternative format: a 'perspectives' dict with Left/Center/Right article lists
        """
        consistent_samples = []
        conflict_samples = []

        for data in data_list:
            question = data.get('question', '')
            sources = []
            
            # Try format 1: 'content' field (original data)
            content = data.get('content', {})
            if content and isinstance(content, dict):
                for key in sorted(content.keys()):
                    if key.startswith('source_') and content[key]:
                        sources.append(content[key])
            
            # Try format 2: 'perspectives' field (result data) if no sources found
            if not sources:
                perspectives = data.get('perspectives', {})
                for side in ['Left', 'Center', 'Right']:
                    if side in perspectives and isinstance(perspectives[side], list):
                        for article in perspectives[side]:
                            if isinstance(article, dict) and 'content' in article:
                                sources.append(article['content'])

            if not question or len(sources) < 2:
                continue

            # consistent: use first perspective only
            consistent_prompt = self.format_prompt(question, [sources[0]])
            consistent_samples.append(consistent_prompt)

            # conflict: use all perspectives
            conflict_prompt = self.format_prompt(question, sources)
            conflict_samples.append(conflict_prompt)

        logger.info(f"Perspective conflict (allsides): {len(consistent_samples)} consistent, {len(conflict_samples)} conflict")
        return consistent_samples, conflict_samples

    def process_perspective_conflict_perspectrum(self, data_list: List[Dict]) -> Tuple[List[str], List[str]]:
        """
        process perspective_conflict/perspectrum using stance_labels
        - consistent: sources from the same stance group (all label=0 or all label=1)
        - conflict: sources from both stance groups (mixed labels)

        Falls back to old behavior (consistent=first source, conflict=all) if stance_labels missing.
        """
        consistent_samples = []
        conflict_samples = []
        used_stance = 0
        no_stance = 0

        for data in data_list:
            question = data.get('question', '')
            content = data.get('content', {})
            stance_labels = data.get('stance_labels', {})

            source_keys = sorted(k for k in content if k.startswith('source_') and content[k])
            if not question or len(source_keys) < 2:
                continue

            if stance_labels:
                # Split sources by stance
                group_0 = [content[k] for k in source_keys if stance_labels.get(k) == 0]
                group_1 = [content[k] for k in source_keys if stance_labels.get(k) == 1]

                if not group_0 or not group_1:
                    # Fallback if stance labels are all same
                    no_stance += 1
                    consistent_prompt = self.format_prompt(question, [content[source_keys[0]]])
                    conflict_prompt = self.format_prompt(question, [content[k] for k in source_keys])
                else:
                    used_stance += 1
                    # consistent: use the larger stance group (same-perspective sources)
                    same_group = group_0 if len(group_0) >= len(group_1) else group_1
                    consistent_prompt = self.format_prompt(question, same_group)
                    # conflict: mix both stance groups (all sources)
                    conflict_prompt = self.format_prompt(question, [content[k] for k in source_keys])

                consistent_samples.append(consistent_prompt)
                conflict_samples.append(conflict_prompt)
            else:
                # Fallback: no stance labels
                no_stance += 1
                consistent_prompt = self.format_prompt(question, [content[source_keys[0]]])
                conflict_prompt = self.format_prompt(question, [content[k] for k in source_keys])
                consistent_samples.append(consistent_prompt)
                conflict_samples.append(conflict_prompt)

        logger.info(f"Perspective conflict (perspectrum): {len(consistent_samples)} consistent, {len(conflict_samples)} conflict")
        logger.info(f"  Stance-based: {used_stance}, fallback: {no_stance}")
        return consistent_samples, conflict_samples

    
    # 6. Temporal Conflict
    
    def process_temporal_conflict(self, data_list: List[Dict]) -> Tuple[List[str], List[str]]:
        """
        process temporal_conflict
        - consistent: use first source only
        - conflict: use all sources
        """
        consistent_samples = []
        conflict_samples = []

        for data in data_list:
            question = data.get('question', '')
            content = data.get('content', {})

            # extract all sources
            sources = []
            for key in sorted(content.keys()):
                if key.startswith('source_') and content[key]:
                    sources.append(content[key])

            if not question or len(sources) < 2:
                continue

            # consistent: use first source only
            consistent_prompt = self.format_prompt(question, [sources[0]])
            consistent_samples.append(consistent_prompt)

            # conflict: use all sources
            conflict_prompt = self.format_prompt(question, sources)
            conflict_samples.append(conflict_prompt)

        logger.info(f"Temporal conflict: {len(consistent_samples)} consistent, {len(conflict_samples)} conflict")
        return consistent_samples, conflict_samples

    
    # unified interface
    
    def load_conflict_type_samples(
        self,
        conflict_type: str,
        limit: int = 100,
        seed: int = 42
    ) -> Tuple[List[str], List[str]]:
        """
        load sample pairs for specified conflict type

        Args:
            conflict_type: conflict type
            limit: total sample limit per conflict type (distributed across subfolders)
            seed: random seed

        Returns:
            (consistent_samples, conflict_samples)
        """
        random.seed(seed)

        logger.info(f"\n{'='*60}")
        logger.info(f"Loading {conflict_type}")
        logger.info(f"{'='*60}")

        if conflict_type == "ambiguity_conflict":
            data_list = self.load_json_files(conflict_type, limit=limit, seed=seed)
            return self.process_ambiguity_conflict(data_list)

        elif conflict_type == "granularity_conflict":
            # process two subfolders: medical_qa and ROAST-ABSA
            all_consistent = []
            all_conflict = []

            subfolders = ["medical_qa", "ROAST-ABSA"]
            limit_per_subfolder = limit // len(subfolders) if limit else None

            for subfolder in subfolders:
                data_list = self.load_json_files(conflict_type, subfolder=subfolder, limit=limit_per_subfolder, seed=seed)
                if data_list:
                    consistent, conflict = self.process_granularity_conflict(data_list)
                    all_consistent.extend(consistent)
                    all_conflict.extend(conflict)

            return all_consistent, all_conflict

        elif conflict_type == "inferential_conflict":
            # process three subfolders: entailment_bank, folio, medical_qa
            all_consistent = []
            all_conflict = []

            subfolders = ["entailment_bank", "folio", "medical_qa"]
            limit_per_subfolder = limit // len(subfolders) if limit else None

            # entailment_bank: exclude last 3 sources
            data_list = self.load_json_files(conflict_type, subfolder="entailment_bank", limit=limit_per_subfolder, seed=seed)
            if data_list:
                consistent, conflict = self.process_inferential_conflict_entailment_bank(data_list)
                all_consistent.extend(consistent)
                all_conflict.extend(conflict)

            # folio: exclude last 2 sources
            data_list = self.load_json_files(conflict_type, subfolder="folio", limit=limit_per_subfolder, seed=seed)
            if data_list:
                consistent, conflict = self.process_inferential_conflict_folio(data_list)
                all_consistent.extend(consistent)
                all_conflict.extend(conflict)

            # medical_qa: use accuracy_labels
            data_list = self.load_json_files(conflict_type, subfolder="medical_qa", limit=limit_per_subfolder, seed=seed)
            if data_list:
                consistent, conflict = self.process_inferential_conflict_medical_qa(data_list)
                all_consistent.extend(consistent)
                all_conflict.extend(conflict)

            return all_consistent, all_conflict

        elif conflict_type == "misinformation_conflict":
            # process two subfolders: cb_claim_evidence and sci_data
            all_consistent = []
            all_conflict = []

            subfolders = ["cb_claim_evidence", "sci_data"]
            limit_per_subfolder = limit // len(subfolders) if limit else None

            for subfolder in subfolders:
                data_list = self.load_json_files(conflict_type, subfolder=subfolder, limit=limit_per_subfolder, seed=seed)
                if data_list:
                    consistent, conflict = self.process_misinformation_conflict(data_list)
                    all_consistent.extend(consistent)
                    all_conflict.extend(conflict)

            return all_consistent, all_conflict

        elif conflict_type == "perspective_conflict":
            # process two subfolders: allsides and perspectrum
            all_consistent = []
            all_conflict = []

            subfolders = ["allsides", "perspectrum"]
            limit_per_subfolder = limit // len(subfolders) if limit else None

            # allsides
            data_list = self.load_json_files(conflict_type, subfolder="allsides", limit=limit_per_subfolder, seed=seed)
            if data_list:
                consistent, conflict = self.process_perspective_conflict_allsides(data_list)
                all_consistent.extend(consistent)
                all_conflict.extend(conflict)

            # perspectrum
            data_list = self.load_json_files(conflict_type, subfolder="perspectrum", limit=limit_per_subfolder, seed=seed)
            if data_list:
                consistent, conflict = self.process_perspective_conflict_perspectrum(data_list)
                all_consistent.extend(consistent)
                all_conflict.extend(conflict)

            return all_consistent, all_conflict

        elif conflict_type == "temporal_conflict":
            # load all json files from folder (including subfolders)
            data_list = self.load_json_files(conflict_type, limit=limit, seed=seed)
            return self.process_temporal_conflict(data_list)

        else:
            raise ValueError(f"Unknown conflict type: {conflict_type}")

    def get_all_conflict_types_samples(
        self,
        limit_per_type: int = 50,
        seed: int = 42
    ) -> Dict[str, Tuple[List[str], List[str]]]:
        """
        load samples for all conflict types

        Args:
            limit_per_type: sample limit per conflict type
            seed: random seed

        Returns:
            {conflict_type: (consistent_samples, conflict_samples)}
        """
        all_samples = {}

        for conflict_type in self.conflict_types:
            try:
                consistent, conflict = self.load_conflict_type_samples(
                    conflict_type,
                    limit=limit_per_type,
                    seed=seed
                )
                all_samples[conflict_type] = (consistent, conflict)
            except Exception as e:
                logger.error(f"Error loading {conflict_type}: {e}")
                all_samples[conflict_type] = ([], [])

        return all_samples
