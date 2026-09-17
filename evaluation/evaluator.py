"""
Shapley-based evidence attribution and auxiliary response metrics for ContextConflict.

This module implements the response-level evaluation of Section 3.1 of
"Large Language Models in Resolving Contextual Knowledge Conflicts" (EMNLP 2026).

Evidence Balance (summarization-type conflicts: ambiguity, granularity, perspective)
-----------------------------------------------------------------------------------
Given a question q, evidence pieces E = {e_1, ..., e_n} and a model response R, every
evidence piece receives a Shapley value phi_i, computed exactly over all 2^n evidence
subsets. The value of a subset S is the length-normalized log-likelihood of the
response under a frozen external scorer M (not the model that produced R):

    v(S) = (1/|R|) * sum_t log p_M(r_t | r_<t, q, E_S) = -avg_nll(R | q, E_S)

Negative Shapley values are clipped at zero and the remaining mass is normalized to
contribution shares p_i (sum_i p_i = 1). The Balance score (Gini coefficient of p,
lower = more balanced) is computed from p downstream in evaluation/balance_score.py.

Where the paper metrics live in this file:
  * HFGenerativeModel.score          value function: average NLL of the response under a
                                     local Hugging Face causal LM. Default scorer in the
                                     paper: meta-llama/Llama-3.2-1B-Instruct; robustness
                                     runs use Llama-3.1-8B-Instruct and Gemma-2B.
  * APIGenerativeModel.score         same value function through an OpenAI-compatible
                                     completions endpoint that returns echo logprobs.
  * Evaluator.perplexity             builds the fixed scoring prompt (question + evidence
                                     subset) and returns avg NLL (or PPL = exp(avg NLL)).
  * Evaluator.perplexity_evaluation  exact Shapley computation, clipping and normalization;
                                     returns 'shapley_values' and 'prob_distribution',
                                     which evaluation/batch_evaluate.py stores under the
                                     "eval" key of every response JSON.

Auxiliary components kept from the research code base (not used for the paper's
headline numbers):
  * HFNLIModel, Evaluator.atomic_claim_evaluation, Evaluator.faithfulness_evaluation
        NLI-based support/undermine counts and a FActScore-style precision with
        sentence-level claims (roberta-large-mnli). The faithfulness reported in the
        paper is RAGAS, see evaluation/faithfulness_ragas.py.
  * OpenAIHelper                     LLM-based atomic claim splitting.
  * Evaluator.similarity_evaluation  embedding-similarity attribution.
  * Evaluator.cluster                optional k-means grouping of evidence; disabled in
                                     all released runs, so each evidence piece is one
                                     Shapley player.

Credentials are read from environment variables only: OPENAI_API_KEY and OPENAI_BASE_URL
(unset -> provider default endpoint) for API scorers, HF_TOKEN / HUGGINGFACE_HUB_TOKEN
for gated Hugging Face models.

Example (from the repository root):
    from evaluation.evaluator import Evaluator
    ev = Evaluator(model="meta-llama/Llama-3.2-1B-Instruct", nli_model=None,
                   use_api=False, enable_clustering=False)
    out = ev.perplexity_evaluation(question, evidence_list, response, return_raw=True)
    out["shapley_values"], out["prob_distribution"]   # {evidence_index: value}

Batch usage over generated responses: python evaluation/batch_evaluate.py --help
"""

import argparse
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import os
import json
import importlib
import re as _re
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification
)
from openai import OpenAI
import asyncio
from bert_score import score as bertscore_fn
import re

class OpenAIHelper:
    """Thin OpenAI-compatible chat client used to split a response into atomic claims (auxiliary).

    api_key defaults to OPENAI_API_KEY and base_url to OPENAI_BASE_URL (None -> provider default).
    """

    def __init__(self, model: str, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.model = model

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")

        if self.api_key is None:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAIHelper.")
        self._ClientClass = OpenAI

    def _client(self):
        return self._ClientClass(api_key=self.api_key, base_url=self.base_url)

    def split_atomic_claims(self, text: str) -> List[str]:
        system = (
            "You are an information extraction assistant. "
            "Extract atomic claims (one fact per item) strictly and return only JSON."
        )
        user = (
            "Task: Extract atomic claims from the following text.\n"
            "Rules: Each claim should be a minimal, verifiable statement. Do not merge multiple facts.\n"
            "Return JSON object with a 'claims' array. No extra text.\n\n"
            f"Text:\n{text}"
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 1,
            "response_format": {"type": "json_object"},
        }
        client = self._client()
        try:
            resp = client.chat.completions.create(**payload)
            content = resp.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"OpenAI split_atomic_claims request failed: {e}")
        try:
            data = json.loads(content)
        except Exception:
            
            
            m = _re.search(r"\{[\s\S]*\}", content)
            if not m:
                raise RuntimeError("Failed to parse JSON for atomic claims.")
            data = json.loads(m.group(0))
        claims = data.get("claims", [])
        if not isinstance(claims, list):
            raise RuntimeError("JSON must contain a 'claims' array.")
        return [str(c).strip() for c in claims if isinstance(c, (str, int, float)) and str(c).strip()]


class APIGenerativeModel:
    """Scorer/generator backed by an OpenAI-compatible API.

    score() implements the Shapley value function through the legacy completions
    endpoint with echo=True and logprobs, which must be supported by the endpoint;
    generate() and encode() serve the auxiliary similarity/clustering paths.
    Credentials come from OPENAI_API_KEY / OPENAI_BASE_URL unless passed explicitly.
    """

    def __init__(self, model_name: str, api_key: Optional[str] = None, base_url: Optional[str] = None, embedding_model: str = "sfr-embedding-mistral"):
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")

        if self.api_key is None:
            raise RuntimeError("API key is required. Set OPENAI_API_KEY environment variable.")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def encode(self, texts: List[str]) -> List[List[float]]:
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            raise RuntimeError(f"API embeddings request failed: {e}")

    def generate(self, question: str, evidence_cluster: List[str], max_new_tokens: int = 256) -> str:

        content_dict = {f"evidence_{i+1}": e.strip() for i, e in enumerate(evidence_cluster) if isinstance(e, str) and e.strip()}

        prompt = f"""You are analyzing evidence to answer a question. Follow these rules strictly:

RULES:
1. Base your answer ONLY on the provided Content.
2. Answer in the exact format requested in the Question:
   - If asked yes/no: answer "yes" or "no"
   - If asked true/false/uncertain: answer "true", "false", or "uncertain"
   - For QA questions: provide a direct answer (e.g., a year, a name, a brief phrase)
   - For summarization: provide a concise summary (2-3 sentences max)
3. Do NOT include any sections labeled "reference", "references", "source", "sources", or any tags such as <reference>...</reference>.

QUESTION:
{question}

CONTENT:
{json.dumps(content_dict, ensure_ascii=False)}

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

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_new_tokens,
                temperature=0.0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"API generation request failed: {e}")

    def score(self, prompt: str, continuation: str) -> Dict[str, Any]:
        full_text = prompt + continuation

        try:
            
            response = self.client.completions.create(
                model=self.model_name,
                prompt=full_text,
                max_tokens=1,  
                echo=True,     
                logprobs=1,    
                temperature=1.0  
            )

            
            if not response.choices or not response.choices[0].logprobs:
                return {"token_logprobs": []}

            token_logprobs = response.choices[0].logprobs.token_logprobs
            tokens = response.choices[0].logprobs.tokens

            
            cont_start_idx = None

            
            for i in range(len(tokens)):
                reconstructed = ''.join(tokens[i:])
                
                
                if continuation.strip().startswith(reconstructed.strip()[:min(50, len(reconstructed))]):
                    cont_start_idx = i
                    break

            
            if cont_start_idx is None and continuation:
                cont_prefix = continuation.strip()[:10]
                for i in range(len(tokens)):
                    token_text = ''.join(tokens[i:i+3])  
                    if cont_prefix.startswith(token_text.strip()):
                        cont_start_idx = i
                        break

            
            if cont_start_idx is None:
                prompt_token_count = len(self._estimate_tokens(prompt))
                cont_start_idx = prompt_token_count
                print(f"[WARNING] Could not match continuation, using estimation at token {cont_start_idx}")

            
            cont_logprobs = token_logprobs[cont_start_idx:]
            cont_logprobs = [lp for lp in cont_logprobs if lp is not None]

            if not cont_logprobs:
                return {"token_logprobs": []}

            avg_nll = -sum(cont_logprobs) / len(cont_logprobs)
            return {"token_logprobs": cont_logprobs, "avg_nll": float(avg_nll)}

        except Exception as e:
            
            print(f"Warning: API score request failed: {e}")
            print("Your API might not support logprobs. Perplexity evaluation will not work.")
            return {"token_logprobs": []}

    def _estimate_tokens(self, text: str) -> List[str]:
        return text.split()

    def logprobs(self, prompt: str, continuation: str) -> List[float]:
        out = self.score(prompt, continuation)
        return out.get("token_logprobs", [])


class HFGenerativeModel:
    """Local Hugging Face causal LM used as the frozen scorer for the value function.

    score(prompt, continuation) returns the per-token log-probabilities of the
    continuation (the model response) given the prompt (question + evidence subset)
    and their average negative log-likelihood, which is v(S) up to sign. This is the
    path used for all released runs (default scorer meta-llama/Llama-3.2-1B-Instruct).
    Gated models are downloaded with HF_TOKEN / HUGGINGFACE_HUB_TOKEN if set.
    """

    def __init__(self, model_name: str, device: Optional[str] = None, use_multi_gpu: bool = True):
        """
        Initialize HuggingFace generative model.

        Args:
            model_name: HuggingFace model path
            device: Specific device (e.g., "cuda:0"). If None and use_multi_gpu=True, auto-distribute across GPUs
            use_multi_gpu: If True and device is None, use device_map="auto" for multi-GPU support
        """
        self.model_name = model_name
        
        hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, token=hf_token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        
        if device is None and use_multi_gpu and torch.cuda.is_available():
            print(f"[HFGenerativeModel] Loading {model_name} with multi-GPU support (device_map='auto')")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                dtype="auto",
                token=hf_token
            )
            self.device = "cuda"  
            self.multi_gpu = True
        else:
            
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            print(f"[HFGenerativeModel] Loading {model_name} on single device: {self.device}")
            self.model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token)
            self.model.to(self.device)
            self.multi_gpu = False

        self.model.eval()

        # Determine a primary device for inputs when model is sharded across GPUs.
        # Inputs (input_ids/attention_mask) must reside on the same device as the embedding layer.
        self.primary_device: torch.device
        if self.multi_gpu:
            self.primary_device = self._resolve_primary_device_for_embeddings()
        else:
            self.primary_device = torch.device(self.device)

    def _resolve_primary_device_for_embeddings(self) -> torch.device:
        # Try to infer device from hf_device_map (available when device_map='auto')
        device_map = getattr(self.model, 'hf_device_map', None)
        if isinstance(device_map, dict) and device_map:
            # Prefer keys that contain 'embed' or common embedding names
            preferred_keys = [
                'embed_tokens',
                'wte',
                'model.embed_tokens',
                'model.model.embed_tokens',
            ]
            # First pass: exact or substring match on preferred keys
            for key in list(device_map.keys()):
                for pref in preferred_keys:
                    if pref in key:
                        dev = device_map[key]
                        try:
                            return torch.device(dev)
                        except Exception:
                            pass
            # Fallback: choose the first device in the map
            try:
                any_dev = next(iter(device_map.values()))
                return torch.device(any_dev)
            except Exception:
                pass
        # Fallback: use device of the first parameter we can find
        try:
            first_param = next(self.model.parameters())
            return first_param.device
        except Exception:
            # Last resort: cuda:0 if available else cpu
            return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def encode(self, texts: List[str]) -> List[List[float]]:
        with torch.no_grad():
            batch = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
            # Ensure inputs are placed on the proper device
            target_device = self.primary_device if self.multi_gpu else torch.device(self.device)
            batch = {k: v.to(target_device) for k, v in batch.items()}
            outputs = self.model(**batch, output_hidden_states=True)
            hs = outputs.hidden_states  
            hidden = (hs[-1] + hs[-2]) / 2  
            mask = batch.get('attention_mask', torch.ones_like(hidden[:, :, 0]))  
            mask = mask.unsqueeze(-1)  
            summed = (hidden * mask).sum(dim=1)  
            counts = mask.sum(dim=1).clamp(min=1)  
            pooled = summed / counts
            vecs = pooled.detach().cpu().tolist()
        return vecs

    def generate(self, question: str, evidence_cluster: List[str], max_new_tokens: int = 256) -> str:

        content_dict = {f"evidence_{i+1}": e.strip() for i, e in enumerate(evidence_cluster) if isinstance(e, str) and e.strip()}

        prompt = f"""You are analyzing evidence to answer a question. Follow these rules strictly:

RULES:
1. Base your answer ONLY on the provided Content.
2. Answer in the exact format requested in the Question:
   - If asked yes/no: answer "yes" or "no"
   - If asked true/false/uncertain: answer "true", "false", or "uncertain"
   - For QA questions: provide a direct answer (e.g., a year, a name, a brief phrase)
   - For summarization: provide a concise summary (2-3 sentences max)
3. Do NOT include any sections labeled "reference", "references", "source", "sources", or any tags such as <reference>...</reference>.

QUESTION:
{question}

CONTENT:
{json.dumps(content_dict, ensure_ascii=False)}

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

        inputs = self.tokenizer(prompt, return_tensors='pt')
        # Ensure inputs are placed on the proper device
        target_device = self.primary_device if self.multi_gpu else torch.device(self.device)
        inputs = inputs.to(target_device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        generated_ids = output_ids[0][inputs['input_ids'].shape[1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def score(self, prompt: str, continuation: str) -> Dict[str, Any]:
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        cont_ids = self.tokenizer(continuation, add_special_tokens=False).input_ids
        if len(cont_ids) == 0:
            return {"token_logprobs": []}

        
        # Place inputs on the embedding device to avoid device mismatch when sharded
        target_device = self.primary_device if self.multi_gpu else torch.device(self.device)
        input_ids = torch.tensor([prompt_ids + cont_ids], dtype=torch.long, device=target_device)
        attn_mask = torch.ones_like(input_ids, device=target_device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attn_mask)
            logits = outputs.logits  
            logprobs_all = torch.log_softmax(logits, dim=-1)
        start = len(prompt_ids)
        L = input_ids.shape[1]
        logs: List[float] = []
        for pos in range(start, L):
            if pos == 0:
                continue
            target_id = input_ids[0, pos].item()
            lp = float(logprobs_all[0, pos - 1, target_id].detach().cpu().item())
            logs.append(lp)
        return {"token_logprobs": logs, "avg_nll": float(-sum(logs) / max(1, len(logs)))}

    def logprobs(self, prompt: str, continuation: str) -> List[float]:
        out = self.score(prompt, continuation)
        return out.get("token_logprobs", [])


class HFNLIModel:
    """MNLI sequence classifier (default roberta-large-mnli) for the auxiliary claim checks.

    nli(premise, hypothesis) returns the arg-max label and a probability dict with the
    keys 'entail', 'neutral' and 'contradict'.
    """

    def __init__(self, model_name: str, device: Optional[str] = None):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        id2label = getattr(self.model.config, 'id2label', {}) or {}
        self.label_map = {int(k): v.lower() for k, v in id2label.items()}

    def nli(self, premise: str, hypothesis: str) -> Tuple[str, Dict[str, float]]:
        with torch.no_grad():
            batch = self.tokenizer(
                premise, hypothesis,
                return_tensors='pt', truncation=True, padding=True
            ).to(self.device)
            logits = self.model(**batch).logits
            probs = torch.softmax(logits, dim=-1)[0].detach().cpu().tolist()

        
        id2label = getattr(self.model.config, 'id2label', {}) or {}
        label_map = {int(k): str(v).lower() for k, v in id2label.items()}

        idx_to_key = {}
        for idx, name in label_map.items():
            if 'entail' in name:
                idx_to_key[idx] = 'entail'
            elif 'contradict' in name:
                idx_to_key[idx] = 'contradict'
            elif 'neutral' in name:
                idx_to_key[idx] = 'neutral'

        if len(idx_to_key) < 3:
            raise RuntimeError("Model does not have standard MNLI labels (entail/neutral/contradict).")

        dist = {idx_to_key.get(i, f'label_{i}'): float(p) for i, p in enumerate(probs)}
        label = max(['entail', 'neutral', 'contradict'], key=lambda k: dist.get(k, 0.0))

        return label, dist


class Evaluator:
    """Response-level evaluator: Shapley evidence attribution plus auxiliary NLI-based checks.

    Args:
        model: scorer for the value function. A Hugging Face model id (use_api=False,
            wrapped in HFGenerativeModel), an API model name (use_api=True, wrapped in
            APIGenerativeModel), an object exposing score()/logprobs(), or None.
        nli_model: MNLI classifier id for the atomic-claim / FActScore-style checks
            (default roberta-large-mnli); None disables them.
        helper_model: optional API model used by OpenAIHelper.split_atomic_claims.
        enable_clustering: group evidence with k-means before attribution. Disabled in
            all released runs, so each evidence piece is one Shapley player.
        use_api: choose APIGenerativeModel (True) or HFGenerativeModel (False) for `model`.
        api_key / helper_api_key / base_url: override OPENAI_API_KEY / OPENAI_BASE_URL.
        embedding_model: embedding model name for the API similarity/clustering path.

    The paper metrics are implemented by perplexity (value function) and
    perplexity_evaluation (Shapley values and contribution shares).
    """

    def __init__(
        self,
        model: str = "gpt-5",
        nli_model: Optional[str] = 'roberta-large-mnli',
        helper_model: Optional[str] = None,
        enable_clustering: bool = False,
        use_api: bool = True,
        api_key: Optional[str] = None,
        helper_api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        embedding_model: str = "sfr-embedding-mistral"
    ):
        
        if model is not None:
            if isinstance(model, str):
                if use_api:
                    print(f"[Evaluator] Initializing API model: {model}")
                    self.model = APIGenerativeModel(
                        model_name=model,
                        api_key=api_key,
                        base_url=base_url,
                        embedding_model=embedding_model
                    )
                else:
                    print(f"[Evaluator] Initializing local HuggingFace model: {model}")
                    self.model = HFGenerativeModel(model_name=model)
            else:
                
                self.model = model
        else:
            print("[Evaluator] No main model - only atomic claim evaluation available")
            self.model = None

        
        if nli_model is not None:
            if isinstance(nli_model, str):
                print(f"[Evaluator] Initializing local NLI model: {nli_model}")
                self.nli_model = HFNLIModel(nli_model)
            else:
                self.nli_model = nli_model
        else:
            print("[Evaluator] No NLI model - atomic claim support/undermine check will not work")
            self.nli_model = None

        
        if helper_model is not None:
            if isinstance(helper_model, str):
                print(f"[Evaluator] Initializing helper model (API): {helper_model}")
                
                effective_helper_key = helper_api_key if helper_api_key is not None else api_key
                self.helper_model = OpenAIHelper(
                    model=helper_model,
                    api_key=effective_helper_key,
                    base_url=base_url
                )
            else:
                self.helper_model = helper_model
        else:
            self.helper_model = None

        
        self.nli_support_threshold = 0.5 
        self.nli_contradict_threshold = 0.5  
        self.nli_margin = 0.1 
        self.enable_clustering = enable_clustering
        self.use_api = use_api

        print(f"[Evaluator] Configuration:")
        print(f"  - Main model: {'API' if use_api else 'Local'} ({model if model else 'None'})")
        print(f"  - NLI model: Local ({nli_model if nli_model else 'None'})")
        print(f"  - Helper model: API ({helper_model if helper_model else 'None'})")
        print(f"  - Clustering: {'Enabled' if enable_clustering else 'Disabled'}")
    
    def evaluate(self, type: str, question: str, evidence:list[str], response:str) -> dict:

        if type == "similarity":
            print("[Evaluate] similarity started")

            return self.similarity_evaluation(question, evidence, response)
        elif type == "perplexity":
            print("[Evaluate] perplexity started")

            return self.perplexity_evaluation(question, evidence, response)
        elif type == "atomic_claim":
            print("[Evaluate] atomic_claim started")

            return self.atomic_claim_evaluation(evidence, response)
        elif type == "faithfulness":
            print("[Evaluate] faithfulness started")

            return self.faithfulness_evaluation(evidence, response)
        else:
            raise ValueError(f"Invalid evaluation type: {type}")



    def cluster(self, evidence: list[str]) -> list[list[str]]:
        cleaned: List[Tuple[int, str]] = []
        seen = set()
        for idx, ev in enumerate(evidence or []):
            if not isinstance(ev, str):
                continue
            s = ev.strip()
            if not s:
                continue
            key = s
            if key in seen:
                continue
            seen.add(key)
            cleaned.append((idx, s))
        if not cleaned:
            print("[Cluster] no evidence after cleaning")
            return []

        texts = [s for _, s in cleaned]
        print(f"[Cluster] {len(texts)} unique evidence items")

        
        if len(texts) <= 2:
            print(f"[Cluster] only {len(texts)} item(s) -> skipping clustering, returning as-is")
            return [[s] for s in texts]

        
        if not self.enable_clustering:
            print("[Cluster] clustering disabled -> each evidence as separate cluster")
            return [[s] for s in texts]

        
        if self.model is None:
            print("[Cluster] no model available -> each evidence is its own cluster")
            return [[s] for s in texts]

        vectors = self._get_embeddings(texts)  
        n = len(vectors)

        
        k = 2
        print(f"[Cluster] clustering into k={k} groups")

        
        assignments_for_best, centroids_for_best = self._kmeans(vectors, k, max_iter=50, seed=42)
        
        cluster_map: Dict[int, List[Tuple[int, str]]] = {}
        for (orig_idx, s), label in zip(cleaned, assignments_for_best):
            cluster_map.setdefault(label, []).append((orig_idx, s))

        
        clusters_indexed: List[Tuple[int, List[str]]] = []
        for label, items in cluster_map.items():
            items.sort(key=lambda x: x[0])
            clusters_indexed.append((min(i for i, _ in items), [s for _, s in items]))
        clusters_indexed.sort(key=lambda x: (-len(x[1]), x[0]))

        return [c for _, c in clusters_indexed]
    
    def similarity(self,response_with_partial_evidence:str, response: str) -> float:
        a = (response_with_partial_evidence or "").strip()
        b = (response or "").strip()
        if not a or not b:
            return 0.0
        vecs = self._get_embeddings([a, b])
        return max(0.0, min(1.0, self._cosine(vecs[0], vecs[1])))

    def perplexity(self, question:str, evidence_cluster: list[str], response: str, return_avg_nll: bool = False) -> float:
        """
        Compute perplexity or average NLL for a response given evidence.
        
        Args:
            question: The question being answered
            evidence_cluster: List of evidence texts
            response: The model response to evaluate
            return_avg_nll: If True, return avg_nll directly (for Shapley value function)
                           If False, return PPL = exp(avg_nll)
        
        Returns:
            PPL (default) or avg_nll (if return_avg_nll=True)
        """
        response = (response or "").strip()
        content_dict = {f"evidence_{i+1}": e.strip() for i, e in enumerate(evidence_cluster) if isinstance(e, str) and e.strip()}

        prompt = f"""You are analyzing evidence to answer a question. Follow these rules strictly:

RULES:
1. Base your answer ONLY on the provided Content.
2. Answer in the exact format requested in the Question:
   - If asked yes/no: answer "yes" or "no"
   - If asked true/false/uncertain: answer "true", "false", or "uncertain"
   - For QA questions: provide a direct answer (e.g., a year, a name, a brief phrase)
   - For summarization: provide a concise summary (2-3 sentences max)
3. Do NOT include any sections labeled "reference", "references", "source", "sources", or any tags such as <reference>...</reference>.

QUESTION:
{question}

CONTENT:
{json.dumps(content_dict, ensure_ascii=False)}

REQUIRED RESPONSE FORMAT:

<answer>[your direct answer or summary]</answer>

[Your detailed explanation with source citations, max 200 words]

IMPORTANT:
- Put ONLY your direct answer/summary inside <answer></answer> tags.
- Do NOT output any additional sections like "reference:" or "<reference>".
- For summarization: keep the summary concise (2-3 sentences max).
- NO additional explanations inside the <answer> tags.
- After </answer> tag: provide reasoning and evidence citations (max 200 words), written naturally in text form (not as a reference list).

YOUR RESPONSE:
"""

        avg_nll = self._score_with_model(prompt, response)
        if avg_nll is None:
            raise RuntimeError("self.model does not provide a compatible scoring API for perplexity. Please supply an adapter with `score` or `logprobs`.")
        
        # Return avg_nll directly for Shapley value function (length-normalized)
        if return_avg_nll:
            return avg_nll
        
        try:
            ppl = math.exp(avg_nll)
        except OverflowError:
            ppl = float('inf')
        return ppl

    def response_2_atomic_claim(self, response: str) -> list[str]:
        """
        Extract atomic facts from a response.

        In the ideal implementation (as in the official FActScore), this should use
        a specialized model (e.g., InstructGPT or ChatGPT) to decompose the response
        into atomic facts. However, for simplicity and cost efficiency, this
        implementation uses sentence-level splitting as an approximation.

        For better results, consider:
        1. Using an LLM API (GPT-4, Claude, etc.) to extract atomic facts
        2. Using the official FActScore package with their atomic fact extractor
        3. Training a specialized model for atomic fact extraction

        Args:
            response: The text response to decompose

        Returns:
            List of atomic facts (currently sentences as approximation)
        """
        text = (response or "").strip()
        if not text:
            return []

        # TODO: Replace with proper atomic fact extraction
        # For now, use sentence splitting as a simple approximation
        # Note: This is NOT true atomic fact extraction!
        # A sentence like "Einstein was born in 1879 and won Nobel Prize in 1921"
        # should be split into TWO atomic facts, but this won't do that.

        claims = self._split_into_sentences(text)

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for c in claims:
            c_clean = c.strip()
            if c_clean and c_clean not in seen:
                seen.add(c_clean)
                unique.append(c_clean)

        return unique


    def check_atomic_claim_support(self, atomic_claims:list[str],evidence_cluster: list[str]) -> bool:
        """
        Check if any atomic claim is supported by the evidence cluster using NLI.

        Args:
            atomic_claims: List of atomic claims to check
            evidence_cluster: List of evidence texts

        Returns:
            True if any claim is supported by any evidence, False otherwise
        """
        if not atomic_claims or not evidence_cluster:
            return False

        for claim in atomic_claims if isinstance(atomic_claims, list) else [atomic_claims]:
            for ev in evidence_cluster:
                # Split evidence into sentences for finer-grained matching
                sentences = self._split_into_sentences(ev)

                for sent in sentences:
                    if not sent.strip():
                        continue

                    # Use NLI to check if evidence sentence entails the claim
                    label, probs = self._nli(sent, claim)
                    entail = probs.get('entail', 0.0)
                    contradict = probs.get('contradict', 0.0)

                    # Check if entailment is strong enough and not contradicted
                    if entail >= self.nli_support_threshold and (entail - contradict) >= self.nli_margin:
                        return True

        return False


    def check_atomic_claim_undermine(self, atomic_claims:list[str],evidence_cluster: list[str]) -> bool:
        """
        Check if any atomic claim is contradicted/undermined by the evidence cluster using NLI.

        Args:
            atomic_claims: List of atomic claims to check
            evidence_cluster: List of evidence texts

        Returns:
            True if any claim is contradicted by any evidence, False otherwise
        """
        if not atomic_claims or not evidence_cluster:
            return False

        for claim in atomic_claims if isinstance(atomic_claims, list) else [atomic_claims]:
            for ev in evidence_cluster:
                # Split evidence into sentences for finer-grained matching
                sentences = self._split_into_sentences(ev)

                for sent in sentences:
                    if not sent.strip():
                        continue

                    # Use NLI to check if evidence sentence contradicts the claim
                    label, probs = self._nli(sent, claim)
                    contradict = probs.get('contradict', 0.0)

                    # Check if contradiction is strong enough
                    if contradict >= self.nli_contradict_threshold:
                        return True

        return False



    def similarity_evaluation(self, question:str, evidence:list[str], response:str) -> dict:
        print(f"[Similarity] start: evidence={len(evidence)} items")
        clusters = self.cluster(evidence)
        print(f"[Similarity] formed {len(clusters)} clusters")
        raw_results: Dict[int, float] = {}
        for i, cluster in enumerate(clusters):
            print(f"[Similarity] cluster {i+1}/{len(clusters)}: size={len(cluster)} -> generating response")
            response_with_partial_evidence = self.model.generate(question, cluster)
            sim = self.similarity(response_with_partial_evidence, response)
            raw_results[i] = sim
            print(f"[Similarity] cluster {i+1}: similarity={sim:.4f}")
        
        
        normalized_results = self._normalize_to_percentage(raw_results, mode="absolute")
        for i, pct in normalized_results.items():
            print(f"[Similarity] cluster {i+1}: percentage={pct*100:.2f}%")
        return normalized_results

    def perplexity_evaluation(self, question:str, evidence:list[str], response:str, return_raw:bool=False) -> dict:
        """
        Compute Shapley values for evidence clusters using length-normalized log-likelihood.

        This is the Evidence Balance attribution of the paper (Section 3.1). With
        clustering disabled (all released runs) every evidence piece is one player, so
        n_clusters equals the number of evidence pieces and the Shapley values are exact:
        all 2^n subsets are scored once (subset values are cached) and combined with the
        standard Shapley weights |S|! (n-|S|-1)! / n!.

        Value function (from paper):
            v(S) = (1/|R|) * sum_{t=1}^{|R|} log p(r_t | r_{<t}, E_S)
                 = -avg_nll  (average log-likelihood, length-normalized)

        This normalization removes the effect of response length on the value function.

        Post-processing: negative Shapley values are clipped at zero (ReLU) and the
        remaining mass is normalized to contribution shares p_i (sum = 1). If every
        value is <= 0 the shares fall back to uniform. The Balance score (Gini
        coefficient of p, lower = more balanced) is computed from 'prob_distribution' by
        evaluation/balance_score.py.

        Returns:
            prob_distribution {index: share}; with return_raw=True a dict with
            'shapley_values', 'prob_distribution', 'n_positive' and 'n_negative'.
        """
        import itertools

        print(f"[Perplexity] start: evidence={len(evidence)} items (Shapley method)")
        print(f"[Perplexity] Using length-normalized log-likelihood as value function")
        clusters = self.cluster(evidence)
        n_clusters = len(clusters)
        print(f"[Perplexity] formed {n_clusters} clusters")

        # Cache for value function v(S) = -avg_nll (length-normalized log-likelihood)
        value_cache = {}
        
        # Compute baseline: avg_nll with no evidence (empty context)
        # This gives us a proper reference point for marginal contributions
        baseline_avg_nll = self.perplexity(question, [], response, return_avg_nll=True)
        baseline_value = -baseline_avg_nll
        print(f"[Perplexity] Baseline (no evidence): avg_nll={baseline_avg_nll:.4f}, value={baseline_value:.4f}")

        def get_value(subset_indices):
            """
            Get value function v(S) = -avg_nll for a subset of clusters.
            
            v(S) = (1/|R|) * sum log p(r_t | r_{<t}, E_S) = -avg_nll
            
            Higher value = better fit (more likely response given evidence)
            """
            key = tuple(sorted(subset_indices))
            if key not in value_cache:
                if len(key) == 0:
                    # No evidence -> use precomputed baseline
                    value_cache[key] = baseline_value
                else:
                    subset_evidence = [e for i in key for e in clusters[i]]
                    # Get avg_nll directly (length-normalized)
                    avg_nll = self.perplexity(question, subset_evidence, response, return_avg_nll=True)
                    # v(S) = -avg_nll = mean(log_probs)
                    value_cache[key] = -avg_nll
            return value_cache[key]

        # Compute Shapley values
        shapley_values = {}
        total_subsets = 2 ** n_clusters
        print(f"[Perplexity] Computing Shapley values for {n_clusters} clusters ({total_subsets} subsets)...")

        for i in range(n_clusters):
            shapley_value = 0.0
            
            all_indices = set(range(n_clusters))
            subsets_without_i = []
            for r in range(n_clusters):
                for subset in itertools.combinations(all_indices - {i}, r):
                    subsets_without_i.append(subset)

            for S in subsets_without_i:
                S_list = list(S)
                S_with_i = S_list + [i]

                # Get length-normalized values
                value_S = get_value(S_list)
                value_S_with_i = get_value(S_with_i)

                # Marginal contribution: how much does adding cluster i improve the value?
                # Positive marginal = adding evidence i improves model fit
                # Negative marginal = adding evidence i hurts model fit
                marginal = value_S_with_i - value_S

                # Shapley weight
                s = len(S)
                weight = math.factorial(s) * math.factorial(n_clusters - s - 1) / math.factorial(n_clusters)

                shapley_value += weight * marginal

            shapley_values[i] = shapley_value
            print(f"[Perplexity] cluster {i+1}/{n_clusters}: shapley_value={shapley_value:.6f}")

        # Convert to valid probability distribution (non-negative, sum=1.0)
        # 
        # Key insight for "treat all evidence equally":
        # - Positive Shapley = evidence SUPPORTS the response (effective contribution)
        # - Negative Shapley = evidence was IGNORED or CONFLICTS with response (no effective contribution)
        # - Zero/Negative should NOT count as "contribution" for fairness evaluation
        #
        # Method: ReLU + normalize (only positive contributions count)
        # - This properly measures "effective contribution" from each evidence
        # - If model ignores evidence (negative Shapley), its contribution = 0%
        # - Ideal "fair" model: all Shapley positive and equal -> uniform distribution
        
        # ReLU: keep positive, zero out negative
        positive_only = {k: max(0.0, v) for k, v in shapley_values.items()}
        total_positive = sum(positive_only.values())
        
        # Normalize to probability distribution
        if total_positive > 0:
            prob_distribution = {k: v / total_positive for k, v in positive_only.items()}
        else:
            # All negative = model rejected all evidence, use uniform as fallback
            prob_distribution = {k: 1.0 / n_clusters for k in shapley_values.keys()}

        # Count positive/negative contributions
        n_positive = sum(1 for v in shapley_values.values() if v > 0)
        n_negative = sum(1 for v in shapley_values.values() if v < 0)
        n_zero = n_clusters - n_positive - n_negative

        print(f"[Perplexity] Shapley results (ReLU normalization - only positive counts):")
        for i in range(n_clusters):
            raw_val = shapley_values[i]
            prob = prob_distribution[i]
            status = "+" if raw_val > 0 else ("-" if raw_val < 0 else "0")

            print(f"  cluster {i+1}: shapley={raw_val:+.6f} {status}, contribution={prob*100:.2f}%")

        # Uniform distribution for reference
        uniform_prob = 1.0 / n_clusters
        print(f"[Perplexity] Uniform distribution: {uniform_prob*100:.2f}% each")
        print(f"[Perplexity] Evidence usage: {n_positive}/{n_clusters} used (+), {n_negative}/{n_clusters} ignored/rejected (-)")

        if return_raw:
            return {
                'shapley_values': shapley_values,  # raw Shapley values (can be negative)
                'prob_distribution': prob_distribution,  # valid probability distribution (only positive counts)
                'n_positive': n_positive,  # count of evidence that contributed
                'n_negative': n_negative,  # count of evidence that was ignored/rejected
            }
        return prob_distribution

    def atomic_claim_evaluation(self, evidence:list[str], response:str) -> dict:
        print(f"[AtomicClaim] start: evidence={len(evidence)} items -> extracting claims")
        clusters = self.cluster(evidence)
        claims = self.response_2_atomic_claim(response)
        print(f"[AtomicClaim] claims extracted: {len(claims)}; clusters: {len(clusters)}")
        raw_counts: Dict[int, Dict[str, int]] = {}
        for i, cluster in enumerate(clusters):
            count_support = 0
            count_undermine = 0
            for claim in claims:
                if self.check_atomic_claim_support(claim, cluster):
                    count_support += 1
                elif self.check_atomic_claim_undermine(claim, cluster):
                    count_undermine += 1
            raw_counts[i] = {"support": count_support, "undermine": count_undermine}
            print(f"[AtomicClaim] cluster {i+1}/{len(clusters)}: support={count_support}, undermine={count_undermine}")

        
        net_contribution_dict = {}
        for i, counts in raw_counts.items():
            support = counts["support"]
            undermine = counts["undermine"]
            net_contribution_dict[i] = support - undermine  

        
        
        total_abs = sum(abs(v) for v in net_contribution_dict.values())
        if total_abs == 0:
            net_percentage = {i: 0.0 for i in raw_counts.keys()}
            net_percentage_abs = {i: 0.0 for i in raw_counts.keys()}
        else:
            
            net_percentage = {i: (v / total_abs) * 100.0 for i, v in net_contribution_dict.items()}

            
            net_percentage_abs = {i: (abs(v) / total_abs) * 100.0 for i, v in net_contribution_dict.items()}

        
        support_dict = {i: counts["support"] for i, counts in raw_counts.items()}
        undermine_dict = {i: counts["undermine"] for i, counts in raw_counts.items()}
        support_percentage = self._normalize_to_percentage(support_dict, mode="absolute")
        undermine_percentage = self._normalize_to_percentage(undermine_dict, mode="absolute")

        contribution: Dict[int, Dict[str, float]] = {}
        for i in raw_counts.keys():
            contribution[i] = {
                "support": support_percentage.get(i, 0.0),
                "undermine": undermine_percentage.get(i, 0.0),
                "net_contribution": net_percentage.get(i, 0.0),  
                "net_contribution_abs": net_percentage_abs.get(i, 0.0)  
            }
            support_count = raw_counts[i]["support"]
            undermine_count = raw_counts[i]["undermine"]
            net = net_contribution_dict[i]
            net_pct = net_percentage.get(i, 0.0)
            net_pct_abs = net_percentage_abs.get(i, 0.0)
            print(f"[AtomicClaim] cluster {i+1}: support={support_count}, undermine={undermine_count}, net={net:+d}, net_contribution={net_pct:+.2f}%, abs={net_pct_abs:.2f}%")

        total_support = sum(raw_counts[i]["support"] for i in raw_counts)
        total_undermine = sum(raw_counts[i]["undermine"] for i in raw_counts)
        total_net = total_support - total_undermine
        print(f"[AtomicClaim] totals: support={total_support}, undermine={total_undermine}, net={total_net:+d}")
        return contribution

    def faithfulness_evaluation(self, evidence: list[str], response: str) -> float:
        """
        FActScore: Fine-grained Atomic Evaluation of Factual Precision.

        Computes the proportion of atomic facts in the response that are supported by evidence.

        FActScore = (# supported atomic facts) / (# total atomic facts)

        Based on: Min et al., "FActScore: Fine-grained Atomic Evaluation of Factual
        Precision in Long Form Text Generation", EMNLP 2023.

        Args:
            evidence: List of evidence texts
            response: Model-generated response to evaluate

        Returns:
            FActScore between 0.0 and 1.0
        """
        print(f"[Faithfulness/FActScore] Evaluation started")

        # Step 1: Extract atomic facts from the response
        atomic_facts = self.response_2_atomic_claim(response)

        if not atomic_facts:
            print(f"[Faithfulness/FActScore] No atomic facts extracted -> score=0.0")
            return 0.0

        print(f"[Faithfulness/FActScore] Extracted {len(atomic_facts)} atomic facts")

        # Step 2: Check each atomic fact against all evidence
        # Note: We don't use clustering here to stay closer to the original FActScore
        # which checks each fact against the entire knowledge base
        supported_facts = 0
        not_supported_facts = 0

        for fact_idx, fact in enumerate(atomic_facts):
            is_supported = False

            # Check if any evidence supports this fact
            for ev_idx, ev in enumerate(evidence):
                # Check support using NLI
                if self.check_atomic_claim_support([fact], [ev]):
                    is_supported = True
                    print(f"[Faithfulness/FActScore] Fact {fact_idx+1}/{len(atomic_facts)}: "
                          f"SUPPORTED by evidence {ev_idx+1}")
                    break

            if is_supported:
                supported_facts += 1
            else:
                not_supported_facts += 1
                print(f"[Faithfulness/FActScore] Fact {fact_idx+1}/{len(atomic_facts)}: "
                      f"NOT SUPPORTED")

        # Step 3: Calculate FActScore
        factscore = supported_facts / len(atomic_facts)

        print(f"[Faithfulness/FActScore] Results:")
        print(f"  - Total atomic facts: {len(atomic_facts)}")
        print(f"  - Supported facts: {supported_facts}")
        print(f"  - Not supported facts: {not_supported_facts}")
        print(f"  - FActScore: {factscore:.4f}")

        return factscore

        

    def _compute_bertscore_recall(self, evidence: list[str], response: str) -> Optional[float]:

        try:
            print("[Faithfulness] Computing BERTScore-Recall...")

            
            evidence_text = " ".join(ev.strip() for ev in evidence if isinstance(ev, str) and ev.strip())

            if not evidence_text:
                print("[Faithfulness] no evidence text -> BERTScore-R = None")
                return None

            
            
            P, R, F1 = bertscore_fn(
                [response],
                [evidence_text],
                lang="en",
                rescale_with_baseline=True,
                verbose=False
            )

            recall = R.item()
            print(f"[Faithfulness] BERTScore-Recall: {recall:.4f}")
            return recall

        except ImportError:
            print("[Faithfulness] bert_score not installed -> skipping BERTScore-R")
            print("[Faithfulness] Install with: pip install bert-score")
            return None
        except Exception as e:
            print(f"[Faithfulness] BERTScore computation failed: {e}")
            return None

    
    def _normalize_to_percentage(self, values: Dict[int, float], mode: str = "softmax") -> Dict[int, float]:
        """
        Convert Shapley values to a valid probability distribution (non-negative, sum=1.0).

        This is essential for computing JS divergence with uniform distribution.

        Args:
            values: Dictionary mapping indices to Shapley values
            mode: Normalization mode
                - "softmax": exp(v_i) / sum exp(v_j) - smooth, differentiable (recommended)
                - "shifted": (v_i - min) / sum (v_j - min) - linear, preserves relative order
                - "absolute": |v_i| / sum |v_j| - uses absolute values
                - "signed": v_i / sum |v_j| - preserves sign (NOT a valid probability distribution!)

        Returns:
            Dictionary mapping indices to probabilities (all non-negative, sum=1.0)
        """
        if not values:
            return {}

        n = len(values)
        uniform = 1.0 / n

        if mode == "softmax":
            # Softmax: exp(v_i) / sum exp(v_j)
            # - All positive, sum = 1.0
            # - Smooth transformation, commonly used in ML
            # - Negative values -> small probabilities (evidence was ignored/conflicting)
            # - Positive values -> larger probabilities (evidence contributed)
            max_val = max(values.values())  # for numerical stability
            exp_values = {k: math.exp(v - max_val) for k, v in values.items()}
            total = sum(exp_values.values())
            if total == 0.0:
                return {k: uniform for k in values.keys()}
            return {k: v / total for k, v in exp_values.items()}

        elif mode == "shifted":
            # Shift-then-normalize: (v_i - min) / sum (v_j - min)
            # - All non-negative, sum = 1.0
            # - Preserves relative ordering
            # - Most negative value becomes 0 (no contribution)
            min_val = min(values.values())
            shifted = {k: v - min_val for k, v in values.items()}
            total = sum(shifted.values())
            if total == 0.0:
                return {k: uniform for k in values.keys()}
            return {k: v / total for k, v in shifted.items()}

        elif mode == "absolute":
            # Absolute value: |v_i| / sum |v_j|
            # - All positive, sum = 1.0
            # - Treats positive and negative contributions as equal "influence"
            total_abs = sum(abs(v) for v in values.values())
            if total_abs == 0.0:
                return {k: uniform for k in values.keys()}
            return {k: abs(v) / total_abs for k, v in values.items()}

        else:  # "signed"
            # Signed: v_i / sum |v_j|
            # WARNING: NOT a valid probability distribution! Can have negative values.
            total_abs = sum(abs(v) for v in values.values())
            if total_abs == 0.0:
                return {k: uniform for k in values.keys()}
            return {k: v / total_abs for k, v in values.items()}

    def _normalize_to_percentage_inverse(self, values: Dict[int, float]) -> Dict[int, float]:

        if not values:
            return {}
        
        inverse_values = {}
        for k, v in values.items():
            if v == float('inf') or v <= 0.0:
                inverse_values[k] = 0.0
            else:
                inverse_values[k] = 1.0 / v
        
        return self._normalize_to_percentage(inverse_values, mode="absolute")

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:

        if self.model is None:
            raise RuntimeError("model is required to compute embeddings.")
        try:
            vecs = self.model.encode(texts)
        except Exception as e:
            raise RuntimeError(f"model.encode failed: {e}")
        return [self._l2_normalize(self._to_list(v)) for v in vecs]


    def _to_list(self, v: Any) -> List[float]:
        
        if isinstance(v, np.ndarray):
            return v.astype(float).tolist()

        if isinstance(v, (list, tuple)):
            return [float(x) for x in v]
        
        return [float(v)] if v is not None else [0.0]

    def _l2_normalize(self, vec: List[float]) -> List[float]:
        s = math.sqrt(sum(x * x for x in vec))
        if s == 0.0:
            return vec
        return [x / s for x in vec]

    def _cosine(self, a: List[float], b: List[float]) -> float:

        
        return max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b))))

    def _kmeans(self, vectors: List[List[float]], k: int, max_iter: int = 50, seed: int = 42) -> Tuple[List[int], List[List[float]]]:

        random.seed(seed)
        n = len(vectors)
        if k <= 1 or k >= n:
            labels = [0] * n
            centroids = [self._mean_vector(vectors)]
            return labels, centroids
        
        indices = list(range(n))
        random.shuffle(indices)
        centroids = [vectors[i][:] for i in indices[:k]]
        labels = [0] * n
        for _ in range(max_iter):
            
            changed = False
            for i, v in enumerate(vectors):
                
                best_j = 0
                best_sim = -2.0
                for j, c in enumerate(centroids):
                    sim = self._cosine(v, c)
                    if sim > best_sim:
                        best_sim = sim
                        best_j = j
                if labels[i] != best_j:
                    labels[i] = best_j
                    changed = True
            
            new_centroids: List[List[float]] = []
            for j in range(k):
                members = [vectors[i] for i in range(n) if labels[i] == j]
                if not members:
                    
                    idx = random.randrange(n)
                    new_centroids.append(vectors[idx][:])
                else:
                    new_centroids.append(self._l2_normalize(self._mean_vector(members)))
            centroids = new_centroids
            if not changed:
                break
        return labels, centroids

    def _mean_vector(self, vectors: List[List[float]]) -> List[float]:

        if not vectors:
            return []
        dim = len(vectors[0])
        acc = [0.0] * dim
        for v in vectors:
            for i, x in enumerate(v):
                acc[i] += x
        n = float(len(vectors))
        return [x / n for x in acc]

    def _silhouette_score(self, vectors: List[List[float]], labels: List[int], centroids: List[List[float]]) -> float:
        
        n = len(vectors)
        if n <= 2:
            return 0.0
        
        label_to_indices: Dict[int, List[int]] = {}
        for i, lab in enumerate(labels):
            label_to_indices.setdefault(lab, []).append(i)
        if len(label_to_indices) <= 1:
            return 0.0
        
        def avg_distance(i: int, indices: List[int]) -> float:
            if not indices:
                return 0.0
            vi = vectors[i]
            s = 0.0
            cnt = 0
            for j in indices:
                if j == i:
                    continue
                sim = self._cosine(vi, vectors[j])
                s += 1.0 - sim
                cnt += 1
            return s / max(1, cnt)

        total = 0.0
        for i in range(n):
            lab = labels[i]
            own = label_to_indices.get(lab, [])
            a = avg_distance(i, own)
            b = float('inf')
            for other_lab, idxs in label_to_indices.items():
                if other_lab == lab:
                    continue
                b = min(b, avg_distance(i, idxs))
            if b == float('inf'):
                b = 0.0
            denom = max(a, b)
            s = 0.0 if denom == 0.0 else (b - a) / denom
            total += s
        return total / n

    def _nli(self, premise: str, hypothesis: str) -> Tuple[str, Dict[str, float]]:
        
        if self.nli_model is None:
            raise RuntimeError("nli_model is required for NLI (no fallback).")
        out = self.nli_model.nli(premise, hypothesis)
        if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict):
            label, probs = out
            return str(label), {k: float(v) for k, v in probs.items()}
        if isinstance(out, dict):
            label = out.get('label', '')
            probs = out.get('probs', {})
            if isinstance(probs, dict):
                return str(label), {k: float(v) for k, v in probs.items()}
        raise RuntimeError("nli must return (label, probs_dict) or {'label':..., 'probs':{...}}")

    def _score_with_model(self, prompt: str, continuation: str) -> Optional[float]:
        
        if self.model is None:
            return None
        try:
            out = self.model.score(prompt=prompt, continuation=continuation)
            avg_nll = self._extract_avg_nll(out)
            if avg_nll is not None:
                return avg_nll
        except Exception as e:
            print(f"[DEBUG] model.score() failed: {e}")
            import traceback
            traceback.print_exc()
        try:
            out = self.model.logprobs(prompt=prompt, continuation=continuation)
            logs = self._extract_logprobs_list(out)
            if logs:
                return -sum(logs) / len(logs)
        except Exception as e:
            print(f"[DEBUG] model.logprobs() failed: {e}")
            import traceback
            traceback.print_exc()
        return None

    def _extract_avg_nll(self, out: Any) -> Optional[float]:
        if isinstance(out, dict):
            if 'avg_nll' in out:
                try:
                    return float(out['avg_nll'])
                except Exception:
                    return None
            if 'token_logprobs' in out and isinstance(out['token_logprobs'], list) and out['token_logprobs']:
                logs = [float(x) for x in out['token_logprobs'] if x is not None]
                if logs:
                    return -sum(logs) / len(logs)
        elif isinstance(out, (list, tuple)) and out:
            try:
                logs = [float(x) for x in out]
                return -sum(logs) / len(logs)
            except Exception:
                return None
        return None

    def _extract_logprobs_list(self, out: Any) -> List[float]:
        if isinstance(out, dict):
            if 'logprobs' in out and isinstance(out['logprobs'], list):
                try:
                    return [float(x) for x in out['logprobs'] if x is not None]
                except Exception:
                    return []
            if 'token_logprobs' in out and isinstance(out['token_logprobs'], list):
                try:
                    return [float(x) for x in out['token_logprobs'] if x is not None]
                except Exception:
                    return []
        elif isinstance(out, (list, tuple)):
            try:
                return [float(x) for x in out]
            except Exception:
                return []
        return []

    def _split_into_sentences(self, text: str) -> List[str]:
        if not text:
            return []
        
        sentences = re.split(r'(?<=[.!?])\s+', text)

        
        sentences = [s.strip()
                    for s in sentences if s.strip()]

        return sentences
