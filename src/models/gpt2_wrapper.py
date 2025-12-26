"""
GPT-2 model wrapper with enhanced interpretability features.
Provides hooks for activation extraction and intervention capabilities.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Callable
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformer_lens import HookedTransformer
import numpy as np
from dataclasses import dataclass
from pathlib import Path
import json

@dataclass
class ActivationCache:
    """Cache for storing model activations."""
    activations: Dict[str, torch.Tensor]
    attention_patterns: Dict[str, torch.Tensor]
    layer_outputs: Dict[str, torch.Tensor]
    metadata: Dict[str, Any]

class GPT2Wrapper:
    """
    Enhanced GPT-2 wrapper for mechanistic interpretability.
    
    Features:
    - Activation caching and extraction
    - Layer-wise intervention capabilities
    - Attribution graph computation support
    - Sparse autoencoder integration
    """
    
    def __init__(
        self,
        model_name: str = "gpt2",
        device: str = "auto",
        cache_activations: bool = True,
        max_length: int = 512
    ):
        self.model_name = model_name
        self.device = self._setup_device(device)
        self.cache_activations = cache_activations
        self.max_length = max_length
        
        # Load model and tokenizer
        self.model = self._load_model()
        self.tokenizer = self._load_tokenizer()
        
        # Initialize hooks and caches
        self.hooks = {}
        self.activation_cache = None
        self.intervention_hooks = {}
        
    def _setup_device(self, device: str) -> str:
        """Setup compute device."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def _load_model(self) -> HookedTransformer:
        """Load the hooked transformer model."""
        model = HookedTransformer.from_pretrained(
            self.model_name,
            device=self.device,
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False
        )
        model.eval()
        return model
    
    def _load_tokenizer(self) -> GPT2Tokenizer:
        """Load the tokenizer."""
        tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    
    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs) -> 'GPT2Wrapper':
        """Load model from pretrained checkpoint."""
        return cls(model_name=model_name, **kwargs)
    
    def tokenize(
        self, 
        text: str, 
        return_tensors: bool = True,
        add_special_tokens: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Tokenize input text."""
        tokens = self.tokenizer(
            text,
            return_tensors="pt" if return_tensors else None,
            add_special_tokens=add_special_tokens,
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        
        if return_tensors:
            tokens = {k: v.to(self.device) for k, v in tokens.items()}
        
        return tokens
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cache_activations: Optional[bool] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[ActivationCache]]:
        """
        Forward pass with optional activation caching.
        
        Returns:
            logits: Model output logits
            cache: Activation cache if enabled
        """
        cache_activations = cache_activations or self.cache_activations
        
        if cache_activations:
            return self._forward_with_cache(input_ids, attention_mask, **kwargs)
        else:
            logits = self.model(input_ids, attention_mask=attention_mask, **kwargs)
            return logits, None
    
    def _forward_with_cache(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, ActivationCache]:
        """Forward pass with activation caching."""
        activations = {}
        attention_patterns = {}
        layer_outputs = {}
        
        def cache_hook(name: str):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    activations[name] = output[0].detach().clone()
                    if len(output) > 1 and "attn" in name:
                        attention_patterns[name] = output[1].detach().clone()
                else:
                    activations[name] = output.detach().clone()
            return hook
        
        # Register hooks for all layers
        hook_handles = []
        for name, module in self.model.named_modules():
            if any(layer_type in name for layer_type in ["mlp", "attn", "ln"]):
                handle = module.register_forward_hook(cache_hook(name))
                hook_handles.append(handle)
        
        try:
            # Forward pass
            logits = self.model(input_ids, attention_mask=attention_mask, **kwargs)
            
            # Create activation cache
            cache = ActivationCache(
                activations=activations,
                attention_patterns=attention_patterns,
                layer_outputs=layer_outputs,
                metadata={
                    "input_ids": input_ids.cpu(),
                    "attention_mask": attention_mask.cpu() if attention_mask is not None else None,
                    "sequence_length": input_ids.shape[1],
                    "batch_size": input_ids.shape[0]
                }
            )
            
            return logits, cache
            
        finally:
            # Clean up hooks
            for handle in hook_handles:
                handle.remove()
    
    def generate_with_cache(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_p: float = 0.9,
        do_sample: bool = True,
        cache_activations: bool = True
    ) -> Dict[str, Any]:
        """
        Generate text with activation caching.
        
        Returns:
            Dictionary containing generated text, tokens, and activations
        """
        # Tokenize input
        inputs = self.tokenize(prompt)
        input_ids = inputs["input_ids"]
        
        # Generate
        with torch.no_grad():
            if cache_activations:
                # Generate with caching (step by step for full cache)
                generated_ids = input_ids.clone()
                all_caches = []
                
                for _ in range(max_new_tokens):
                    logits, cache = self.forward(generated_ids, cache_activations=True)
                    all_caches.append(cache)
                    
                    # Sample next token
                    next_token_logits = logits[0, -1, :] / temperature
                    if do_sample:
                        if top_p < 1.0:
                            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                            sorted_indices_to_remove = cumulative_probs > top_p
                            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                            sorted_indices_to_remove[0] = 0
                            indices_to_remove = sorted_indices[sorted_indices_to_remove]
                            next_token_logits[indices_to_remove] = float('-inf')
                        
                        probs = torch.softmax(next_token_logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                    else:
                        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    
                    # Check for EOS
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break
                        
                    generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)
                
                final_cache = all_caches[-1] if all_caches else None
            else:
                # Standard generation
                generated_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                final_cache = None
        
        # Decode generated text
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        new_text = generated_text[len(prompt):]
        
        return {
            "prompt": prompt,
            "generated_text": new_text,
            "full_text": generated_text,
            "input_ids": input_ids.cpu(),
            "generated_ids": generated_ids.cpu(),
            "cache": final_cache
        }
    
    def intervene_on_layer(
        self,
        layer_idx: int,
        intervention_fn: Callable[[torch.Tensor], torch.Tensor],
        component: str = "mlp"
    ) -> None:
        """
        Add intervention hook to specific layer.
        
        Args:
            layer_idx: Layer index to intervene on
            intervention_fn: Function to apply to activations
            component: Component to intervene on ("mlp", "attn", "resid")
        """
        hook_name = f"blocks.{layer_idx}.{component}"
        
        def intervention_hook(module, input, output):
            if isinstance(output, tuple):
                modified_output = list(output)
                modified_output[0] = intervention_fn(output[0])
                return tuple(modified_output)
            else:
                return intervention_fn(output)
        
        # Remove existing hook if present
        if hook_name in self.intervention_hooks:
            self.intervention_hooks[hook_name].remove()
        
        # Add new hook
        layer_module = dict(self.model.named_modules())[hook_name]
        handle = layer_module.register_forward_hook(intervention_hook)
        self.intervention_hooks[hook_name] = handle
    
    def clear_interventions(self) -> None:
        """Clear all intervention hooks."""
        for handle in self.intervention_hooks.values():
            handle.remove()
        self.intervention_hooks.clear()
    
    def get_layer_activations(
        self,
        input_ids: torch.Tensor,
        layer_idx: int,
        component: str = "mlp"
    ) -> torch.Tensor:
        """Get activations for a specific layer and component."""
        _, cache = self.forward(input_ids, cache_activations=True)
        hook_name = f"blocks.{layer_idx}.{component}"
        return cache.activations.get(hook_name)
    
    def get_attention_patterns(
        self,
        input_ids: torch.Tensor,
        layer_idx: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """Get attention patterns for specified layers."""
        _, cache = self.forward(input_ids, cache_activations=True)
        
        if layer_idx is not None:
            pattern_name = f"blocks.{layer_idx}.attn"
            return {pattern_name: cache.attention_patterns.get(pattern_name)}
        else:
            return cache.attention_patterns
    
    def save_cache(self, cache: ActivationCache, path: str) -> None:
        """Save activation cache to file."""
        cache_data = {
            "activations": {k: v.cpu().numpy() for k, v in cache.activations.items()},
            "attention_patterns": {k: v.cpu().numpy() for k, v in cache.attention_patterns.items()},
            "metadata": cache.metadata
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache_data, path)
    
    def load_cache(self, path: str) -> ActivationCache:
        """Load activation cache from file."""
        cache_data = torch.load(path, map_location=self.device)
        
        return ActivationCache(
            activations={k: torch.from_numpy(v).to(self.device) for k, v in cache_data["activations"].items()},
            attention_patterns={k: torch.from_numpy(v).to(self.device) for k, v in cache_data["attention_patterns"].items()},
            layer_outputs={},
            metadata=cache_data["metadata"]
        )
    
    def analyze_reasoning_step(
        self,
        prompt: str,
        target_token: str,
        layer_range: Optional[Tuple[int, int]] = None
    ) -> Dict[str, Any]:
        """
        Analyze a single reasoning step in detail.
        
        Args:
            prompt: Input prompt
            target_token: Token to analyze
            layer_range: Range of layers to analyze
            
        Returns:
            Analysis results including activations and attribution scores
        """
        # Generate with caching
        result = self.generate_with_cache(prompt, max_new_tokens=1, cache_activations=True)
        
        # Find target token position
        full_tokens = self.tokenizer.tokenize(result["full_text"])
        target_idx = None
        for i, token in enumerate(full_tokens):
            if target_token in token:
                target_idx = i
                break
        
        if target_idx is None:
            raise ValueError(f"Target token '{target_token}' not found in generation")
        
        # Extract relevant activations
        cache = result["cache"]
        layer_range = layer_range or (0, self.model.cfg.n_layers)
        
        analysis = {
            "prompt": prompt,
            "target_token": target_token,
            "target_position": target_idx,
            "generated_text": result["generated_text"],
            "layer_activations": {},
            "attention_patterns": {},
            "attribution_scores": {}
        }
        
        # Extract layer-wise information
        for layer_idx in range(layer_range[0], layer_range[1]):
            mlp_key = f"blocks.{layer_idx}.mlp"
            attn_key = f"blocks.{layer_idx}.attn"
            
            if mlp_key in cache.activations:
                analysis["layer_activations"][layer_idx] = {
                    "mlp": cache.activations[mlp_key][:, target_idx, :].cpu().numpy(),
                }
            
            if attn_key in cache.attention_patterns:
                analysis["attention_patterns"][layer_idx] = cache.attention_patterns[attn_key].cpu().numpy()
        
        return analysis
    
    @property
    def num_layers(self) -> int:
        """Number of transformer layers."""
        return self.model.cfg.n_layers
    
    @property
    def hidden_size(self) -> int:
        """Hidden dimension size."""
        return self.model.cfg.d_model
    
    @property
    def vocab_size(self) -> int:
        """Vocabulary size."""
        return self.model.cfg.d_vocab
