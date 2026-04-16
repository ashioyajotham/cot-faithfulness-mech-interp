"""
Targeted interventions for manipulating faithfulness in chain-of-thought reasoning.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from pathlib import Path
import json

from ..models.gpt2_wrapper import GPT2Wrapper
from ..analysis.attribution_graphs import AttributionGraphBuilder, AttributionGraph

@dataclass
class InterventionResult:
    """Result of a faithfulness intervention."""
    original_output: str
    modified_output: str
    intervention_strength: float
    faithfulness_change: float
    accuracy_preserved: bool
    confidence_score: float
    intervention_details: Dict[str, Any]

class FaithfulnessInterventions:
    """
    System for performing targeted interventions to manipulate faithfulness.
    
    Implements methods to:
    - Increase faithfulness by strengthening reasoning circuits
    - Decrease faithfulness by introducing shortcuts
    - Analyze the effects of interventions
    """
    
    def __init__(
        self,
        model: GPT2Wrapper,
        intervention_layers: Optional[List[int]] = None,
        strength_range: Tuple[float, float] = (-3.0, 3.0)
    ):
        self.model = model
        self.intervention_layers = intervention_layers or [6, 7, 8]  # Middle layers
        self.strength_range = strength_range
        
        # Initialize attribution graph builder for analysis
        self.graph_builder = AttributionGraphBuilder(model)
        
        # Track active interventions
        self.active_interventions = {}
        
    def increase_faithfulness(
        self,
        prompt: str,
        strength: float = 2.0,
        target_layers: Optional[List[int]] = None,
        method: str = "strengthen_reasoning"
    ) -> InterventionResult:
        """
        Increase faithfulness of reasoning by strengthening reasoning circuits.
        
        Args:
            prompt: Input prompt
            strength: Intervention strength (positive values)
            target_layers: Layers to intervene on
            method: Intervention method
            
        Returns:
            Intervention result
        """
        
        target_layers = target_layers or self.intervention_layers
        
        # Get baseline output
        baseline_result = self.model.generate_with_cache(
            prompt, max_new_tokens=100, temperature=0.7
        )
        baseline_output = baseline_result['generated_text']
        
        # Clear any existing interventions
        self.clear_interventions()
        
        # Apply faithfulness-increasing intervention
        if method == "strengthen_reasoning":
            intervention_fn = self._create_reasoning_strengthening_intervention(strength)
        elif method == "suppress_shortcuts":
            intervention_fn = self._create_shortcut_suppression_intervention(strength)
        else:
            raise ValueError(f"Unknown intervention method: {method}")
        
        # Apply interventions to target layers
        for layer_idx in target_layers:
            self.model.intervene_on_layer(layer_idx, intervention_fn, component="mlp")
        
        try:
            # Generate with intervention
            modified_result = self.model.generate_with_cache(
                prompt, max_new_tokens=100, temperature=0.7
            )
            modified_output = modified_result['generated_text']
            
            # Analyze changes
            faithfulness_change = self._measure_faithfulness_change(
                prompt, baseline_output, modified_output
            )
            
            accuracy_preserved = self._check_accuracy_preservation(
                prompt, baseline_output, modified_output
            )
            
            confidence_score = self._compute_intervention_confidence(
                baseline_result, modified_result
            )
            
            return InterventionResult(
                original_output=baseline_output,
                modified_output=modified_output,
                intervention_strength=strength,
                faithfulness_change=faithfulness_change,
                accuracy_preserved=accuracy_preserved,
                confidence_score=confidence_score,
                intervention_details={
                    'method': method,
                    'target_layers': target_layers,
                    'prompt': prompt
                }
            )
            
        finally:
            # Clean up interventions
            self.clear_interventions()
    
    def decrease_faithfulness(
        self,
        prompt: str,
        strength: float = 1.5,
        target_layers: Optional[List[int]] = None,
        method: str = "introduce_shortcuts"
    ) -> InterventionResult:
        """
        Decrease faithfulness by introducing unfaithful reasoning patterns.
        
        Args:
            prompt: Input prompt
            strength: Intervention strength (positive values)
            target_layers: Layers to intervene on
            method: Intervention method
            
        Returns:
            Intervention result
        """
        
        target_layers = target_layers or self.intervention_layers
        
        # Get baseline output
        baseline_result = self.model.generate_with_cache(
            prompt, max_new_tokens=100, temperature=0.7
        )
        baseline_output = baseline_result['generated_text']
        
        # Clear any existing interventions
        self.clear_interventions()
        
        # Apply faithfulness-decreasing intervention
        if method == "introduce_shortcuts":
            intervention_fn = self._create_shortcut_introduction_intervention(strength)
        elif method == "add_noise":
            intervention_fn = self._create_noise_intervention(strength)
        elif method == "backward_flow":
            intervention_fn = self._create_backward_flow_intervention(strength)
        else:
            raise ValueError(f"Unknown intervention method: {method}")
        
        # Apply interventions to target layers
        for layer_idx in target_layers:
            self.model.intervene_on_layer(layer_idx, intervention_fn, component="mlp")
        
        try:
            # Generate with intervention
            modified_result = self.model.generate_with_cache(
                prompt, max_new_tokens=100, temperature=0.7
            )
            modified_output = modified_result['generated_text']
            
            # Analyze changes
            faithfulness_change = self._measure_faithfulness_change(
                prompt, baseline_output, modified_output
            )
            
            accuracy_preserved = self._check_accuracy_preservation(
                prompt, baseline_output, modified_output
            )
            
            confidence_score = self._compute_intervention_confidence(
                baseline_result, modified_result
            )
            
            return InterventionResult(
                original_output=baseline_output,
                modified_output=modified_output,
                intervention_strength=-strength,  # Negative for decreasing
                faithfulness_change=faithfulness_change,
                accuracy_preserved=accuracy_preserved,
                confidence_score=confidence_score,
                intervention_details={
                    'method': method,
                    'target_layers': target_layers,
                    'prompt': prompt
                }
            )
            
        finally:
            # Clean up interventions
            self.clear_interventions()
    
    def _create_reasoning_strengthening_intervention(self, strength: float) -> Callable:
        """Create intervention that strengthens reasoning pathways."""
        
        def intervention_fn(activations: torch.Tensor) -> torch.Tensor:
            """
            Strengthen activations that appear to be involved in reasoning.
            
            Heuristic: Amplify neurons with moderate positive activation
            """
            # Find neurons with moderate positive activation (reasoning-like)
            reasoning_mask = (activations > 0.1) & (activations < 2.0)
            
            # Strengthen these activations
            modified_activations = activations.clone()
            modified_activations[reasoning_mask] *= (1.0 + strength * 0.2)
            
            return modified_activations
        
        return intervention_fn
    
    def _create_shortcut_suppression_intervention(self, strength: float) -> Callable:
        """Create intervention that suppresses shortcut pathways."""
        
        def intervention_fn(activations: torch.Tensor) -> torch.Tensor:
            """
            Suppress activations that might represent shortcuts.
            
            Heuristic: Reduce very high activations (potential shortcuts)
            """
            # Find neurons with very high activation (potential shortcuts)
            shortcut_mask = activations > 3.0
            
            # Suppress these activations
            modified_activations = activations.clone()
            modified_activations[shortcut_mask] *= (1.0 - strength * 0.3)
            
            return modified_activations
        
        return intervention_fn
    
    def _create_shortcut_introduction_intervention(self, strength: float) -> Callable:
        """Create intervention that introduces shortcut pathways."""
        
        def intervention_fn(activations: torch.Tensor) -> torch.Tensor:
            """
            Introduce artificial shortcuts by amplifying random neurons.
            """
            # Create random shortcut pattern
            batch_size, seq_len, hidden_size = activations.shape
            
            # Select random neurons to amplify (shortcuts)
            num_shortcuts = int(hidden_size * 0.05)  # 5% of neurons
            shortcut_indices = torch.randperm(hidden_size)[:num_shortcuts]
            
            modified_activations = activations.clone()
            
            # Amplify selected neurons across all positions
            for idx in shortcut_indices:
                modified_activations[:, :, idx] += strength * torch.randn_like(
                    modified_activations[:, :, idx]
                ) * 0.5
            
            return modified_activations
        
        return intervention_fn
    
    def _create_noise_intervention(self, strength: float) -> Callable:
        """Create intervention that adds noise to disrupt reasoning."""
        
        def intervention_fn(activations: torch.Tensor) -> torch.Tensor:
            """Add noise to activations to disrupt coherent reasoning."""
            
            noise = torch.randn_like(activations) * strength * 0.1
            return activations + noise
        
        return intervention_fn
    
    def _create_backward_flow_intervention(self, strength: float) -> Callable:
        """Create intervention that simulates backward reasoning flow."""
        
        def intervention_fn(activations: torch.Tensor) -> torch.Tensor:
            """
            Simulate backward flow by mixing future and past information.
            """
            batch_size, seq_len, hidden_size = activations.shape
            
            # Create backward-flowing information
            if seq_len > 1:
                # Mix activations from later positions to earlier ones
                modified_activations = activations.clone()
                
                for pos in range(seq_len - 1):
                    # Add information from later positions
                    future_info = activations[:, pos + 1:, :].mean(dim=1, keepdim=True)
                    modified_activations[:, pos, :] += strength * future_info.squeeze(1) * 0.2
                
                return modified_activations
            else:
                return activations
        
        return intervention_fn
    
    def _measure_faithfulness_change(
        self, 
        prompt: str, 
        baseline_output: str, 
        modified_output: str
    ) -> float:
        """Measure change in faithfulness between baseline and modified outputs."""
        
        # Simple heuristic: compare reasoning consistency
        # In practice, this would use the faithfulness detector
        
        baseline_steps = self._extract_reasoning_steps(baseline_output)
        modified_steps = self._extract_reasoning_steps(modified_output)
        
        # Measure step coherence
        baseline_coherence = self._measure_step_coherence(baseline_steps)
        modified_coherence = self._measure_step_coherence(modified_steps)
        
        return modified_coherence - baseline_coherence
    
    def _extract_reasoning_steps(self, output: str) -> List[str]:
        """Extract reasoning steps from model output."""
        
        steps = []
        lines = output.split('\n')
        
        for line in lines:
            line = line.strip()
            if line and any(keyword in line.lower() for keyword in ['therefore', 'so', 'thus', 'because']):
                steps.append(line)
        
        return steps
    
    def _measure_step_coherence(self, steps: List[str]) -> float:
        """Measure coherence of reasoning steps."""
        
        if len(steps) < 2:
            return 0.5  # Neutral score for insufficient steps
        
        # Simple heuristic: consistent step length and presence of logical connectors
        step_lengths = [len(step.split()) for step in steps]
        length_consistency = 1.0 - np.std(step_lengths) / max(np.mean(step_lengths), 1)
        
        # Count logical connectors
        logical_connectors = ['therefore', 'so', 'thus', 'because', 'since', 'if', 'then']
        connector_count = sum(
            1 for step in steps 
            for connector in logical_connectors 
            if connector in step.lower()
        )
        connector_density = connector_count / len(steps)
        
        return (length_consistency + min(connector_density, 1.0)) / 2
    
    def _check_accuracy_preservation(
        self, 
        prompt: str, 
        baseline_output: str, 
        modified_output: str
    ) -> bool:
        """Check if intervention preserved answer accuracy."""
        
        # Extract final answers (simplified)
        baseline_answer = self._extract_final_answer(baseline_output)
        modified_answer = self._extract_final_answer(modified_output)
        
        # Check if answers are similar
        return self._answers_are_similar(baseline_answer, modified_answer)
    
    def _extract_final_answer(self, output: str) -> str:
        """Extract the final answer from model output."""
        
        # Look for numbers or key phrases at the end
        lines = output.strip().split('\n')
        
        for line in reversed(lines):
            line = line.strip()
            if line and (any(char.isdigit() for char in line) or len(line) > 3):
                return line
        
        return output.strip()[-50:]  # Last 50 characters as fallback
    
    def _answers_are_similar(self, answer1: str, answer2: str) -> bool:
        """Check if two answers are similar."""
        
        # Extract numbers
        import re
        numbers1 = re.findall(r'\d+(?:\.\d+)?', answer1)
        numbers2 = re.findall(r'\d+(?:\.\d+)?', answer2)
        
        if numbers1 and numbers2:
            # Compare numeric answers
            try:
                num1 = float(numbers1[-1])  # Last number
                num2 = float(numbers2[-1])
                return abs(num1 - num2) / max(abs(num1), abs(num2), 1) < 0.1
            except:
                pass
        
        # Fallback to string similarity
        answer1_clean = answer1.lower().strip()
        answer2_clean = answer2.lower().strip()
        
        return answer1_clean == answer2_clean
    
    def _compute_intervention_confidence(
        self, 
        baseline_result: Dict[str, Any], 
        modified_result: Dict[str, Any]
    ) -> float:
        """Compute confidence in the intervention effect."""
        
        # Analyze activation changes
        baseline_cache = baseline_result.get('cache')
        modified_cache = modified_result.get('cache')
        
        if baseline_cache is None or modified_cache is None:
            return 0.5  # Neutral confidence
        
        # Compare activation magnitudes in intervention layers
        total_change = 0.0
        num_comparisons = 0
        
        for layer_idx in self.intervention_layers:
            baseline_key = f"blocks.{layer_idx}.mlp"
            
            if baseline_key in baseline_cache.activations and baseline_key in modified_cache.activations:
                baseline_acts = baseline_cache.activations[baseline_key]
                modified_acts = modified_cache.activations[baseline_key]
                
                # Compute relative change
                change = torch.norm(modified_acts - baseline_acts) / torch.norm(baseline_acts)
                total_change += change.item()
                num_comparisons += 1
        
        if num_comparisons == 0:
            return 0.5
        
        avg_change = total_change / num_comparisons
        
        # Convert to confidence (higher change = higher confidence)
        confidence = min(avg_change / 2.0, 1.0)  # Normalize
        
        return confidence
    
    def clear_interventions(self) -> None:
        """Clear all active interventions."""
        self.model.clear_interventions()
        self.active_interventions.clear()
    
    def run_intervention_sweep(
        self,
        prompt: str,
        strength_levels: Optional[List[float]] = None,
        methods: Optional[List[str]] = None
    ) -> Dict[str, List[InterventionResult]]:
        """
        Run a sweep of interventions with different strengths and methods.
        
        Args:
            prompt: Input prompt
            strength_levels: List of strength values to test
            methods: List of intervention methods to test
            
        Returns:
            Dictionary mapping method names to lists of results
        """
        
        strength_levels = strength_levels or [0.5, 1.0, 1.5, 2.0, 2.5]
        methods = methods or ["strengthen_reasoning", "introduce_shortcuts"]
        
        results = {}
        
        for method in methods:
            method_results = []
            
            for strength in strength_levels:
                try:
                    if "strengthen" in method or "suppress" in method:
                        result = self.increase_faithfulness(
                            prompt, strength=strength, method=method
                        )
                    else:
                        result = self.decrease_faithfulness(
                            prompt, strength=strength, method=method
                        )
                    
                    method_results.append(result)
                    
                except Exception as e:
                    print(f"Warning: Intervention failed for {method} with strength {strength}: {e}")
                    continue
            
            results[method] = method_results
        
        return results
    
    def analyze_intervention_effects(
        self,
        sweep_results: Dict[str, List[InterventionResult]]
    ) -> Dict[str, Any]:
        """Analyze the effects of intervention sweeps."""
        
        analysis = {
            'method_summaries': {},
            'strength_effects': {},
            'overall_patterns': {}
        }
        
        for method, results in sweep_results.items():
            if not results:
                continue
            
            # Method summary
            faithfulness_changes = [r.faithfulness_change for r in results]
            accuracy_rates = [r.accuracy_preserved for r in results]
            confidence_scores = [r.confidence_score for r in results]
            
            analysis['method_summaries'][method] = {
                'avg_faithfulness_change': np.mean(faithfulness_changes),
                'accuracy_preservation_rate': np.mean(accuracy_rates),
                'avg_confidence': np.mean(confidence_scores),
                'num_successful': len(results)
            }
            
            # Strength effects
            strengths = [r.intervention_strength for r in results]
            analysis['strength_effects'][method] = {
                'strengths': strengths,
                'faithfulness_changes': faithfulness_changes,
                'accuracy_preserved': accuracy_rates
            }
        
        # Overall patterns
        all_results = [r for results in sweep_results.values() for r in results]
        if all_results:
            analysis['overall_patterns'] = {
                'total_interventions': len(all_results),
                'avg_faithfulness_change': np.mean([r.faithfulness_change for r in all_results]),
                'overall_accuracy_rate': np.mean([r.accuracy_preserved for r in all_results]),
                'effective_intervention_rate': np.mean([abs(r.faithfulness_change) > 0.1 for r in all_results])
            }
        
        return analysis
    
    def save_intervention_results(
        self, 
        results: Dict[str, List[InterventionResult]], 
        path: str
    ) -> None:
        """Save intervention results to file."""
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format
        serializable_results = {}
        
        for method, method_results in results.items():
            serializable_results[method] = []
            
            for result in method_results:
                serializable_results[method].append({
                    'original_output': result.original_output,
                    'modified_output': result.modified_output,
                    'intervention_strength': result.intervention_strength,
                    'faithfulness_change': result.faithfulness_change,
                    'accuracy_preserved': result.accuracy_preserved,
                    'confidence_score': result.confidence_score,
                    'intervention_details': result.intervention_details
                })
        
        with open(path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    
    def load_intervention_results(self, path: str) -> Dict[str, List[InterventionResult]]:
        """Load intervention results from file."""
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        results = {}
        
        for method, method_data in data.items():
            results[method] = []
            
            for result_data in method_data:
                results[method].append(InterventionResult(**result_data))
        
        return results
