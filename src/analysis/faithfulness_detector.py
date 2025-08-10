"""
Main faithfulness detection system for chain-of-thought reasoning.
Combines attribution graphs with machine learning for automated detection.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path
import json

from .attribution_graphs import AttributionGraphBuilder, AttributionGraph, FaithfulnessAnalyzer
from ..models.gpt2_wrapper import GPT2Wrapper

@dataclass
class FaithfulnessResult:
    """Result of faithfulness analysis."""
    is_faithful: bool
    confidence: float
    pattern_type: str
    attribution_score: float
    reasoning_quality: float
    explanation: str
    intermediate_steps: Dict[str, Any]

@dataclass
class DetectionFeatures:
    """Features extracted for faithfulness detection."""
    # Graph structure features
    num_nodes: int
    num_edges: int
    avg_path_length: float
    max_path_length: int
    graph_density: float
    
    # Attribution features
    total_attribution: float
    max_attribution: float
    attribution_variance: float
    layer_distribution: List[float]
    
    # Pattern features
    has_shortcuts: bool
    has_backward_flow: bool
    reasoning_depth: int
    step_consistency: float
    
    # Token features
    target_token_frequency: float
    context_relevance: float
    semantic_coherence: float

class FaithfulnessDetector:
    """
    Main system for detecting faithful vs unfaithful chain-of-thought reasoning.
    
    Uses attribution graphs and machine learning to classify reasoning patterns.
    """
    
    def __init__(
        self,
        model: GPT2Wrapper,
        detector_type: str = "random_forest",
        confidence_threshold: float = 0.7
    ):
        self.model = model
        self.detector_type = detector_type
        self.confidence_threshold = confidence_threshold
        
        # Initialize components
        self.graph_builder = AttributionGraphBuilder(model)
        self.faithfulness_analyzer = FaithfulnessAnalyzer(self.graph_builder)
        
        # Initialize classifier
        self.classifier = self._create_classifier(detector_type)
        self.feature_scaler = None
        self.is_trained = False
        
        # Feature extraction settings
        self.feature_extractors = {
            'graph_structure': self._extract_graph_structure_features,
            'attribution_patterns': self._extract_attribution_features,
            'reasoning_flow': self._extract_reasoning_flow_features,
            'semantic_features': self._extract_semantic_features
        }
    
    def _create_classifier(self, detector_type: str):
        """Create the appropriate classifier."""
        if detector_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
        elif detector_type == "logistic_regression":
            return LogisticRegression(
                random_state=42,
                class_weight='balanced',
                max_iter=1000
            )
        else:
            raise ValueError(f"Unknown detector type: {detector_type}")
    
    def train(
        self,
        training_examples: List[Dict[str, Any]],
        validation_split: float = 0.2,
        save_path: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Train the faithfulness detector.
        
        Args:
            training_examples: List of training examples with labels
            validation_split: Fraction for validation
            save_path: Path to save trained model
            
        Returns:
            Training metrics
        """
        
        print(f"Training faithfulness detector with {len(training_examples)} examples...")
        
        # Extract features and labels
        features_list = []
        labels = []
        
        for example in training_examples:
            try:
                # Extract features from example
                features = self._extract_all_features(
                    example['prompt'],
                    example['reasoning_steps'],
                    example['target_tokens']
                )
                
                features_list.append(self._features_to_vector(features))
                labels.append(1 if example['is_faithful'] else 0)
                
            except Exception as e:
                print(f"Warning: Failed to process example: {e}")
                continue
        
        if len(features_list) == 0:
            raise ValueError("No valid training examples processed")
        
        # Convert to arrays
        X = np.array(features_list)
        y = np.array(labels)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        # Scale features
        from sklearn.preprocessing import StandardScaler
        self.feature_scaler = StandardScaler()
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_val_scaled = self.feature_scaler.transform(X_val)
        
        # Train classifier
        self.classifier.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Evaluate on validation set
        val_predictions = self.classifier.predict(X_val_scaled)
        val_probabilities = self.classifier.predict_proba(X_val_scaled)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_val, val_predictions),
            'precision': precision_recall_fscore_support(y_val, val_predictions, average='binary')[0],
            'recall': precision_recall_fscore_support(y_val, val_predictions, average='binary')[1],
            'f1_score': precision_recall_fscore_support(y_val, val_predictions, average='binary')[2],
            'auc_roc': roc_auc_score(y_val, val_probabilities)
        }
        
        print(f"Validation metrics: {metrics}")
        
        # Save model if requested
        if save_path:
            self.save(save_path)
        
        return metrics
    
    def analyze_reasoning(
        self,
        prompt: str,
        reasoning_steps: Optional[List[str]] = None,
        target_tokens: Optional[List[str]] = None,
        generate_steps: bool = True
    ) -> FaithfulnessResult:
        """
        Analyze the faithfulness of reasoning for a given prompt.
        
        Args:
            prompt: Input prompt
            reasoning_steps: Manual reasoning steps (optional)
            target_tokens: Tokens to analyze (optional)
            generate_steps: Whether to generate reasoning if not provided
            
        Returns:
            Faithfulness analysis result
        """
        
        if not self.is_trained:
            raise ValueError("Detector must be trained before use")
        
        # Generate reasoning steps if not provided
        if reasoning_steps is None and generate_steps:
            reasoning_steps = self._generate_reasoning_steps(prompt)
        
        if target_tokens is None and reasoning_steps:
            target_tokens = self._extract_key_tokens(reasoning_steps)
        
        if not reasoning_steps or not target_tokens:
            return FaithfulnessResult(
                is_faithful=False,
                confidence=0.0,
                pattern_type="no_reasoning",
                attribution_score=0.0,
                reasoning_quality=0.0,
                explanation="No reasoning steps found",
                intermediate_steps={}
            )
        
        # Extract features
        features = self._extract_all_features(prompt, reasoning_steps, target_tokens)
        feature_vector = self._features_to_vector(features)
        
        # Scale features
        feature_vector_scaled = self.feature_scaler.transform([feature_vector])
        
        # Predict faithfulness
        prediction = self.classifier.predict(feature_vector_scaled)[0]
        probability = self.classifier.predict_proba(feature_vector_scaled)[0]
        
        # Determine confidence and pattern
        confidence = max(probability)
        is_faithful = bool(prediction)
        pattern_type = self._determine_pattern_type(features)
        
        # Compute additional scores
        attribution_score = features.total_attribution
        reasoning_quality = self._compute_reasoning_quality(features)
        
        # Generate explanation
        explanation = self._generate_explanation(
            is_faithful, confidence, pattern_type, features
        )
        
        return FaithfulnessResult(
            is_faithful=is_faithful,
            confidence=confidence,
            pattern_type=pattern_type,
            attribution_score=attribution_score,
            reasoning_quality=reasoning_quality,
            explanation=explanation,
            intermediate_steps={
                'features': features,
                'raw_prediction': prediction,
                'probabilities': probability.tolist(),
                'reasoning_steps': reasoning_steps,
                'target_tokens': target_tokens
            }
        )
    
    def _extract_all_features(
        self,
        prompt: str,
        reasoning_steps: List[str],
        target_tokens: List[str]
    ) -> DetectionFeatures:
        """Extract all features for faithfulness detection."""
        
        # Analyze reasoning with attribution graphs
        analysis = self.faithfulness_analyzer.analyze_faithfulness(
            prompt, reasoning_steps, target_tokens
        )
        
        # Extract features from different aspects
        graph_features = {}
        attribution_features = {}
        flow_features = {}
        semantic_features = {}
        
        # Process each step analysis
        valid_steps = [step for step in analysis['step_analyses'] if 'graph' in step]
        
        if valid_steps:
            # Aggregate features across steps
            all_graphs = [step['graph'] for step in valid_steps]
            
            graph_features = self._extract_graph_structure_features(all_graphs)
            attribution_features = self._extract_attribution_features(all_graphs)
            flow_features = self._extract_reasoning_flow_features(all_graphs, analysis)
            semantic_features = self._extract_semantic_features(prompt, reasoning_steps)
        
        # Combine all features
        return DetectionFeatures(
            # Graph structure
            num_nodes=graph_features.get('avg_nodes', 0),
            num_edges=graph_features.get('avg_edges', 0),
            avg_path_length=graph_features.get('avg_path_length', 0),
            max_path_length=graph_features.get('max_path_length', 0),
            graph_density=graph_features.get('density', 0),
            
            # Attribution
            total_attribution=attribution_features.get('total_attribution', 0),
            max_attribution=attribution_features.get('max_attribution', 0),
            attribution_variance=attribution_features.get('attribution_variance', 0),
            layer_distribution=attribution_features.get('layer_distribution', [0] * 12),
            
            # Pattern features
            has_shortcuts=flow_features.get('has_shortcuts', False),
            has_backward_flow=flow_features.get('has_backward_flow', False),
            reasoning_depth=flow_features.get('reasoning_depth', 0),
            step_consistency=flow_features.get('step_consistency', 0),
            
            # Semantic features
            target_token_frequency=semantic_features.get('target_token_frequency', 0),
            context_relevance=semantic_features.get('context_relevance', 0),
            semantic_coherence=semantic_features.get('semantic_coherence', 0)
        )
    
    def _extract_graph_structure_features(self, graphs: List[AttributionGraph]) -> Dict[str, float]:
        """Extract graph structure features."""
        if not graphs:
            return {}
        
        features = {}
        
        # Average metrics across graphs
        features['avg_nodes'] = np.mean([len(g.nodes) for g in graphs])
        features['avg_edges'] = np.mean([len(g.edges) for g in graphs])
        
        # Path analysis
        path_lengths = []
        densities = []
        
        for graph in graphs:
            output_nodes = [n for n in graph.nodes.values() if n.feature_type == "output"]
            if output_nodes:
                paths = graph.get_paths_to_output(output_nodes[0].node_id)
                if paths:
                    path_lengths.extend([len(path) for path in paths])
            
            # Graph density
            n_nodes = len(graph.nodes)
            n_edges = len(graph.edges)
            max_edges = n_nodes * (n_nodes - 1) / 2 if n_nodes > 1 else 1
            density = n_edges / max_edges if max_edges > 0 else 0
            densities.append(density)
        
        features['avg_path_length'] = np.mean(path_lengths) if path_lengths else 0
        features['max_path_length'] = max(path_lengths) if path_lengths else 0
        features['density'] = np.mean(densities) if densities else 0
        
        return features
    
    def _extract_attribution_features(self, graphs: List[AttributionGraph]) -> Dict[str, float]:
        """Extract attribution-based features."""
        if not graphs:
            return {}
        
        features = {}
        
        # Attribution statistics
        all_attributions = []
        layer_attributions = [[] for _ in range(12)]  # GPT-2 has 12 layers
        
        for graph in graphs:
            for node in graph.nodes.values():
                all_attributions.append(abs(node.attribution_score))
                if 0 <= node.layer < 12:
                    layer_attributions[node.layer].append(abs(node.attribution_score))
        
        features['total_attribution'] = sum(all_attributions)
        features['max_attribution'] = max(all_attributions) if all_attributions else 0
        features['attribution_variance'] = np.var(all_attributions) if all_attributions else 0
        
        # Layer distribution
        layer_sums = [sum(layer_attrs) for layer_attrs in layer_attributions]
        total_sum = sum(layer_sums) if sum(layer_sums) > 0 else 1
        features['layer_distribution'] = [s / total_sum for s in layer_sums]
        
        return features
    
    def _extract_reasoning_flow_features(
        self, 
        graphs: List[AttributionGraph], 
        analysis: Dict[str, Any]
    ) -> Dict[str, float]:
        """Extract reasoning flow features."""
        if not graphs:
            return {}
        
        features = {}
        
        # Pattern detection
        shortcuts = 0
        backward_flows = 0
        reasoning_depths = []
        
        for graph in graphs:
            # Check for shortcuts (direct input to output connections)
            input_nodes = [n.node_id for n in graph.nodes.values() if n.feature_type == "input"]
            output_nodes = [n.node_id for n in graph.nodes.values() if n.feature_type == "output"]
            
            for inp in input_nodes:
                for out in output_nodes:
                    if graph.graph.has_edge(inp, out):
                        shortcuts += 1
            
            # Check for backward flow
            for edge in graph.edges:
                source_layer = graph.nodes[edge.source].layer
                target_layer = graph.nodes[edge.target].layer
                if source_layer > target_layer:
                    backward_flows += 1
            
            # Reasoning depth (max layer with significant attribution)
            max_layer = max([n.layer for n in graph.nodes.values() 
                           if abs(n.attribution_score) > 0.1], default=0)
            reasoning_depths.append(max_layer)
        
        features['has_shortcuts'] = shortcuts > 0
        features['has_backward_flow'] = backward_flows > 0
        features['reasoning_depth'] = np.mean(reasoning_depths) if reasoning_depths else 0
        
        # Step consistency
        step_scores = [step.get('faithfulness_score', 0) for step in analysis.get('step_analyses', [])]
        features['step_consistency'] = 1 - np.var(step_scores) if step_scores else 0
        
        return features
    
    def _extract_semantic_features(
        self, 
        prompt: str, 
        reasoning_steps: List[str]
    ) -> Dict[str, float]:
        """Extract semantic coherence features."""
        
        features = {}
        
        # Token frequency analysis
        prompt_tokens = self.model.tokenizer.tokenize(prompt.lower())
        all_step_tokens = []
        
        for step in reasoning_steps:
            step_tokens = self.model.tokenizer.tokenize(step.lower())
            all_step_tokens.extend(step_tokens)
        
        # Target token frequency
        if all_step_tokens:
            unique_tokens = set(all_step_tokens)
            features['target_token_frequency'] = len(unique_tokens) / len(all_step_tokens)
        else:
            features['target_token_frequency'] = 0
        
        # Context relevance (overlap between prompt and reasoning)
        prompt_set = set(prompt_tokens)
        step_set = set(all_step_tokens)
        overlap = len(prompt_set.intersection(step_set))
        features['context_relevance'] = overlap / max(len(prompt_set), 1)
        
        # Semantic coherence (simplified as step length consistency)
        if reasoning_steps:
            step_lengths = [len(step.split()) for step in reasoning_steps]
            features['semantic_coherence'] = 1 - (np.std(step_lengths) / np.mean(step_lengths)) if np.mean(step_lengths) > 0 else 0
        else:
            features['semantic_coherence'] = 0
        
        return features
    
    def _features_to_vector(self, features: DetectionFeatures) -> np.ndarray:
        """Convert features object to numerical vector."""
        
        vector = [
            features.num_nodes,
            features.num_edges,
            features.avg_path_length,
            features.max_path_length,
            features.graph_density,
            features.total_attribution,
            features.max_attribution,
            features.attribution_variance,
            *features.layer_distribution,  # 12 values
            int(features.has_shortcuts),
            int(features.has_backward_flow),
            features.reasoning_depth,
            features.step_consistency,
            features.target_token_frequency,
            features.context_relevance,
            features.semantic_coherence
        ]
        
        return np.array(vector, dtype=np.float32)
    
    def _generate_reasoning_steps(self, prompt: str) -> List[str]:
        """Generate reasoning steps using the model."""
        
        # Add thinking prompt
        thinking_prompt = prompt + " Let me think step by step:\n"
        
        # Generate response
        result = self.model.generate_with_cache(
            thinking_prompt,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True
        )
        
        generated_text = result['generated_text']
        
        # Parse into steps (simple splitting by newlines/periods)
        steps = []
        for line in generated_text.split('\n'):
            line = line.strip()
            if line and len(line) > 10:  # Filter very short lines
                steps.append(line)
        
        return steps[:5]  # Limit to 5 steps
    
    def _extract_key_tokens(self, reasoning_steps: List[str]) -> List[str]:
        """Extract key tokens from reasoning steps."""
        
        key_tokens = []
        
        for step in reasoning_steps:
            tokens = self.model.tokenizer.tokenize(step)
            
            # Simple heuristic: pick tokens that appear to be important
            # (numbers, key words, etc.)
            for token in tokens:
                if any(char.isdigit() for char in token) or len(token) > 4:
                    key_tokens.append(token)
                    break  # One token per step
        
        return key_tokens
    
    def _determine_pattern_type(self, features: DetectionFeatures) -> str:
        """Determine the reasoning pattern type."""
        
        if features.has_shortcuts and features.avg_path_length < 2:
            return "shortcut"
        elif features.has_backward_flow:
            return "backward_chaining"
        elif features.reasoning_depth >= 6 and features.step_consistency > 0.7:
            return "faithful"
        elif features.step_consistency < 0.3:
            return "confabulation"
        else:
            return "mixed"
    
    def _compute_reasoning_quality(self, features: DetectionFeatures) -> float:
        """Compute overall reasoning quality score."""
        
        quality_factors = [
            features.step_consistency,
            min(features.reasoning_depth / 6, 1.0),  # Normalize to 0-1
            features.context_relevance,
            features.semantic_coherence,
            1.0 if not features.has_shortcuts else 0.5,  # Penalty for shortcuts
        ]
        
        return np.mean(quality_factors)
    
    def _generate_explanation(
        self,
        is_faithful: bool,
        confidence: float,
        pattern_type: str,
        features: DetectionFeatures
    ) -> str:
        """Generate human-readable explanation."""
        
        base_msg = f"The reasoning appears to be {'faithful' if is_faithful else 'unfaithful'} "
        base_msg += f"(confidence: {confidence:.2f}). "
        
        pattern_explanations = {
            "faithful": "The model follows a coherent logical progression through its reasoning steps.",
            "shortcut": "The model appears to jump directly to conclusions without proper intermediate steps.",
            "backward_chaining": "The model seems to work backwards from a desired conclusion.",
            "confabulation": "The reasoning steps appear plausible but lack internal consistency.",
            "mixed": "The reasoning shows a mixture of different patterns."
        }
        
        explanation = base_msg + pattern_explanations.get(pattern_type, "")
        
        # Add specific observations
        if features.has_shortcuts:
            explanation += " Direct input-output connections were detected."
        if features.has_backward_flow:
            explanation += " Backward information flow was observed."
        if features.step_consistency < 0.5:
            explanation += " Low consistency between reasoning steps."
        
        return explanation
    
    def save(self, path: str) -> None:
        """Save trained detector to file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'classifier': self.classifier,
            'feature_scaler': self.feature_scaler,
            'detector_type': self.detector_type,
            'confidence_threshold': self.confidence_threshold,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, path)
    
    @classmethod
    def load(cls, path: str, model: GPT2Wrapper) -> 'FaithfulnessDetector':
        """Load trained detector from file."""
        model_data = joblib.load(path)
        
        detector = cls(
            model=model,
            detector_type=model_data['detector_type'],
            confidence_threshold=model_data['confidence_threshold']
        )
        
        detector.classifier = model_data['classifier']
        detector.feature_scaler = model_data['feature_scaler']
        detector.is_trained = model_data['is_trained']
        
        return detector
    
    def evaluate(self, test_examples: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate detector on test examples."""
        
        if not self.is_trained:
            raise ValueError("Detector must be trained before evaluation")
        
        predictions = []
        true_labels = []
        confidences = []
        
        for example in test_examples:
            try:
                result = self.analyze_reasoning(
                    example['prompt'],
                    example.get('reasoning_steps'),
                    example.get('target_tokens')
                )
                
                predictions.append(1 if result.is_faithful else 0)
                true_labels.append(1 if example['is_faithful'] else 0)
                confidences.append(result.confidence)
                
            except Exception as e:
                print(f"Warning: Failed to evaluate example: {e}")
                continue
        
        if not predictions:
            return {"error": "No valid test examples"}
        
        # Compute metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='binary'
        )
        
        try:
            auc = roc_auc_score(true_labels, confidences)
        except:
            auc = 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc,
            'avg_confidence': np.mean(confidences),
            'num_examples': len(predictions)
        }
