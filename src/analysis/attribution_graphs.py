"""
Attribution graph construction and analysis for chain-of-thought reasoning.
Inspired by Anthropic's attribution graphs methodology.
"""

from kiwisolver import strength
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Set, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict
import networkx as nx
from pathlib import Path
import json
import pickle
from dataclasses import is_dataclass
import inspect

def _normalize_node_kwargs(kwargs: dict) -> dict:
    # layer_idx -> layer
    if "layer" not in kwargs and "layer_idx" in kwargs:
        kwargs["layer"] = kwargs.pop("layer_idx")
    # pos -> position
    if "position" not in kwargs and "pos" in kwargs:
        kwargs["position"] = kwargs.pop("pos")
    # component_type <-> feature_type
    if "feature_type" not in kwargs and "component_type" in kwargs:
        kwargs["feature_type"] = kwargs["component_type"]
    if "component_type" not in kwargs and "feature_type" in kwargs:
        kwargs["component_type"] = kwargs["feature_type"]
    # activation_strength fallbacks
    if "activation_strength" not in kwargs:
        for k in ("activation_value", "score", "importance", "weight"):
            if k in kwargs:
                kwargs["activation_strength"] = kwargs[k]
                break
    # Drop unknown parameters not in AttributionNode signature
    try:
        from .attribution_graphs import AttributionNode  # or local if defined here
    except Exception:
        pass
    try:
        sig = inspect.signature(AttributionNode)
        valid = set(sig.parameters.keys())
        kwargs = {k: v for k, v in kwargs.items() if k in valid}
    except Exception:
        pass
    return kwargs

def _make_node(**kwargs):
    from .attribution_graphs import AttributionNode  # adjust if class is in this module
    return AttributionNode(**_normalize_node_kwargs(kwargs))

@dataclass
class AttributionNode:
    """Represents a node in the attribution graph."""
    node_id: str
    layer: int
    position: int
    feature_type: str  # "mlp", "attn", "resid", "input", "output"
    activation_value: float
    attribution_score: float
    interpretation: Optional[str] = None
    examples: List[str] = field(default_factory=list)

    # Compatibility alias if other code uses `layer_idx`
    @property
    def layer_idx(self) -> int:
        return self.layer
    
@dataclass
class AttributionEdge:
    """Represents an edge in the attribution graph."""
    source: str
    target: str
    weight: float
    attribution_type: str  # "direct", "residual", "attention"
    confidence: float
    
@dataclass
class AttributionGraph:
    """Complete attribution graph for a reasoning step."""
    nodes: Dict[str, AttributionNode]
    edges: List[AttributionEdge]
    metadata: Dict[str, Any]
    pruning_threshold: float = 0.1
    
    def __post_init__(self):
        """Initialize graph structure."""
        self.graph = nx.DiGraph()
        self._build_networkx_graph()
    
    def _build_networkx_graph(self):
        """Build NetworkX representation for analysis."""
        # Add nodes
        for node_id, node in self.nodes.items():
            self.graph.add_node(
                node_id,
                layer=node.layer,
                position=node.position,
                feature_type=node.feature_type,
                activation=node.activation_value,
                attribution=node.attribution_score,
                interpretation=node.interpretation
            )
        
        # Add edges
        for edge in self.edges:
            if abs(edge.weight) >= self.pruning_threshold:
                self.graph.add_edge(
                    edge.source,
                    edge.target,
                    weight=edge.weight,
                    type=edge.attribution_type,
                    confidence=edge.confidence
                )
    
    def get_paths_to_output(self, output_node: str, max_depth: int = 5) -> List[List[str]]:
        """Find all paths from input to specified output node."""
        paths = []
        input_nodes = [n for n, d in self.graph.nodes(data=True) if d['feature_type'] == 'input']
        
        for input_node in input_nodes:
            try:
                for path in nx.all_simple_paths(self.graph, input_node, output_node, cutoff=max_depth):
                    paths.append(path)
            except nx.NetworkXNoPath:
                continue
                
        return paths
    
    def get_most_influential_path(self, output_node: str) -> Optional[List[str]]:
        """Get the path with highest total attribution score."""
        paths = self.get_paths_to_output(output_node)
        if not paths:
            return None
        
        best_path = None
        best_score = float('-inf')
        
        for path in paths:
            score = sum(abs(self.graph[path[i]][path[i+1]]['weight']) for i in range(len(path)-1))
            if score > best_score:
                best_score = score
                best_path = path
        
        return best_path
    
    def prune_graph(self, threshold: Optional[float] = None) -> 'AttributionGraph':
        """Create pruned version of graph by removing low-weight edges."""
        threshold = threshold or self.pruning_threshold
        
        pruned_edges = [e for e in self.edges if abs(e.weight) >= threshold]
        
        # Keep only nodes that are connected to remaining edges
        connected_nodes = set()
        for edge in pruned_edges:
            connected_nodes.add(edge.source)
            connected_nodes.add(edge.target)
        
        pruned_nodes = {nid: node for nid, node in self.nodes.items() if nid in connected_nodes}
        
        return AttributionGraph(
            nodes=pruned_nodes,
            edges=pruned_edges,
            metadata=self.metadata.copy(),
            pruning_threshold=threshold
        )
    
    def get_layer_summary(self) -> Dict[int, Dict[str, Any]]:
        """Summarize attribution by layer."""
        layer_summary = defaultdict(lambda: {'nodes': 0, 'total_attribution': 0, 'avg_attribution': 0})
        
        for node in self.nodes.values():
            layer = node.layer
            layer_summary[layer]['nodes'] += 1
            layer_summary[layer]['total_attribution'] += abs(node.attribution_score)
        
        for layer_data in layer_summary.values():
            if layer_data['nodes'] > 0:
                layer_data['avg_attribution'] = layer_data['total_attribution'] / layer_data['nodes']
        
        return dict(layer_summary)

class AttributionGraphBuilder:
    """
    Builds attribution graphs from model activations and gradients.
    Implements core methodology inspired by Anthropic's attribution graphs.
    """
    
    def __init__(
        self,
        model,
        top_k_features: int = 50,
        pruning_threshold: float = 0.0,
        max_graph_depth: int = 3
    ):
        self.model = model
        self.top_k_features = top_k_features
        self.pruning_threshold = pruning_threshold
        self.max_graph_depth = max_graph_depth

    # --- Compatibility helpers for node construction ---
    def _normalize_node_kwargs(self, kwargs: dict) -> dict:
        k = dict(kwargs)
        if "layer" not in k and "layer_idx" in k:
            k["layer"] = k.pop("layer_idx")
        if "position" not in k and "pos" in k:
            k["position"] = k.pop("pos")
        # Unify feature/component naming - map to feature_type and remove component_type
        if "feature_type" not in k and "component_type" in k:
            k["feature_type"] = k.pop("component_type")
        elif "component_type" in k:
            k.pop("component_type")  # Remove if feature_type already exists
        # Map activation_strength to activation_value (which AttributionNode expects)
        if "activation_value" not in k and "activation_strength" in k:
            k["activation_value"] = k.pop("activation_strength")
        elif "activation_strength" in k:
            k.pop("activation_strength")
        # Set defaults for required fields
        if "attribution_score" not in k:
            k["attribution_score"] = k.get("activation_value", 0.0)
        # Filter to only valid AttributionNode fields
        valid_fields = {"node_id", "layer", "position", "feature_type", "activation_value", 
                        "attribution_score", "interpretation", "examples"}
        k = {key: val for key, val in k.items() if key in valid_fields}
        return k

    def _make_node(self, **kwargs):
        # Import here to avoid circulars
        from .attribution_graphs import AttributionNode
        return AttributionNode(**self._normalize_node_kwargs(kwargs))

    def build_attribution_graph(
        self,
        input_ids: torch.Tensor,
        target_token_idx: int,
        target_layer: Optional[int] = None
    ) -> AttributionGraph:
        """
        Build attribution graph for a specific target token.
        
        Args:
            input_ids: Input token sequence
            target_token_idx: Index of token to analyze
            target_layer: Layer to focus analysis on (optional)
            
        Returns:
            Complete attribution graph
        """
        # Get activations and gradients
        activations, gradients = self._compute_activations_and_gradients(
            input_ids, target_token_idx
        )
        
        # Identify important features
        important_features = self._identify_important_features(
            activations, gradients, target_token_idx
        )
        
        # Build nodes
        nodes = self._create_nodes(important_features, activations, target_token_idx)
        
        # Build edges
        edges = self._create_edges(nodes, activations, gradients, target_token_idx)
        
        # Create graph
        metadata = {
            "input_text": self.model.tokenizer.decode(input_ids[0]),
            "target_token": self.model.tokenizer.decode([input_ids[0, target_token_idx]]),
            "target_position": target_token_idx,
            "num_layers": self.model.num_layers,
            "build_params": {
                "top_k_features": getattr(self, "top_k_features", 50),
                "pruning_threshold": getattr(self, "pruning_threshold", 0.0),  # Fixed: consistent naming
                "max_graph_depth": getattr(self, "max_graph_depth", 3),
            },
        }
        
        return AttributionGraph(
            nodes=nodes,
            edges=edges,
            metadata=metadata,
            pruning_threshold=self.pruning_threshold
        )
    
    def _compute_activations_and_gradients(
        self,
        input_ids: torch.Tensor,
        target_token_idx: int
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Compute forward activations and backward gradients."""
        
        # Forward pass with caching
        logits, cache = self.model.forward(input_ids, cache_activations=True)
        
        # Target logit for gradient computation
        target_logit = logits[0, target_token_idx, :]
        target_token_id = torch.argmax(target_logit, dim=-1)
        
        # Compute gradients
        target_logit[target_token_id].backward(retain_graph=True)
        
        # Extract gradients
        gradients = {}
        for name, module in self.model.model.named_modules():
            if hasattr(module, 'weight') and module.weight.grad is not None:
                gradients[name + '.weight'] = module.weight.grad.clone()
            if hasattr(module, 'bias') and module.bias is not None and module.bias.grad is not None:
                gradients[name + '.bias'] = module.bias.grad.clone()
        
        return cache.activations, gradients
    
    def _identify_important_features(
        self,
        activations: Dict[str, torch.Tensor],
        gradients: Dict[str, torch.Tensor],
        target_token_idx: int
    ) -> Dict[str, List[Tuple[int, float]]]:
        """Identify most important features based on activation * gradient."""
        
        important_features = {}
        
        for layer_idx in range(self.model.num_layers):
            # MLP features
            mlp_key = f"blocks.{layer_idx}.mlp"
            if mlp_key in activations:
                mlp_acts = activations[mlp_key][0, target_token_idx, :]  # [d_mlp]
                
                # Compute importance scores (activation magnitude for now)
                importance_scores = mlp_acts.abs()
                
                # Get top-k features
                top_k_values, top_k_indices = torch.topk(importance_scores, 
                                                       min(self.top_k_features, len(importance_scores)))
                
                important_features[mlp_key] = [
                    (idx.item(), val.item()) 
                    for idx, val in zip(top_k_indices, top_k_values)
                ]
        
        return important_features
    
    def _create_nodes(
        self,
        important_features: Dict[str, List[Tuple[int, float]]],
        activations: Dict[str, torch.Tensor],
        target_token_idx: int
    ) -> Dict[str, AttributionNode]:
        """Create attribution nodes from important features."""
        
        nodes = {}
        
        # Input node
        input_node_id = f"input_{target_token_idx}"
        nodes[input_node_id] = AttributionNode(
            node_id=input_node_id,
            layer=0,
            position=target_token_idx,
            feature_type="input",
            activation_value=1.0,
            attribution_score=1.0,
            interpretation="Input token"
        )
        
        # Feature nodes
        for layer_name, features in important_features.items():
            layer_idx = int(layer_name.split('.')[1])
            
            for feature_idx, activation_val in features:
                node_id = f"{layer_name}_feat_{feature_idx}"
                
                nodes[node_id] = AttributionNode(
                    node_id=node_id,
                    layer=layer_idx,
                    position=target_token_idx,
                    feature_type="mlp",
                    activation_value=activation_val,
                    attribution_score=activation_val,  # Simplified for now
                    interpretation=f"MLP feature {feature_idx} in layer {layer_idx}"
                )
        
        # Output node
        output_node_id = f"output_{target_token_idx}"
        nodes[output_node_id] = AttributionNode(
            node_id=output_node_id,
            layer=self.model.num_layers,
            position=target_token_idx,
            feature_type="output",
            activation_value=1.0,
            attribution_score=1.0,
            interpretation="Output logits"
        )
        
        return nodes
    
    def _create_edges(
        self,
        nodes: Dict[str, AttributionNode],
        activations: Dict[str, torch.Tensor],
        gradients: Dict[str, torch.Tensor],
        target_token_idx: int
    ) -> List[AttributionEdge]:
        """Create attribution edges between nodes."""
        
        edges = []
        
        # Sort nodes by layer for sequential connections
        nodes_by_layer = defaultdict(list)
        for node in nodes.values():
            nodes_by_layer[node.layer].append(node)
        
        # Create edges between consecutive layers
        for layer in sorted(nodes_by_layer.keys())[:-1]:
            current_layer_nodes = nodes_by_layer[layer]
            next_layer_nodes = nodes_by_layer[layer + 1]
            
            for curr_node in current_layer_nodes:
                for next_node in next_layer_nodes:
                    # Compute edge weight (simplified)
                    weight = self._compute_edge_weight(curr_node, next_node, activations)
                    
                    if abs(weight) >= self.pruning_threshold:  # Fixed: was self.prune_threshold
                        edges.append(AttributionEdge(
                            source=curr_node.node_id,
                            target=next_node.node_id,
                            weight=weight,
                            attribution_type="direct",
                            confidence=min(abs(weight), 1.0)
                        ))
        
        return edges
    
    def _compute_edge_weight(
        self,
        source_node: AttributionNode,
        target_node: AttributionNode,
        activations: Dict[str, torch.Tensor]
    ) -> float:
        """Compute edge weight between two nodes."""
        
        # Simplified edge weight computation
        # In practice, this would use more sophisticated attribution methods
        
        source_activation = abs(source_node.activation_value)
        target_activation = abs(target_node.activation_value)
        
        # Simple multiplicative interaction
        weight = source_activation * target_activation * 0.1
        
        # Add some randomness for demonstration
        weight += np.random.normal(0, 0.01)
        
        return weight
    
    def build_comparison_graph(
        self,
        faithful_example: str,
        unfaithful_example: str,
        target_token: str
    ) -> Dict[str, AttributionGraph]:
        """
        Build attribution graphs for faithful vs unfaithful examples.
        
        Args:
            faithful_example: Example with faithful reasoning
            unfaithful_example: Example with unfaithful reasoning
            target_token: Token to analyze in both examples
            
        Returns:
            Dictionary with 'faithful' and 'unfaithful' graphs
        """
        
        results = {}
        
        for label, example in [("faithful", faithful_example), ("unfaithful", unfaithful_example)]:
            # Tokenize
            tokens = self.model.tokenize(example)
            input_ids = tokens["input_ids"]
            
            # Find target token position
            target_idx = self._find_token_position(input_ids, target_token)
            
            if target_idx is not None:
                # Build attribution graph
                graph = self.build_attribution_graph(input_ids, target_idx)
                graph.metadata['example_type'] = label
                graph.metadata['example_text'] = example
                
                results[label] = graph
            else:
                print(f"Warning: Target token '{target_token}' not found in {label} example")
        
        return results
    
    def _find_token_position(self, input_ids: torch.Tensor, target_token: str) -> Optional[int]:
        """Find position of target token in sequence."""
        tokens = self.model.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        for i, token in enumerate(tokens):
            if target_token.lower() in token.lower():
                return i
        
        return None
    
    def save_graph(self, graph: AttributionGraph, path: str) -> None:
        """Save attribution graph to file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format
        graph_data = {
            'nodes': {
                nid: {
                    'node_id': node.node_id,
                    'layer': node.layer,
                    'position': node.position,
                    'feature_type': node.feature_type,
                    'activation_value': node.activation_value,
                    'attribution_score': node.attribution_score,
                    'interpretation': node.interpretation,
                    'examples': node.examples
                }
                for nid, node in graph.nodes.items()
            },
            'edges': [
                {
                    'source': edge.source,
                    'target': edge.target,
                    'weight': edge.weight,
                    'attribution_type': edge.attribution_type,
                    'confidence': edge.confidence
                }
                for edge in graph.edges
            ],
            'metadata': graph.metadata,
            'pruning_threshold': graph.pruning_threshold
        }
        
        with open(path, 'w') as f:
            json.dump(graph_data, f, indent=2)
    
    def load_graph(self, path: str) -> AttributionGraph:
        """Load attribution graph from file."""
        with open(path, 'r') as f:
            graph_data = json.load(f)
        
        # Reconstruct nodes
        nodes = {}
        for nid, node_data in graph_data['nodes'].items():
            nodes[nid] = AttributionNode(**node_data)
        
        # Reconstruct edges
        edges = [AttributionEdge(**edge_data) for edge_data in graph_data['edges']]
        
        return AttributionGraph(
            nodes=nodes,
            edges=edges,
            metadata=graph_data['metadata'],
            pruning_threshold=graph_data['pruning_threshold']
        )
    
    def build_graph_from_cache(
        self,
        cache,
        reasoning_step: str = "reasoning",
        target_layers: Optional[List[int]] = None,
        target_position: Optional[int] = None
    ) -> AttributionGraph:
        """
        Build attribution graph from existing activation cache.
        
        Args:
            cache: Activation cache from model forward pass
            reasoning_step: Description of the reasoning step
            target_layers: Specific layers to analyze (default: all layers)
            target_position: Specific token position to focus on (default: last position)
            
        Returns:
            Attribution graph built from cache
        """
        if target_layers is None:
            target_layers = list(range(self.model.model.cfg.n_layers))
        
        if target_position is None:
            # Use the last position as target
            target_position = -1
        
        # Extract activations from cache for target layers
        nodes = []
        edges = []
        
        # Create nodes from MLP activations
        for layer_idx in target_layers:
            mlp_key = f'blocks.{layer_idx}.mlp.hook_post'
            if mlp_key in cache.activations:
                # Shape: [batch=1, seq_len, d_mlp]
                activations = cache.activations[mlp_key][0]  # Remove batch dim
                
                if target_position == -1:
                    target_pos = activations.shape[0] - 1  # Last position
                else:
                    target_pos = min(target_position, activations.shape[0] - 1)
                
                # Get activation at target position
                activation_vector = activations[target_pos]  # Shape: [d_mlp]
                activation_magnitude = torch.norm(activation_vector).item()
                
                # Create node
                ctype = "mlp"
                strength = activation_magnitude
                layer = layer_idx
                # Define a stable unique node ID
                node_id = f"L{layer_idx}-P{target_pos}-{ctype}"

                node = self._make_node(
                    layer_idx=layer,
                    position=target_pos,
                    component_type=ctype,
                    activation_strength=strength,
                    node_id=node_id,
                    feature_type=ctype
                )
                nodes.append(node)
        
        # Create simple edges between consecutive layers
        for i in range(len(nodes) - 1):
            source_node = nodes[i]
            target_node = nodes[i + 1]
            
            # Simple attribution strength based on activation difference
            attribution_strength = abs(target_node.activation_strength - source_node.activation_strength)
            
            edge = AttributionEdge(
                source=source_node,
                target=target_node,
                attribution_strength=attribution_strength,
                attribution_type="layer_transfer"
            )
            edges.append(edge)
        
        # Create attribution graph
        attribution_graph = AttributionGraph(
            nodes=nodes,
            edges=edges,
            reasoning_step=reasoning_step
        )
        
        return attribution_graph

class FaithfulnessAnalyzer:
    """
    Analyzes attribution graphs to determine faithfulness patterns.
    """
    
    def __init__(self, graph_builder: AttributionGraphBuilder):
        self.graph_builder = graph_builder
    
    def analyze_faithfulness(
        self,
        prompt: str,
        reasoning_steps: List[str],
        target_tokens: List[str]
    ) -> Dict[str, Any]:
        """
        Analyze faithfulness of chain-of-thought reasoning.
        
        Args:
            prompt: Original prompt
            reasoning_steps: List of reasoning steps
            target_tokens: Tokens to analyze in each step
            
        Returns:
            Faithfulness analysis results
        """
        
        results = {
            'prompt': prompt,
            'reasoning_steps': reasoning_steps,
            'target_tokens': target_tokens,
            'step_analyses': [],
            'overall_faithfulness': None,
            'patterns_detected': []
        }
        
        # Analyze each reasoning step
        for i, (step, token) in enumerate(zip(reasoning_steps, target_tokens)):
            step_analysis = self._analyze_single_step(prompt + " " + step, token, i)
            results['step_analyses'].append(step_analysis)
        
        # Determine overall faithfulness
        results['overall_faithfulness'] = self._determine_overall_faithfulness(
            results['step_analyses']
        )
        
        # Detect patterns
        results['patterns_detected'] = self._detect_patterns(results['step_analyses'])
        
        return results
    
    def _analyze_single_step(self, text: str, target_token: str, step_idx: int) -> Dict[str, Any]:
        """Analyze a single reasoning step."""
        
        # Tokenize
        tokens = self.graph_builder.model.tokenize(text)
        input_ids = tokens["input_ids"]
        
        # Find target token
        target_idx = self.graph_builder._find_token_position(input_ids, target_token)
        
        if target_idx is None:
            return {
                'step_idx': step_idx,
                'target_token': target_token,
                'faithfulness_score': 0.0,
                'confidence': 0.0,
                'error': f"Target token '{target_token}' not found"
            }
        
        # Build attribution graph
        graph = self.graph_builder.build_attribution_graph(input_ids, target_idx)
        
        # Analyze graph properties
        faithfulness_score = self._compute_faithfulness_score(graph)
        confidence = self._compute_confidence(graph)
        pattern_type = self._classify_pattern(graph)
        
        return {
            'step_idx': step_idx,
            'target_token': target_token,
            'target_position': target_idx,
            'faithfulness_score': faithfulness_score,
            'confidence': confidence,
            'pattern_type': pattern_type,
            'graph': graph,
            'attribution_summary': graph.get_layer_summary()
        }
    
    def _compute_faithfulness_score(self, graph: AttributionGraph) -> float:
        """Compute faithfulness score based on graph properties."""
        
        # Simple heuristic: more direct paths = more faithful
        output_nodes = [n for n in graph.nodes.values() if n.feature_type == "output"]
        if not output_nodes:
            return 0.0
        
        output_node = output_nodes[0]
        paths = graph.get_paths_to_output(output_node.node_id)
        
        if not paths:
            return 0.0
        
        # Score based on path diversity and directness
        avg_path_length = np.mean([len(path) for path in paths])
        path_diversity = len(set(tuple(path) for path in paths)) / max(len(paths), 1)
        
        # Higher score for moderate path length and high diversity
        length_score = max(0, 1 - (avg_path_length - 3) / 5)  # Optimal around 3 steps
        diversity_score = path_diversity
        
        return (length_score + diversity_score) / 2
    
    def _compute_confidence(self, graph: AttributionGraph) -> float:
        """Compute confidence in the analysis."""
        
        # Confidence based on graph connectivity and attribution strengths
        total_nodes = len(graph.nodes)
        total_edges = len(graph.edges)
        
        if total_nodes == 0:
            return 0.0
        
        connectivity = total_edges / (total_nodes * (total_nodes - 1) / 2) if total_nodes > 1 else 0
        
        # Average attribution strength
        avg_attribution = np.mean([abs(edge.weight) for edge in graph.edges]) if graph.edges else 0
        
        return min((connectivity + avg_attribution) / 2, 1.0)
    
    def _classify_pattern(self, graph: AttributionGraph) -> str:
        """Classify the reasoning pattern from the graph."""
        
        output_nodes = [n for n in graph.nodes.values() if n.feature_type == "output"]
        if not output_nodes:
            return "unknown"
        
        output_node = output_nodes[0]
        paths = graph.get_paths_to_output(output_node.node_id)
        
        if not paths:
            return "disconnected"
        
        # Analyze path characteristics
        avg_path_length = np.mean([len(path) for path in paths])
        
        # Simple classification based on path properties
        if avg_path_length <= 2:
            return "shortcut"
        elif avg_path_length >= 5:
            return "complex_reasoning"
        else:
            # Check for backward flow (simplified)
            has_backward_flow = any(
                graph.nodes[path[i]].layer > graph.nodes[path[i+1]].layer
                for path in paths
                for i in range(len(path)-1)
            )
            
            if has_backward_flow:
                return "backward_chaining"
            else:
                return "faithful"
    
    def _determine_overall_faithfulness(self, step_analyses: List[Dict[str, Any]]) -> str:
        """Determine overall faithfulness from step analyses."""
        
        if not step_analyses:
            return "unknown"
        
        # Average faithfulness score
        avg_score = np.mean([step.get('faithfulness_score', 0) for step in step_analyses])
        
        if avg_score >= 0.7:
            return "faithful"
        elif avg_score >= 0.4:
            return "partially_faithful"
        else:
            return "unfaithful"
    
    def _detect_patterns(self, step_analyses: List[Dict[str, Any]]) -> List[str]:
        """Detect common patterns across reasoning steps."""
        
        patterns = []
        pattern_counts = defaultdict(int)
        
        for step in step_analyses:
            pattern_type = step.get('pattern_type', 'unknown')
            pattern_counts[pattern_type] += 1
        
        total_steps = len(step_analyses)
        
        for pattern, count in pattern_counts.items():
            if count / total_steps >= 0.5:  # Pattern appears in >50% of steps
                patterns.append(f"dominant_{pattern}")
            elif count / total_steps >= 0.3:  # Pattern appears in >30% of steps
                patterns.append(f"frequent_{pattern}")
        
        return patterns
