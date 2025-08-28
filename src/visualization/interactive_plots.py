"""
Interactive visualization for attribution graphs and faithfulness analysis.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import torch
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

from analysis.attribution_graphs import AttributionGraph, AttributionNode, AttributionEdge
from analysis.faithfulness_detector import FaithfulnessDetector, DetectionFeatures
from interventions.targeted_interventions import InterventionResult

class AttributionGraphVisualizer:
    """
    Interactive visualization tools for attribution graphs and faithfulness analysis.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        self.color_palette = sns.color_palette("husl", 8)
        
        # Set style - with fallback for newer seaborn versions
        try:
            plt.style.use('seaborn-v0_8')
        except OSError:
            # Fallback for newer matplotlib/seaborn versions
            plt.style.use('seaborn')
        except:
            # Final fallback
            plt.style.use('default')
        
        sns.set_palette("husl")
    
    def plot_attribution_graph(
        self,
        graph: AttributionGraph,
        layout: str = "spring",
        highlight_critical: bool = True,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create an interactive plot of the attribution graph.
        
        Args:
            graph: Attribution graph to visualize
            layout: Graph layout algorithm ('spring', 'circular', 'kamada_kawai')
            highlight_critical: Whether to highlight critical nodes/edges
            save_path: Optional path to save the plot
            
        Returns:
            Plotly figure object
        """
        
        # Convert to NetworkX graph for layout
        nx_graph = self._attribution_to_networkx(graph)
        
        # Calculate layout
        if layout == "spring":
            pos = nx.spring_layout(nx_graph, k=3, iterations=50)
        elif layout == "circular":
            pos = nx.circular_layout(nx_graph)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(nx_graph)
        else:
            pos = nx.spring_layout(nx_graph)
        
        # Prepare node data
        node_data = self._prepare_node_data(graph, pos, highlight_critical)
        edge_data = self._prepare_edge_data(graph, pos, highlight_critical)
        
        # Create plotly figure
        fig = go.Figure()
        
        # Add edges
        for edge_trace in edge_data:
            fig.add_trace(edge_trace)
        
        # Add nodes
        fig.add_trace(node_data)
        
        # Update layout
        fig.update_layout(
            title=f"Attribution Graph - {graph.reasoning_step}",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[
                dict(
                    text="Node size: Activation strength | Edge width: Attribution strength",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002, 
                    xanchor="left", yanchor="bottom",
                    font=dict(size=12)
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=1000,
            height=700
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_faithfulness_trends(
        self,
        detector: FaithfulnessDetector,
        examples: List[Dict[str, Any]],
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Plot faithfulness trends across different reasoning types and difficulties.
        
        Args:
            detector: Trained faithfulness detector
            examples: List of reasoning examples with predictions
            save_path: Optional path to save the plot
            
        Returns:
            Plotly figure object
        """
        
        # Prepare data
        df_data = []
        for example in examples:
            df_data.append({
                'reasoning_type': example.get('reasoning_type', 'unknown'),
                'difficulty': example.get('difficulty_level', 'medium'),
                'is_faithful': example.get('is_faithful', False),
                'predicted_faithful': example.get('predicted_faithful', False),
                'confidence': example.get('confidence', 0.5),
                'faithfulness_score': example.get('faithfulness_score', 0.5)
            })
        
        df = pd.DataFrame(df_data)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Faithfulness by Reasoning Type',
                'Confidence Distribution',
                'Difficulty vs Faithfulness',
                'Prediction Accuracy'
            ],
            specs=[[{"type": "bar"}, {"type": "histogram"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # Plot 1: Faithfulness by reasoning type
        type_counts = df.groupby(['reasoning_type', 'is_faithful']).size().unstack(fill_value=0)
        
        fig.add_trace(
            go.Bar(
                x=type_counts.index,
                y=type_counts.get(True, [0]*len(type_counts)),
                name='Faithful',
                marker_color='green'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=type_counts.index,
                y=type_counts.get(False, [0]*len(type_counts)),
                name='Unfaithful',
                marker_color='red'
            ),
            row=1, col=1
        )
        
        # Plot 2: Confidence distribution
        fig.add_trace(
            go.Histogram(
                x=df[df['is_faithful']]['confidence'],
                name='Faithful Examples',
                opacity=0.7,
                marker_color='green'
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Histogram(
                x=df[~df['is_faithful']]['confidence'],
                name='Unfaithful Examples',
                opacity=0.7,
                marker_color='red'
            ),
            row=1, col=2
        )
        
        # Plot 3: Difficulty vs Faithfulness
        fig.add_trace(
            go.Scatter(
                x=df['difficulty'],
                y=df['faithfulness_score'],
                mode='markers',
                marker=dict(
                    color=df['is_faithful'].map({True: 'green', False: 'red'}),
                    size=df['confidence'] * 20,
                    opacity=0.6
                ),
                text=df['reasoning_type'],
                name='Examples'
            ),
            row=2, col=1
        )
        
        # Plot 4: Prediction accuracy
        df['correct_prediction'] = df['is_faithful'] == df['predicted_faithful']
        accuracy_by_type = df.groupby('reasoning_type')['correct_prediction'].mean()
        
        fig.add_trace(
            go.Bar(
                x=accuracy_by_type.index,
                y=accuracy_by_type.values,
                marker_color='blue',
                name='Accuracy'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="Faithfulness Analysis Dashboard",
            showlegend=True,
            height=800,
            width=1200
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_intervention_effects(
        self,
        intervention_results: Dict[str, List[InterventionResult]],
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Visualize the effects of different interventions.
        
        Args:
            intervention_results: Results from intervention experiments
            save_path: Optional path to save the plot
            
        Returns:
            Plotly figure object
        """
        
        # Prepare data
        plot_data = []
        for method, results in intervention_results.items():
            for result in results:
                plot_data.append({
                    'method': method,
                    'strength': abs(result.intervention_strength),
                    'faithfulness_change': result.faithfulness_change,
                    'accuracy_preserved': result.accuracy_preserved,
                    'confidence': result.confidence_score,
                    'direction': 'increase' if result.intervention_strength > 0 else 'decrease'
                })
        
        df = pd.DataFrame(plot_data)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Faithfulness Change vs Intervention Strength',
                'Method Effectiveness',
                'Accuracy Preservation by Method',
                'Confidence vs Faithfulness Change'
            ]
        )
        
        # Plot 1: Strength vs Change
        for method in df['method'].unique():
            method_data = df[df['method'] == method]
            
            fig.add_trace(
                go.Scatter(
                    x=method_data['strength'],
                    y=method_data['faithfulness_change'],
                    mode='markers+lines',
                    name=method,
                    marker=dict(size=8),
                    line=dict(width=2)
                ),
                row=1, col=1
            )
        
        # Plot 2: Method effectiveness
        method_effectiveness = df.groupby('method')['faithfulness_change'].agg(['mean', 'std'])
        
        fig.add_trace(
            go.Bar(
                x=method_effectiveness.index,
                y=method_effectiveness['mean'],
                error_y=dict(type='data', array=method_effectiveness['std']),
                marker_color='lightblue',
                name='Avg Change'
            ),
            row=1, col=2
        )
        
        # Plot 3: Accuracy preservation
        accuracy_by_method = df.groupby('method')['accuracy_preserved'].mean()
        
        fig.add_trace(
            go.Bar(
                x=accuracy_by_method.index,
                y=accuracy_by_method.values,
                marker_color='lightgreen',
                name='Accuracy Rate'
            ),
            row=2, col=1
        )
        
        # Plot 4: Confidence vs Change
        fig.add_trace(
            go.Scatter(
                x=df['confidence'],
                y=df['faithfulness_change'],
                mode='markers',
                marker=dict(
                    color=df['method'].astype('category').cat.codes,
                    size=df['strength'] * 5,
                    opacity=0.7,
                    colorscale='viridis'
                ),
                text=df['method'],
                name='Interventions'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="Intervention Effects Analysis",
            showlegend=True,
            height=800,
            width=1200
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Intervention Strength", row=1, col=1)
        fig.update_yaxes(title_text="Faithfulness Change", row=1, col=1)
        
        fig.update_xaxes(title_text="Method", row=1, col=2)
        fig.update_yaxes(title_text="Avg Faithfulness Change", row=1, col=2)
        
        fig.update_xaxes(title_text="Method", row=2, col=1)
        fig.update_yaxes(title_text="Accuracy Preservation Rate", row=2, col=1)
        
        fig.update_xaxes(title_text="Confidence Score", row=2, col=2)
        fig.update_yaxes(title_text="Faithfulness Change", row=2, col=2)
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_activation_heatmap(
        self,
        activations: torch.Tensor,
        layer_names: List[str],
        sequence_tokens: List[str],
        title: str = "Activation Heatmap",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a heatmap of model activations.
        
        Args:
            activations: Tensor of activations [layers, seq_len, hidden_dim]
            layer_names: Names of the layers
            sequence_tokens: Tokens in the sequence
            title: Plot title
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        
        # Aggregate activations (mean across hidden dimension)
        if len(activations.shape) == 3:
            agg_activations = activations.mean(dim=-1)  # [layers, seq_len]
        else:
            agg_activations = activations
        
        # Convert to numpy
        agg_activations = agg_activations.detach().cpu().numpy()
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create heatmap
        im = ax.imshow(
            agg_activations,
            cmap='RdYlBu_r',
            aspect='auto',
            interpolation='nearest'
        )
        
        # Set ticks and labels
        ax.set_xticks(range(len(sequence_tokens)))
        ax.set_xticklabels(sequence_tokens, rotation=45, ha='right')
        ax.set_yticks(range(len(layer_names)))
        ax.set_yticklabels(layer_names)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Activation Strength')
        
        # Set title and labels
        ax.set_title(title)
        ax.set_xlabel('Tokens')
        ax.set_ylabel('Layers')
        
        # Adjust layout
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_faithfulness_dashboard(
        self,
        graph: AttributionGraph,
        detector_results: Dict[str, Any],
        intervention_results: Dict[str, List[InterventionResult]],
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create a comprehensive dashboard combining all visualizations.
        
        Args:
            graph: Attribution graph
            detector_results: Results from faithfulness detection
            intervention_results: Results from interventions
            save_path: Optional path to save the dashboard
            
        Returns:
            Plotly figure object
        """
        
        # Create subplots with custom specs
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Attribution Graph',
                'Faithfulness Score Distribution',
                'Feature Importance',
                'Intervention Effects',
                'Reasoning Step Analysis',
                'Model Confidence'
            ],
            specs=[
                [{"type": "scatter"}, {"type": "histogram"}],
                [{"type": "bar"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "scatter"}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.1
        )
        
        # Add attribution graph (simplified)
        nx_graph = self._attribution_to_networkx(graph)
        pos = nx.spring_layout(nx_graph)
        
        # Extract node positions
        node_x = [pos[node][0] for node in nx_graph.nodes()]
        node_y = [pos[node][1] for node in nx_graph.nodes()]
        node_text = [f"Layer {node.layer_idx}<br>Pos {node.position}" for node in graph.nodes]
        
        fig.add_trace(
            go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                marker=dict(size=10, color='lightblue'),
                text=[str(i) for i in range(len(node_x))],
                textposition="middle center",
                hovertext=node_text,
                name='Nodes'
            ),
            row=1, col=1
        )
        
        # Add other plots...
        # (Implementation would continue with adding traces for each subplot)
        
        # Update layout
        fig.update_layout(
            title="Faithfulness Analysis Dashboard",
            showlegend=False,
            height=1200,
            width=1400
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def _get_node_positions(self, graph: AttributionGraph, layout: str = "spring") -> Dict:
        """Get node positions for layout."""
        import networkx as nx
        
        # Create NetworkX graph
        G = nx.DiGraph()
        
        # Handle both dict and list formats for graph.nodes
        if isinstance(graph.nodes, dict):
            # Your format: {node_id: node_object}
            for node_id, node in graph.nodes.items():
                layer = getattr(node, 'layer_idx', 0)
                G.add_node(node_id, layer=layer)
        else:
            # Expected format: [node_object, ...]
            for i, node in enumerate(graph.nodes):
                layer = getattr(node, 'layer_idx', 0)
                G.add_node(i, layer=layer)
        
        # Add edges
        for edge in graph.edges:
            source = getattr(edge, 'source', None)
            target = getattr(edge, 'target', None)
            weight = abs(getattr(edge, 'weight', 0))
            
            if source in G.nodes and target in G.nodes:
                G.add_edge(source, target, weight=weight)
        
        # Generate layout
        if layout == "spring":
            try:
                pos = nx.spring_layout(G, k=3, iterations=50)
            except:
                pos = nx.random_layout(G)
        elif layout == "hierarchical":
            try:
                pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
            except:
                pos = nx.spring_layout(G, k=3, iterations=50)
        else:
            pos = nx.random_layout(G)
        
        return pos
    
    def _attribution_to_networkx(self, graph) -> nx.DiGraph:
        """Convert attribution graph to NetworkX with dict/list + ID/object edge support."""
        G = nx.DiGraph()

        # Normalize nodes
        if isinstance(graph.nodes, dict):
            node_ids = list(graph.nodes.keys())
            node_objs = list(graph.nodes.values())
        else:
            node_ids = list(range(len(graph.nodes)))
            node_objs = list(graph.nodes)

        id_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}
        obj_to_idx = {id(obj): i for i, obj in enumerate(node_objs) if not isinstance(obj, (str, int))}

        # Add nodes
        for i, obj in enumerate(node_objs):
            layer = getattr(obj, "layer_idx", getattr(obj, "layer", 0))
            position = getattr(obj, "position", getattr(obj, "pos", 0))
            component = getattr(obj, "component_type", getattr(obj, "feature_type", "unknown"))
            activation = abs(getattr(obj, "activation_strength", getattr(obj, "weight", 0)))
            G.add_node(
                i,
                original_id=node_ids[i],
                layer=layer,
                position=position,
                component=component,
                activation=activation,
            )

        # Add edges
        for e in getattr(graph, "edges", []):
            src = getattr(e, "source", None)
            tgt = getattr(e, "target", None)

            s_idx = id_to_idx.get(src)
            t_idx = id_to_idx.get(tgt)

            if s_idx is None and src is not None:
                s_idx = obj_to_idx.get(id(src))
            if t_idx is None and tgt is not None:
                t_idx = obj_to_idx.get(id(tgt))

            if s_idx is None or t_idx is None:
                continue

            weight = abs(getattr(e, "attribution_strength", getattr(e, "weight", 0)))
            attr_type = getattr(e, "attribution_type", "unknown")
            G.add_edge(s_idx, t_idx, weight=weight, attribution_type=attr_type)

        return G
    
    def _prepare_node_data(
        self, 
        graph: AttributionGraph, 
        pos: Dict, 
        highlight_critical: bool
    ) -> go.Scatter:
        """Prepare node data for plotting."""
        
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        node_size = []
        
        # Handle both dict and list formats consistently
        if isinstance(graph.nodes, dict):
            nodes = list(graph.nodes.values())
        else:
            nodes = graph.nodes
        
        for i, node in enumerate(nodes):
            x, y = pos[i]
            node_x.append(x)
            node_y.append(y)
            
            # Create hover text with safe attribute access
            text = f"Layer {getattr(node, 'layer_idx', 'Unknown')}<br>"
            text += f"Position {getattr(node, 'position', 'Unknown')}<br>"
            text += f"Component: {getattr(node, 'component_type', 'Unknown')}<br>"
            text += f"Activation: {getattr(node, 'activation_strength', 0):.3f}"
            node_text.append(text)
            
            # Color based on component type
            color_map = {'attention': 'red', 'mlp': 'blue', 'embedding': 'green'}
            component_type = getattr(node, 'component_type', 'unknown')
            node_color.append(color_map.get(component_type, 'gray'))
            
            # Size based on activation strength
            activation = abs(getattr(node, 'activation_strength', 0))
            size = max(10, min(50, activation * 30))
            node_size.append(size)
        
        return go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=2, color='black'),
                opacity=0.8
            ),
            name="Nodes"
        )

    def _prepare_edge_data(
        self,
        graph: AttributionGraph,
        pos: Dict,
        highlight_critical: bool
    ) -> List[go.Scatter]:
        """Prepare edge data for plotting."""
        edge_traces = []

        # Build a robust mapping from node identifiers/objects to indices
        if isinstance(graph.nodes, dict):
            node_keys = list(graph.nodes.keys())
            id_to_idx = {k: i for i, k in enumerate(node_keys)}
        else:
            id_to_idx = {node: i for i, node in enumerate(graph.nodes)}

        for edge in graph.edges:
            src = getattr(edge, 'source', None)
            tgt = getattr(edge, 'target', None)
            source_idx = id_to_idx.get(src)
            target_idx = id_to_idx.get(tgt)
            if source_idx is None or target_idx is None:
                continue

            x0, y0 = pos[source_idx]
            x1, y1 = pos[target_idx]

            # Edge color and width based on attribution strength
            strength = abs(getattr(edge, 'attribution_strength', getattr(edge, 'weight', 0)))
            width = max(1, min(10, strength * 20))
            sign = getattr(edge, 'attribution_strength', 0)
            color = 'green' if sign > 0 else 'red'
            opacity = min(1.0, max(0.1, strength * 2))

            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=width, color=color),
                opacity=opacity,
                hoverinfo='none',
                showlegend=False
            )
            edge_traces.append(edge_trace)

        return edge_traces

    def create_hierarchical_flow_visualization(self, graph: AttributionGraph) -> go.Figure:
        """Fallback implementation to avoid missing method errors."""
        return self.plot_attribution_graph(graph, layout="spring")

    def save_all_visualizations(
        self,
        graph: AttributionGraph,
        detector_results: Dict[str, Any],
        intervention_results: Dict[str, List[InterventionResult]],
        output_dir: str
    ) -> None:
        """
        Save all visualizations to specified directory.
        
        Args:
            graph: Attribution graph
            detector_results: Faithfulness detection results
            intervention_results: Intervention experiment results
            output_dir: Directory to save visualizations
        """
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save attribution graph
        graph_fig = self.plot_attribution_graph(graph)
        graph_fig.write_html(output_path / "attribution_graph.html")
        
        # Save intervention effects
        intervention_fig = self.plot_intervention_effects(intervention_results)
        intervention_fig.write_html(output_path / "intervention_effects.html")
        
        # Save comprehensive dashboard
        dashboard_fig = self.create_faithfulness_dashboard(
            graph, detector_results, intervention_results
        )
        dashboard_fig.write_html(output_path / "faithfulness_dashboard.html")
        
        # Save hierarchical flow visualization
        flow_fig = self.create_hierarchical_flow_visualization(graph)
        flow_fig.write_html(output_path / "hierarchical_flow_visualization.html")
        
        print(f"All visualizations saved to {output_dir}")
