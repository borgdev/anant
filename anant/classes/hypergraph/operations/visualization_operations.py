"""
Visualization operations for Hypergraph

Handles layout generation, coordinate calculation, and visualization support
for various layout algorithms including spring, circular, random, and bipartite.
"""

from typing import Dict, Tuple, Any, Optional
import math
import random
from ....exceptions import HypergraphError, ValidationError


class VisualizationOperations:
    """
    Visualization operations for hypergraph
    
    Provides layout generation algorithms and coordinate calculation
    for visualizing hypergraphs using various layout techniques.
    """
    
    def __init__(self, hypergraph):
        """
        Initialize VisualizationOperations
        
        Parameters
        ----------
        hypergraph : Hypergraph
            Parent hypergraph instance
        """
        if hypergraph is None:
            raise HypergraphError("Hypergraph instance cannot be None")
        self.hypergraph = hypergraph
    
    def get_layout_coordinates(self, layout_type: str = "spring", **kwargs) -> Dict[Any, Tuple[float, float]]:
        """
        Generate layout coordinates for visualization
        
        Parameters
        ----------
        layout_type : str, default "spring"
            Type of layout algorithm to use:
            - 'spring': Force-directed spring layout
            - 'circular': Nodes arranged in a circle
            - 'random': Random node positions
            - 'bipartite': Two-column layout based on node degrees
        **kwargs
            Additional layout parameters:
            - radius (float): Radius for circular layout
            - width, height (float): Dimensions for random layout
            - seed (int): Random seed for reproducibility
            - iterations (int): Number of iterations for spring layout
            - k (float): Spring constant for force calculations
            
        Returns
        -------
        Dict[Any, Tuple[float, float]]
            Dictionary mapping node IDs to (x, y) coordinates
            
        Raises
        ------
        ValidationError
            If layout_type is not supported or parameters are invalid
        HypergraphError
            If layout generation fails
        """
        if not isinstance(layout_type, str):
            raise ValidationError("layout_type must be a string")
        
        supported_layouts = {'spring', 'circular', 'random', 'bipartite'}
        if layout_type not in supported_layouts:
            raise ValidationError(f"Unsupported layout type: {layout_type}. "
                                f"Supported types: {supported_layouts}")
        
        try:
            nodes = list(self.hypergraph.nodes)
            n = len(nodes)
            
            if n == 0:
                return {}
            
            if layout_type == "circular":
                return self._circular_layout(nodes, **kwargs)
            elif layout_type == "random":
                return self._random_layout(nodes, **kwargs)
            elif layout_type == "bipartite":
                return self._bipartite_layout(nodes, **kwargs)
            else:  # spring layout (default)
                return self._spring_layout(nodes, **kwargs)
                
        except Exception as e:
            raise HypergraphError(f"Layout generation failed for {layout_type}: {e}")
    
    def _circular_layout(self, nodes: list, **kwargs) -> Dict[Any, Tuple[float, float]]:
        """
        Generate circular layout coordinates
        
        Parameters
        ----------
        nodes : list
            List of nodes to position
        **kwargs
            radius (float): Circle radius, default 1.0
            
        Returns
        -------
        Dict[Any, Tuple[float, float]]
            Node coordinates
        """
        coordinates = {}
        n = len(nodes)
        radius = kwargs.get('radius', 1.0)
        
        if radius <= 0:
            raise ValidationError("radius must be positive")
        
        angle_step = 2 * math.pi / n if n > 0 else 0
        
        for i, node in enumerate(nodes):
            angle = i * angle_step
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            coordinates[node] = (x, y)
        
        return coordinates
    
    def _random_layout(self, nodes: list, **kwargs) -> Dict[Any, Tuple[float, float]]:
        """
        Generate random layout coordinates
        
        Parameters
        ----------
        nodes : list
            List of nodes to position
        **kwargs
            width, height (float): Layout dimensions, default 2.0
            seed (int): Random seed, default 42
            
        Returns
        -------
        Dict[Any, Tuple[float, float]]
            Node coordinates
        """
        width = kwargs.get('width', 2.0)
        height = kwargs.get('height', 2.0)
        seed = kwargs.get('seed', 42)
        
        if width <= 0 or height <= 0:
            raise ValidationError("width and height must be positive")
        
        random.seed(seed)
        coordinates = {}
        
        for node in nodes:
            x = random.uniform(-width/2, width/2)
            y = random.uniform(-height/2, height/2)
            coordinates[node] = (x, y)
        
        return coordinates
    
    def _bipartite_layout(self, nodes: list, **kwargs) -> Dict[Any, Tuple[float, float]]:
        """
        Generate bipartite layout coordinates
        
        Separates nodes into two columns based on their degree.
        Higher degree nodes go to the left column.
        
        Parameters
        ----------
        nodes : list
            List of nodes to position
        **kwargs
            separation (float): Distance between columns, default 2.0
            
        Returns
        -------
        Dict[Any, Tuple[float, float]]
            Node coordinates
        """
        separation = kwargs.get('separation', 2.0)
        
        if separation <= 0:
            raise ValidationError("separation must be positive")
        
        # Separate nodes into two sets based on their connections
        left_nodes = []
        right_nodes = []
        
        # Simple heuristic: nodes with more connections go to left
        node_degrees = {}
        for node in nodes:
            node_degrees[node] = len(self.hypergraph.incidences.get_node_edges(node))
        
        sorted_nodes = sorted(nodes, key=lambda x: node_degrees[x], reverse=True)
        
        # Alternate assignment
        for i, node in enumerate(sorted_nodes):
            if i % 2 == 0:
                left_nodes.append(node)
            else:
                right_nodes.append(node)
        
        coordinates = {}
        
        # Position left nodes
        for i, node in enumerate(left_nodes):
            if len(left_nodes) > 1:
                y = (i - (len(left_nodes) - 1)/2) / max(len(left_nodes) - 1, 1)
            else:
                y = 0.0
            coordinates[node] = (-separation/2, y)
        
        # Position right nodes
        for i, node in enumerate(right_nodes):
            if len(right_nodes) > 1:
                y = (i - (len(right_nodes) - 1)/2) / max(len(right_nodes) - 1, 1)
            else:
                y = 0.0
            coordinates[node] = (separation/2, y)
        
        return coordinates
    
    def _spring_layout(self, nodes: list, **kwargs) -> Dict[Any, Tuple[float, float]]:
        """
        Generate spring (force-directed) layout coordinates
        
        Uses a simple force-directed algorithm with repulsive and attractive forces.
        
        Parameters
        ----------
        nodes : list
            List of nodes to position
        **kwargs
            iterations (int): Number of simulation iterations, default 50
            k (float): Spring constant, default 1.0
            damping (float): Damping factor, default 0.1
            seed (int): Random seed for initial positions, default 42
            
        Returns
        -------
        Dict[Any, Tuple[float, float]]
            Node coordinates
        """
        iterations = kwargs.get('iterations', 50)
        k = kwargs.get('k', 1.0)
        damping = kwargs.get('damping', 0.1)
        seed = kwargs.get('seed', 42)
        
        if iterations <= 0:
            raise ValidationError("iterations must be positive")
        if k <= 0:
            raise ValidationError("k must be positive")
        if not 0 < damping <= 1:
            raise ValidationError("damping must be between 0 and 1")
        
        # Initialize random positions
        random.seed(seed)
        coordinates = {}
        for node in nodes:
            coordinates[node] = (random.uniform(-1, 1), random.uniform(-1, 1))
        
        # Force-directed iterations
        for iteration in range(iterations):
            forces = {node: [0.0, 0.0] for node in nodes}
            
            # Repulsive forces between all node pairs
            for i, node1 in enumerate(nodes):
                for j, node2 in enumerate(nodes[i+1:], i+1):
                    x1, y1 = coordinates[node1]
                    x2, y2 = coordinates[node2]
                    
                    dx = x1 - x2
                    dy = y1 - y2
                    distance = math.sqrt(dx*dx + dy*dy) + 1e-6  # Avoid division by zero
                    
                    # Repulsive force (inverse square law)
                    force = k / (distance * distance)
                    fx = force * dx / distance
                    fy = force * dy / distance
                    
                    forces[node1][0] += fx
                    forces[node1][1] += fy
                    forces[node2][0] -= fx
                    forces[node2][1] -= fy
            
            # Attractive forces between connected nodes
            for edge in self.hypergraph.edges:
                edge_nodes = self.hypergraph.incidences.get_edge_nodes(edge)
                
                # For hyperedges, attract all pairs within the edge
                for i, node1 in enumerate(edge_nodes):
                    for j, node2 in enumerate(edge_nodes[i+1:], i+1):
                        x1, y1 = coordinates[node1]
                        x2, y2 = coordinates[node2]
                        
                        dx = x2 - x1
                        dy = y2 - y1
                        distance = math.sqrt(dx*dx + dy*dy) + 1e-6
                        
                        # Attractive force (Hooke's law)
                        force = distance * k
                        fx = force * dx / distance
                        fy = force * dy / distance
                        
                        forces[node1][0] += fx
                        forces[node1][1] += fy
                        forces[node2][0] -= fx
                        forces[node2][1] -= fy
            
            # Update positions with damping
            for node in nodes:
                x, y = coordinates[node]
                fx, fy = forces[node]
                coordinates[node] = (x + fx * damping, y + fy * damping)
        
        return coordinates
    
    def layout(self, algorithm: str = 'spring', **kwargs) -> Dict[Any, Tuple[float, float]]:
        """
        Generate layout coordinates (alias for get_layout_coordinates)
        
        Parameters
        ----------
        algorithm : str, default 'spring'
            Layout algorithm name
        **kwargs
            Layout-specific parameters
            
        Returns
        -------
        Dict[Any, Tuple[float, float]]
            Node coordinates
        """
        return self.get_layout_coordinates(algorithm, **kwargs)
    
    def draw(self, layout: str = "spring", node_size: int = 300, 
             edge_color: str = "gray", node_color: str = "lightblue",
             with_labels: bool = True, figsize: Tuple[int, int] = (10, 8),
             **kwargs) -> None:
        """
        Draw the hypergraph using matplotlib
        
        Note: This is a simplified drawing function that projects hypergraph
        to a regular graph for visualization.
        
        Parameters
        ----------
        layout : str, default "spring"
            Layout algorithm to use
        node_size : int, default 300
            Size of nodes in the visualization
        edge_color : str, default "gray"
            Color of edges
        node_color : str, default "lightblue"
            Color of nodes
        with_labels : bool, default True
            Whether to show node labels
        figsize : Tuple[int, int], default (10, 8)
            Figure size
        **kwargs
            Additional layout parameters
            
        Raises
        ------
        HypergraphError
            If drawing fails or matplotlib is not available
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
        except ImportError:
            raise HypergraphError("matplotlib is required for drawing. Install with: pip install matplotlib")
        
        try:
            # Get layout coordinates
            pos = self.get_layout_coordinates(layout, **kwargs)
            
            if not pos:
                print("No nodes to draw")
                return
            
            # Create figure
            fig, ax = plt.subplots(figsize=figsize)
            
            # Draw hyperedges as polygons/lines
            for edge in self.hypergraph.edges:
                edge_nodes = self.hypergraph.incidences.get_edge_nodes(edge)
                
                if len(edge_nodes) >= 2:
                    # Get positions for edge nodes
                    edge_positions = [pos[node] for node in edge_nodes if node in pos]
                    
                    if len(edge_positions) >= 2:
                        if len(edge_positions) == 2:
                            # Draw as line for binary edges
                            x_coords, y_coords = zip(*edge_positions)
                            ax.plot(x_coords, y_coords, color=edge_color, alpha=0.6, linewidth=1)
                        else:
                            # Draw as polygon for hyperedges
                            polygon = patches.Polygon(edge_positions, closed=True, 
                                                    fill=False, edgecolor=edge_color, 
                                                    alpha=0.3, linewidth=1)
                            ax.add_patch(polygon)
            
            # Draw nodes
            for node, (x, y) in pos.items():
                circle = patches.Circle((x, y), radius=0.1, color=node_color, 
                                      zorder=2, alpha=0.8)
                ax.add_patch(circle)
                
                if with_labels:
                    ax.text(x, y, str(node), ha='center', va='center', 
                           fontsize=8, zorder=3)
            
            # Set equal aspect ratio and adjust limits
            ax.set_aspect('equal')
            if pos:
                x_coords = [x for x, y in pos.values()]
                y_coords = [y for x, y in pos.values()]
                margin = 0.2
                ax.set_xlim(min(x_coords) - margin, max(x_coords) + margin)
                ax.set_ylim(min(y_coords) - margin, max(y_coords) + margin)
            
            ax.set_title(f"Hypergraph: {self.hypergraph.name}")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            raise HypergraphError(f"Drawing failed: {e}")
    
    def get_node_positions_array(self, layout: str = "spring", **kwargs):
        """
        Get node positions as numpy arrays for external visualization
        
        Parameters
        ----------
        layout : str, default "spring"
            Layout algorithm to use
        **kwargs
            Layout parameters
            
        Returns
        -------
        Tuple[list, numpy.ndarray]
            Tuple of (node_ids, positions_array) where positions_array 
            has shape (n_nodes, 2)
            
        Raises
        ------
        HypergraphError
            If numpy is not available or position calculation fails
        """
        try:
            import numpy as np
        except ImportError:
            raise HypergraphError("numpy is required for array positions. Install with: pip install numpy")
        
        try:
            pos = self.get_layout_coordinates(layout, **kwargs)
            
            if not pos:
                return [], np.array([])
            
            nodes = list(pos.keys())
            positions = np.array([pos[node] for node in nodes])
            
            return nodes, positions
            
        except Exception as e:
            raise HypergraphError(f"Failed to get position arrays: {e}")
    
    def export_layout(self, layout: str = "spring", filename: Optional[str] = None, **kwargs) -> Dict[Any, Tuple[float, float]]:
        """
        Export layout coordinates to file or return them
        
        Parameters
        ----------
        layout : str, default "spring"
            Layout algorithm to use
        filename : Optional[str]
            If provided, save coordinates to this file as JSON
        **kwargs
            Layout parameters
            
        Returns
        -------
        Dict[Any, Tuple[float, float]]
            Node coordinates
            
        Raises
        ------
        HypergraphError
            If export fails
        """
        try:
            coordinates = self.get_layout_coordinates(layout, **kwargs)
            
            if filename:
                import json
                # Convert to JSON-serializable format
                json_coords = {str(node): [x, y] for node, (x, y) in coordinates.items()}
                
                with open(filename, 'w') as f:
                    json.dump({
                        'layout': layout,
                        'parameters': kwargs,
                        'coordinates': json_coords
                    }, f, indent=2)
                
                print(f"Layout exported to {filename}")
            
            return coordinates
            
        except Exception as e:
            raise HypergraphError(f"Failed to export layout: {e}")