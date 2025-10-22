"""
I/O operations for Hypergraph

Handles save/load operations, format conversion, and import/export utilities
for various file formats including JSON, CSV, GraphML, GEXF, and NetworkX.
"""

from typing import Dict, List, Any, Optional, Union
import json
from pathlib import Path
import polars as pl
from ....exceptions import HypergraphError, ValidationError


class IOOperations:
    """
    Input/Output operations for hypergraph
    
    Handles serialization, deserialization, and format conversion operations
    including support for JSON, CSV, GraphML, GEXF, and NetworkX formats.
    """
    
    def __init__(self, hypergraph):
        """
        Initialize IOOperations
        
        Parameters
        ----------
        hypergraph : Hypergraph
            Parent hypergraph instance
        """
        if hypergraph is None:
            raise HypergraphError("Hypergraph instance cannot be None")
        self.hypergraph = hypergraph
    
    def to_dict(self) -> Dict[Any, List[Any]]:
        """
        Convert hypergraph to edge dictionary representation
        
        Returns
        -------
        Dict[Any, List[Any]]
            Dictionary mapping edge IDs to lists of node IDs
            
        Raises
        ------
        HypergraphError
            If conversion fails
        """
        try:
            if self.hypergraph.incidences.data.is_empty():
                return {}
            
            # Group by edge_id and collect node_ids
            result = {}
            
            edge_groups = self.hypergraph.incidences.data.group_by('edge_id', maintain_order=True)
            
            for (edge_id,), group_data in edge_groups:
                node_list = group_data.select('node_id').unique().to_series().to_list()
                result[edge_id] = node_list
            
            return result
            
        except Exception as e:
            raise HypergraphError(f"Failed to convert hypergraph to dictionary: {e}")
    
    def to_dataframe(self) -> pl.DataFrame:
        """
        Get the underlying incidence DataFrame
        
        Returns
        -------
        pl.DataFrame
            Clone of the incidence data DataFrame
            
        Raises
        ------
        HypergraphError
            If DataFrame access fails
        """
        try:
            return self.hypergraph.incidences.data.clone()
        except Exception as e:
            raise HypergraphError(f"Failed to convert hypergraph to DataFrame: {e}")
    
    def save(self, filepath: Union[str, Path], format: Optional[str] = None) -> None:
        """
        Save hypergraph to file
        
        Parameters
        ----------
        filepath : Union[str, Path]
            Path to save the file
        format : Optional[str]
            File format ('csv', 'json', 'parquet'). If None, inferred from extension
            
        Raises
        ------
        ValidationError
            If filepath or format is invalid
        HypergraphError
            If save operation fails
        """
        if not filepath:
            raise ValidationError("Filepath cannot be empty")
        
        try:
            filepath = Path(filepath)
            
            if format is None:
                format = filepath.suffix.lower().lstrip('.')
                if not format:
                    raise ValidationError("Cannot determine file format from path, please specify format")
            
            # Create parent directory if it doesn't exist
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            if format == 'csv':
                self.hypergraph.incidences.data.write_csv(filepath)
            elif format == 'json':
                json_data = self.to_json()
                with open(filepath, 'w') as f:
                    f.write(json_data)
            elif format == 'parquet':
                self.hypergraph.incidences.data.write_parquet(filepath)
            elif format in ['gexf', 'graphml']:
                if format == 'gexf':
                    data = self.to_gexf()
                else:
                    data = self.to_graphml()
                with open(filepath, 'w') as f:
                    f.write(data)
            else:
                raise ValidationError(f"Unsupported format: {format}")
                
        except Exception as e:
            raise HypergraphError(f"Failed to save hypergraph to {filepath}: {e}")
    
    @classmethod
    def load(cls, hypergraph_class, filepath: Union[str, Path], format: Optional[str] = None):
        """
        Load hypergraph from file
        
        Parameters
        ----------
        hypergraph_class : type
            Hypergraph class for creating new instance
        filepath : Union[str, Path]
            Path to the file to load
        format : Optional[str]
            File format. If None, inferred from extension
            
        Returns
        -------
        Hypergraph
            Loaded hypergraph instance
            
        Raises
        ------
        ValidationError
            If filepath or format is invalid
        HypergraphError
            If load operation fails
        """
        if not filepath:
            raise ValidationError("Filepath cannot be empty")
        
        try:
            filepath = Path(filepath)
            
            if not filepath.exists():
                raise ValidationError(f"File does not exist: {filepath}")
            
            if format is None:
                format = filepath.suffix.lower().lstrip('.')
                if not format:
                    raise ValidationError("Cannot determine file format from path, please specify format")
            
            if format == 'csv':
                data = pl.read_csv(filepath)
                return hypergraph_class.from_dataframe(data, name=filepath.stem)
            elif format == 'json':
                with open(filepath, 'r') as f:
                    json_data = f.read()
                return cls.from_json(hypergraph_class, json_data)
            elif format == 'parquet':
                data = pl.read_parquet(filepath)
                return hypergraph_class.from_dataframe(data, name=filepath.stem)
            elif format in ['gexf', 'graphml']:
                with open(filepath, 'r') as f:
                    data = f.read()
                if format == 'gexf':
                    return cls.from_gexf(hypergraph_class, data, name=filepath.stem)
                else:
                    return cls.from_graphml(hypergraph_class, data, name=filepath.stem)
            else:
                raise ValidationError(f"Unsupported format: {format}")
                
        except Exception as e:
            raise HypergraphError(f"Failed to load hypergraph from {filepath}: {e}")
    
    def to_json(self) -> str:
        """
        Convert hypergraph to JSON representation
        
        Returns
        -------
        str
            JSON string representation of the hypergraph
            
        Raises
        ------
        HypergraphError
            If JSON conversion fails
        """
        try:
            data = {
                'name': self.hypergraph.name,
                'nodes': list(self.hypergraph.nodes),
                'edges': {},
                'node_properties': {},
                'edge_properties': {},
                'metadata': getattr(self.hypergraph, 'metadata', {})
            }
            
            # Add edges
            for edge in self.hypergraph.edges:
                edge_nodes = self.hypergraph.incidences.get_edge_nodes(edge)
                data['edges'][str(edge)] = edge_nodes
            
            # Add properties
            for node in self.hypergraph.nodes:
                props = self.hypergraph.properties.get_node_properties(node)
                if props:
                    data['node_properties'][str(node)] = props
            
            for edge in self.hypergraph.edges:
                props = self.hypergraph.properties.get_edge_properties(edge)
                if props:
                    data['edge_properties'][str(edge)] = props
            
            return json.dumps(data, indent=2, default=str)
            
        except Exception as e:
            raise HypergraphError(f"Failed to convert hypergraph to JSON: {e}")
    
    @classmethod
    def from_json(cls, hypergraph_class, json_str: str):
        """
        Create hypergraph from JSON representation
        
        Parameters
        ----------
        hypergraph_class : type
            Hypergraph class for creating new instance
        json_str : str
            JSON string representation
            
        Returns
        -------
        Hypergraph
            Hypergraph instance created from JSON
            
        Raises
        ------
        ValidationError
            If JSON data is invalid
        HypergraphError
            If hypergraph creation fails
        """
        if not json_str:
            raise ValidationError("JSON string cannot be empty")
        
        try:
            data = json.loads(json_str)
            
            if not isinstance(data, dict):
                raise ValidationError("JSON data must be a dictionary")
            
            # Create hypergraph from edges
            edges = data.get('edges', {})
            name = data.get('name', 'from_json')
            metadata = data.get('metadata', {})
            
            hg = hypergraph_class.from_dict(edges, name=name)
            hg.metadata.update(metadata)
            
            # Set properties
            for node, props in data.get('node_properties', {}).items():
                hg.properties.set_node_properties(node, props)
            
            for edge, props in data.get('edge_properties', {}).items():
                hg.properties.set_edge_properties(edge, props)
            
            return hg
            
        except json.JSONDecodeError as e:
            raise ValidationError(f"Invalid JSON format: {e}")
        except Exception as e:
            raise HypergraphError(f"Failed to create hypergraph from JSON: {e}")
    
    def to_gexf(self) -> str:
        """
        Export hypergraph to GEXF format
        
        Returns
        -------
        str
            GEXF XML string representation
            
        Raises
        ------
        HypergraphError
            If GEXF conversion fails
        """
        try:
            from xml.etree.ElementTree import Element, SubElement, tostring
            from xml.dom import minidom
            
            # Create root element
            gexf = Element('gexf')
            gexf.set('xmlns', 'http://www.gexf.net/1.2draft')
            gexf.set('version', '1.2')
            
            # Add graph element
            graph = SubElement(gexf, 'graph')
            graph.set('mode', 'static')
            graph.set('defaultedgetype', 'undirected')
            
            # Add nodes
            nodes_elem = SubElement(graph, 'nodes')
            for i, node in enumerate(self.hypergraph.nodes):
                node_elem = SubElement(nodes_elem, 'node')
                node_elem.set('id', str(node))
                node_elem.set('label', str(node))
            
            # Add edges (convert hyperedges to pairwise edges)
            edges_elem = SubElement(graph, 'edges')
            edge_id = 0
            
            for hyperedge in self.hypergraph.edges:
                edge_nodes = self.hypergraph.incidences.get_edge_nodes(hyperedge)
                
                # Create pairwise connections for hyperedge
                for i, node1 in enumerate(edge_nodes):
                    for node2 in edge_nodes[i+1:]:
                        edge_elem = SubElement(edges_elem, 'edge')
                        edge_elem.set('id', str(edge_id))
                        edge_elem.set('source', str(node1))
                        edge_elem.set('target', str(node2))
                        edge_elem.set('label', str(hyperedge))
                        edge_id += 1
            
            # Convert to pretty string
            rough_string = tostring(gexf, 'unicode')
            reparsed = minidom.parseString(rough_string)
            return reparsed.toprettyxml(indent='  ')
            
        except Exception as e:
            raise HypergraphError(f"Failed to convert hypergraph to GEXF: {e}")
    
    @classmethod
    def from_gexf(cls, hypergraph_class, gexf_str: str, name: str = "from_gexf"):
        """
        Create hypergraph from GEXF string
        
        Parameters
        ----------
        hypergraph_class : type
            Hypergraph class for creating new instance
        gexf_str : str
            GEXF XML string
        name : str
            Name for the hypergraph
            
        Returns
        -------
        Hypergraph
            Hypergraph instance created from GEXF
            
        Raises
        ------
        ValidationError
            If GEXF data is invalid
        HypergraphError
            If hypergraph creation fails
        """
        if not gexf_str:
            raise ValidationError("GEXF string cannot be empty")
        
        try:
            from xml.etree import ElementTree as ET
            
            root = ET.fromstring(gexf_str)
            
            # Find graph element
            graph = root.find('.//{http://www.gexf.net/1.2draft}graph')
            if graph is None:
                # Try without namespace
                graph = root.find('.//graph')
            
            if graph is None:
                raise ValidationError("No graph element found in GEXF")
            
            # Extract edges and convert to hypergraph format
            edges = {}
            edge_counter = 0
            
            edges_elem = graph.find('.//{http://www.gexf.net/1.2draft}edges')
            if edges_elem is None:
                edges_elem = graph.find('.//edges')
            
            if edges_elem is not None:
                for edge in edges_elem.findall('.//{http://www.gexf.net/1.2draft}edge'):
                    if edge is None:
                        edge = edges_elem.findall('.//edge')
                    
                    source = edge.get('source')
                    target = edge.get('target')
                    label = edge.get('label', f'edge_{edge_counter}')
                    
                    if source and target:
                        if label not in edges:
                            edges[label] = []
                        if source not in edges[label]:
                            edges[label].append(source)
                        if target not in edges[label]:
                            edges[label].append(target)
                    
                    edge_counter += 1
            
            return hypergraph_class.from_dict(edges, name=name)
            
        except ET.ParseError as e:
            raise ValidationError(f"Invalid GEXF XML format: {e}")
        except Exception as e:
            raise HypergraphError(f"Failed to create hypergraph from GEXF: {e}")
    
    def to_graphml(self) -> str:
        """
        Export hypergraph to GraphML format
        
        Returns
        -------
        str
            GraphML XML string representation
            
        Raises
        ------
        HypergraphError
            If GraphML conversion fails
        """
        try:
            from xml.etree.ElementTree import Element, SubElement, tostring
            from xml.dom import minidom
            
            # Create root element
            graphml = Element('graphml')
            graphml.set('xmlns', 'http://graphml.graphdrawing.org/xmlns')
            
            # Add graph element
            graph = SubElement(graphml, 'graph')
            graph.set('id', 'G')
            graph.set('edgedefault', 'undirected')
            
            # Add nodes
            for node in self.hypergraph.nodes:
                node_elem = SubElement(graph, 'node')
                node_elem.set('id', str(node))
            
            # Add edges (convert hyperedges to pairwise edges)
            edge_id = 0
            
            for hyperedge in self.hypergraph.edges:
                edge_nodes = self.hypergraph.incidences.get_edge_nodes(hyperedge)
                
                # Create pairwise connections for hyperedge
                for i, node1 in enumerate(edge_nodes):
                    for node2 in edge_nodes[i+1:]:
                        edge_elem = SubElement(graph, 'edge')
                        edge_elem.set('id', f'e{edge_id}')
                        edge_elem.set('source', str(node1))
                        edge_elem.set('target', str(node2))
                        edge_id += 1
            
            # Convert to pretty string
            rough_string = tostring(graphml, 'unicode')
            reparsed = minidom.parseString(rough_string)
            return reparsed.toprettyxml(indent='  ')
            
        except Exception as e:
            raise HypergraphError(f"Failed to convert hypergraph to GraphML: {e}")
    
    @classmethod
    def from_graphml(cls, hypergraph_class, graphml_str: str, name: str = "from_graphml"):
        """
        Create hypergraph from GraphML string
        
        Parameters
        ----------
        hypergraph_class : type
            Hypergraph class for creating new instance
        graphml_str : str
            GraphML XML string
        name : str
            Name for the hypergraph
            
        Returns
        -------
        Hypergraph
            Hypergraph instance created from GraphML
            
        Raises
        ------
        ValidationError
            If GraphML data is invalid
        HypergraphError
            If hypergraph creation fails
        """
        if not graphml_str:
            raise ValidationError("GraphML string cannot be empty")
        
        try:
            from xml.etree import ElementTree as ET
            
            root = ET.fromstring(graphml_str)
            
            # Find graph element
            graph = root.find('.//{http://graphml.graphdrawing.org/xmlns}graph')
            if graph is None:
                # Try without namespace
                graph = root.find('.//graph')
            
            if graph is None:
                raise ValidationError("No graph element found in GraphML")
            
            # Extract edges and convert to hypergraph format
            edges = {}
            edge_counter = 0
            
            for edge in graph.findall('.//{http://graphml.graphdrawing.org/xmlns}edge'):
                if not edge:
                    edge = graph.findall('.//edge')
                
                source = edge.get('source')
                target = edge.get('target')
                edge_id = f'edge_{edge_counter}'
                
                if source and target:
                    edges[edge_id] = [source, target]
                    edge_counter += 1
            
            return hypergraph_class.from_dict(edges, name=name)
            
        except ET.ParseError as e:
            raise ValidationError(f"Invalid GraphML XML format: {e}")
        except Exception as e:
            raise HypergraphError(f"Failed to create hypergraph from GraphML: {e}")
    
    def to_networkx(self):
        """
        Convert hypergraph to NetworkX graph
        
        Note: This converts hyperedges to pairwise edges, losing hypergraph structure
        
        Returns
        -------
        networkx.Graph
            NetworkX graph representation
            
        Raises
        ------
        HypergraphError
            If NetworkX conversion fails
        """
        try:
            try:
                import networkx as nx
            except ImportError:
                raise HypergraphError("NetworkX is required for this operation. Install with: pip install networkx")
            
            G = nx.Graph()
            
            # Add nodes
            G.add_nodes_from(self.hypergraph.nodes)
            
            # Convert hyperedges to pairwise edges
            for hyperedge in self.hypergraph.edges:
                edge_nodes = self.hypergraph.incidences.get_edge_nodes(hyperedge)
                
                # Create pairwise connections
                for i, node1 in enumerate(edge_nodes):
                    for node2 in edge_nodes[i+1:]:
                        G.add_edge(node1, node2, hyperedge=hyperedge)
            
            return G
            
        except Exception as e:
            raise HypergraphError(f"Failed to convert hypergraph to NetworkX: {e}")
    
    @classmethod
    def from_networkx(cls, hypergraph_class, nx_graph, name: str = "from_networkx"):
        """
        Create hypergraph from NetworkX graph
        
        Parameters
        ----------
        hypergraph_class : type
            Hypergraph class for creating new instance
        nx_graph : networkx.Graph
            NetworkX graph instance
        name : str
            Name for the hypergraph
            
        Returns
        -------
        Hypergraph
            Hypergraph instance created from NetworkX graph
            
        Raises
        ------
        ValidationError
            If NetworkX graph is invalid
        HypergraphError
            If hypergraph creation fails
        """
        if nx_graph is None:
            raise ValidationError("NetworkX graph cannot be None")
        
        try:
            # Convert each edge to a hyperedge
            edges_dict = {}
            
            for i, (u, v) in enumerate(nx_graph.edges()):
                edge_id = f'edge_{i}'
                edges_dict[edge_id] = [u, v]
            
            return hypergraph_class.from_dict(edges_dict, name=name)
            
        except Exception as e:
            raise HypergraphError(f"Failed to create hypergraph from NetworkX: {e}")