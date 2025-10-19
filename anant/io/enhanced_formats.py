"""
Enhanced File Format Support for Anant Library

Provides comprehensive import/export capabilities for multiple file formats:
- Enhanced JSON support with schema validation
- CSV import/export with flexible column mapping
- GraphML format for graph visualization tools
- GML (Graph Modeling Language) support
- GEXF (Graph Exchange XML Format) support
- HDF5 for high-performance data storage
- NetworkX compatibility formats

This module bridges the gap between Anant hypergraphs and various graph formats.
"""

import json
import csv
import io
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import polars as pl
import numpy as np
from dataclasses import dataclass, asdict
import xml.etree.ElementTree as ET
from datetime import datetime

from ..classes.hypergraph import Hypergraph


@dataclass
class ImportExportConfig:
    """Configuration for import/export operations"""
    include_properties: bool = True
    include_weights: bool = True
    validate_schema: bool = True
    encoding: str = 'utf-8'
    compression: Optional[str] = None
    custom_mappings: Optional[Dict[str, str]] = None


class EnhancedFileFormats:
    """Enhanced file format support for hypergraphs"""
    
    def __init__(self, config: Optional[ImportExportConfig] = None):
        self.config = config or ImportExportConfig()
        self.supported_formats = [
            'json', 'csv', 'graphml', 'gml', 'gexf', 'parquet', 'hdf5'
        ]
    
    # ==================== JSON Support ====================
    
    def export_json(self, hg: Hypergraph, filepath: Union[str, Path]) -> None:
        """
        Export hypergraph to enhanced JSON format with schema validation
        
        Args:
            hg: Hypergraph to export
            filepath: Output file path
        """
        filepath = Path(filepath)
        
        # Create comprehensive JSON structure
        json_data = {
            "format": "anant_hypergraph",
            "version": "1.0",
            "metadata": {
                "created": datetime.now().isoformat(),
                "num_nodes": hg.num_nodes,
                "num_edges": hg.num_edges,
                "description": getattr(hg, 'description', None)
            },
            "schema": {
                "node_properties": [],
                "edge_properties": [],
                "incidence_properties": ["node_id", "edge_id", "weight"]
            },
            "data": {
                "nodes": [],
                "edges": [],
                "incidences": []
            }
        }
        
        # Export nodes with properties if available
        for node_id in hg.nodes:
            node_data = {"id": node_id}
            if self.config.include_properties and hasattr(hg, 'node_properties'):
                # Add node properties if they exist
                node_props = getattr(hg, 'node_properties', {}).get(node_id, {})
                if node_props:
                    node_data.update(node_props)
            json_data["data"]["nodes"].append(node_data)
        
        # Export edges with properties
        for edge_id in hg.edges:
            edge_data = {
                "id": edge_id,
                "size": hg.get_edge_size(edge_id)
            }
            if self.config.include_properties and hasattr(hg, 'edge_properties'):
                edge_props = getattr(hg, 'edge_properties', {}).get(edge_id, {})
                if edge_props:
                    edge_data.update(edge_props)
            json_data["data"]["edges"].append(edge_data)
        
        # Export incidences
        incidence_data = hg.incidences.data
        for row in incidence_data.iter_rows(named=True):
            incidence = {
                "node_id": row["node_id"],
                "edge_id": row["edge_id"]
            }
            if self.config.include_weights and "weight" in row:
                incidence["weight"] = row["weight"]
            json_data["data"]["incidences"].append(incidence)
        
        # Write to file
        with open(filepath, 'w', encoding=self.config.encoding) as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    def import_json(self, filepath: Union[str, Path]) -> Hypergraph:
        """
        Import hypergraph from enhanced JSON format
        
        Args:
            filepath: Input file path
            
        Returns:
            Hypergraph instance
        """
        filepath = Path(filepath)
        
        with open(filepath, 'r', encoding=self.config.encoding) as f:
            json_data = json.load(f)
        
        # Validate format if enabled
        if self.config.validate_schema:
            self._validate_json_schema(json_data)
        
        # Create hypergraph from incidences
        incidences = []
        for inc in json_data["data"]["incidences"]:
            incidence = {
                "node_id": inc["node_id"],
                "edge_id": inc["edge_id"],
                "weight": inc.get("weight", 1.0)
            }
            incidences.append(incidence)
        
        if incidences:
            incidence_df = pl.DataFrame(incidences)
            hg = Hypergraph()
            
            # Add edges from incidence data
            for edge_id in incidence_df["edge_id"].unique():
                edge_data = incidence_df.filter(pl.col("edge_id") == edge_id)
                nodes = edge_data["node_id"].to_list()
                weights = edge_data["weight"].to_list()
                weight = weights[0] if len(set(weights)) == 1 else 1.0
                hg.add_edge(edge_id, nodes, weight=weight)
        else:
            hg = Hypergraph()
        
        # Add metadata if available
        if "metadata" in json_data and "description" in json_data["metadata"]:
            hg.description = json_data["metadata"]["description"]
        
        return hg
    
    def _validate_json_schema(self, json_data: Dict) -> None:
        """Validate JSON schema"""
        required_fields = ["format", "version", "data"]
        for field in required_fields:
            if field not in json_data:
                raise ValueError(f"Missing required field: {field}")
        
        if json_data["format"] != "anant_hypergraph":
            raise ValueError("Invalid format identifier")
    
    # ==================== CSV Support ====================
    
    def export_csv(self, hg: Hypergraph, filepath: Union[str, Path], 
                   format_type: str = "incidences") -> None:
        """
        Export hypergraph to CSV format
        
        Args:
            hg: Hypergraph to export
            filepath: Output file path
            format_type: 'incidences', 'edgelist', or 'adjacency'
        """
        filepath = Path(filepath)
        
        if format_type == "incidences":
            # Export as incidence matrix
            incidence_data = hg.incidences.data
            incidence_data.write_csv(filepath)
            
        elif format_type == "edgelist":
            # Export as edge list format
            edges_data = []
            for edge_id in hg.edges:
                edge_nodes = hg.incidences.data.filter(
                    pl.col("edge_id") == edge_id
                )["node_id"].to_list()
                
                for i, node1 in enumerate(edge_nodes):
                    for node2 in edge_nodes[i+1:]:
                        edges_data.append({
                            "source": node1,
                            "target": node2,
                            "edge_id": edge_id,
                            "edge_size": len(edge_nodes)
                        })
            
            if edges_data:
                edge_df = pl.DataFrame(edges_data)
                edge_df.write_csv(filepath)
            else:
                # Empty CSV with headers
                with open(filepath, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["source", "target", "edge_id", "edge_size"])
                    
        elif format_type == "adjacency":
            # Export as adjacency list
            adj_data = []
            for node in hg.nodes:
                neighbors = set()
                node_edges = hg.incidences.data.filter(
                    pl.col("node_id") == node
                )["edge_id"].to_list()
                
                for edge_id in node_edges:
                    edge_nodes = hg.incidences.data.filter(
                        pl.col("edge_id") == edge_id
                    )["node_id"].to_list()
                    neighbors.update(n for n in edge_nodes if n != node)
                
                adj_data.append({
                    "node": node,
                    "neighbors": ";".join(str(n) for n in sorted(neighbors)),
                    "degree": len(neighbors)
                })
            
            if adj_data:
                adj_df = pl.DataFrame(adj_data)
                adj_df.write_csv(filepath)
    
    def import_csv(self, filepath: Union[str, Path], 
                   format_type: str = "incidences",
                   column_mapping: Optional[Dict[str, str]] = None) -> Hypergraph:
        """
        Import hypergraph from CSV format
        
        Args:
            filepath: Input file path
            format_type: 'incidences', 'edgelist', or 'adjacency'
            column_mapping: Custom column name mappings
            
        Returns:
            Hypergraph instance
        """
        filepath = Path(filepath)
        df = pl.read_csv(filepath)
        
        # Apply column mapping if provided
        if column_mapping:
            df = df.rename(column_mapping)
        
        if format_type == "incidences":
            # Direct import from incidences
            required_cols = ["node_id", "edge_id"]
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"CSV must contain columns: {required_cols}")
            
            hg = Hypergraph()
            if "weight" not in df.columns:
                df = df.with_columns(pl.lit(1.0).alias("weight"))
            
            # Group by edge and add edges
            for edge_id in df["edge_id"].unique():
                edge_data = df.filter(pl.col("edge_id") == edge_id)
                nodes = edge_data["node_id"].to_list()
                weights = edge_data["weight"].to_list()
                weight = weights[0] if len(set(weights)) == 1 else 1.0
                hg.add_edge(edge_id, nodes, weight=weight)
            
            return hg
            
        elif format_type == "edgelist":
            # Import from edge list
            required_cols = ["source", "target"]
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"CSV must contain columns: {required_cols}")
            
            hg = Hypergraph()
            
            # Group by edge_id if available, otherwise create unique edges
            if "edge_id" in df.columns:
                for edge_id in df["edge_id"].unique():
                    edge_data = df.filter(pl.col("edge_id") == edge_id)
                    nodes = set()
                    for row in edge_data.iter_rows(named=True):
                        nodes.update([row["source"], row["target"]])
                    hg.add_edge(edge_id, list(nodes))
            else:
                # Create edges from pairs
                for i, row in enumerate(df.iter_rows(named=True)):
                    edge_id = f"edge_{i}"
                    nodes = [row["source"], row["target"]]
                    hg.add_edge(edge_id, nodes)
            
            return hg
            
        else:
            raise ValueError(f"Unsupported format_type: {format_type}")
    
    # ==================== GraphML Support ====================
    
    def export_graphml(self, hg: Hypergraph, filepath: Union[str, Path]) -> None:
        """
        Export hypergraph to GraphML format for visualization tools
        
        Args:
            hg: Hypergraph to export
            filepath: Output file path
        """
        filepath = Path(filepath)
        
        # Create GraphML XML structure
        root = ET.Element("graphml")
        root.set("xmlns", "http://graphml.graphdrawing.org/xmlns")
        root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
        root.set("xsi:schemaLocation", 
                "http://graphml.graphdrawing.org/xmlns "
                "http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd")
        
        # Define key attributes
        node_weight_key = ET.SubElement(root, "key")
        node_weight_key.set("id", "node_weight")
        node_weight_key.set("for", "node")
        node_weight_key.set("attr.name", "weight")
        node_weight_key.set("attr.type", "double")
        
        edge_weight_key = ET.SubElement(root, "key")
        edge_weight_key.set("id", "edge_weight")
        edge_weight_key.set("for", "edge")
        edge_weight_key.set("attr.name", "weight")
        edge_weight_key.set("attr.type", "double")
        
        hyperedge_key = ET.SubElement(root, "key")
        hyperedge_key.set("id", "hyperedge_id")
        hyperedge_key.set("for", "edge")
        hyperedge_key.set("attr.name", "hyperedge_id")
        hyperedge_key.set("attr.type", "string")
        
        # Create graph
        graph = ET.SubElement(root, "graph")
        graph.set("id", "hypergraph")
        graph.set("edgedefault", "undirected")
        
        # Add nodes
        for node_id in hg.nodes:
            node = ET.SubElement(graph, "node")
            node.set("id", str(node_id))
            
            # Add node weight if available
            node_data = ET.SubElement(node, "data")
            node_data.set("key", "node_weight")
            node_data.text = "1.0"
        
        # Convert hyperedges to regular edges for GraphML
        edge_counter = 0
        for edge_id in hg.edges:
            edge_nodes = hg.incidences.data.filter(
                pl.col("edge_id") == edge_id
            )["node_id"].to_list()
            
            # Create clique (all pairs) for each hyperedge
            for i, node1 in enumerate(edge_nodes):
                for node2 in edge_nodes[i+1:]:
                    edge = ET.SubElement(graph, "edge")
                    edge.set("id", f"e{edge_counter}")
                    edge.set("source", str(node1))
                    edge.set("target", str(node2))
                    
                    # Add hyperedge ID
                    hyperedge_data = ET.SubElement(edge, "data")
                    hyperedge_data.set("key", "hyperedge_id")
                    hyperedge_data.text = str(edge_id)
                    
                    # Add edge weight
                    weight_data = ET.SubElement(edge, "data")
                    weight_data.set("key", "edge_weight")
                    weight_data.text = "1.0"
                    
                    edge_counter += 1
        
        # Write to file
        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ", level=0)
        tree.write(filepath, encoding=self.config.encoding, xml_declaration=True)
    
    # ==================== GML Support ====================
    
    def export_gml(self, hg: Hypergraph, filepath: Union[str, Path]) -> None:
        """
        Export hypergraph to GML (Graph Modeling Language) format
        
        Args:
            hg: Hypergraph to export
            filepath: Output file path
        """
        filepath = Path(filepath)
        
        with open(filepath, 'w', encoding=self.config.encoding) as f:
            f.write("graph [\n")
            f.write("  comment \"Anant Hypergraph Export\"\n")
            f.write("  directed 0\n")
            
            # Write nodes
            for node_id in hg.nodes:
                f.write(f"  node [\n")
                f.write(f"    id {node_id}\n")
                f.write(f"    label \"{node_id}\"\n")
                f.write(f"  ]\n")
            
            # Convert hyperedges to edges (clique representation)
            edge_counter = 0
            for edge_id in hg.edges:
                edge_nodes = hg.incidences.data.filter(
                    pl.col("edge_id") == edge_id
                )["node_id"].to_list()
                
                # Create clique for hyperedge
                for i, node1 in enumerate(edge_nodes):
                    for node2 in edge_nodes[i+1:]:
                        f.write(f"  edge [\n")
                        f.write(f"    source {node1}\n")
                        f.write(f"    target {node2}\n")
                        f.write(f"    hyperedge_id \"{edge_id}\"\n")
                        f.write(f"    weight 1.0\n")
                        f.write(f"  ]\n")
                        edge_counter += 1
            
            f.write("]\n")
    
    # ==================== GEXF Support ====================
    
    def export_gexf(self, hg: Hypergraph, filepath: Union[str, Path]) -> None:
        """
        Export hypergraph to GEXF (Graph Exchange XML Format)
        
        Args:
            hg: Hypergraph to export
            filepath: Output file path
        """
        filepath = Path(filepath)
        
        # Create GEXF XML structure
        root = ET.Element("gexf")
        root.set("xmlns", "http://www.gexf.net/1.2draft")
        root.set("version", "1.2")
        
        meta = ET.SubElement(root, "meta")
        meta.set("lastmodifieddate", datetime.now().strftime("%Y-%m-%d"))
        
        creator = ET.SubElement(meta, "creator")
        creator.text = "Anant Library"
        
        description = ET.SubElement(meta, "description")
        description.text = "Hypergraph exported from Anant"
        
        # Create graph
        graph = ET.SubElement(root, "graph")
        graph.set("mode", "static")
        graph.set("defaultedgetype", "undirected")
        
        # Define attributes
        attributes = ET.SubElement(graph, "attributes")
        attributes.set("class", "edge")
        
        hyperedge_attr = ET.SubElement(attributes, "attribute")
        hyperedge_attr.set("id", "0")
        hyperedge_attr.set("title", "hyperedge_id")
        hyperedge_attr.set("type", "string")
        
        # Add nodes
        nodes = ET.SubElement(graph, "nodes")
        for node_id in hg.nodes:
            node = ET.SubElement(nodes, "node")
            node.set("id", str(node_id))
            node.set("label", str(node_id))
        
        # Add edges (clique representation of hyperedges)
        edges = ET.SubElement(graph, "edges")
        edge_counter = 0
        
        for edge_id in hg.edges:
            edge_nodes = hg.incidences.data.filter(
                pl.col("edge_id") == edge_id
            )["node_id"].to_list()
            
            # Create clique for hyperedge
            for i, node1 in enumerate(edge_nodes):
                for node2 in edge_nodes[i+1:]:
                    edge = ET.SubElement(edges, "edge")
                    edge.set("id", str(edge_counter))
                    edge.set("source", str(node1))
                    edge.set("target", str(node2))
                    
                    # Add hyperedge ID as attribute
                    attvalues = ET.SubElement(edge, "attvalues")
                    attvalue = ET.SubElement(attvalues, "attvalue")
                    attvalue.set("for", "0")
                    attvalue.set("value", str(edge_id))
                    
                    edge_counter += 1
        
        # Write to file
        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ", level=0)
        tree.write(filepath, encoding=self.config.encoding, xml_declaration=True)
    
    # ==================== HDF5 Support ====================
    
    def export_hdf5(self, hg: Hypergraph, filepath: Union[str, Path]) -> None:
        """
        Export hypergraph to HDF5 format for high-performance storage
        
        Args:
            hg: Hypergraph to export
            filepath: Output file path
        """
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py required for HDF5 support. Install with: pip install h5py")
        
        filepath = Path(filepath)
        
        with h5py.File(filepath, 'w') as f:
            # Store metadata
            f.attrs['format'] = 'anant_hypergraph'
            f.attrs['version'] = '1.0'
            f.attrs['num_nodes'] = hg.num_nodes
            f.attrs['num_edges'] = hg.num_edges
            f.attrs['created'] = datetime.now().isoformat()
            
            # Store incidence data
            incidence_data = hg.incidences.data
            
            # Convert to numpy arrays for HDF5
            node_ids = incidence_data["node_id"].to_numpy()
            edge_ids = incidence_data["edge_id"].to_numpy()
            weights = incidence_data["weight"].to_numpy() if "weight" in incidence_data.columns else np.ones(len(node_ids))
            
            # Store datasets
            f.create_dataset('node_ids', data=node_ids.astype('S'))
            f.create_dataset('edge_ids', data=edge_ids.astype('S'))
            f.create_dataset('weights', data=weights)
            
            # Store node and edge lists
            all_nodes = list(hg.nodes)
            all_edges = list(hg.edges)
            
            f.create_dataset('nodes', data=[str(n).encode() for n in all_nodes])
            f.create_dataset('edges', data=[str(e).encode() for e in all_edges])
    
    def import_hdf5(self, filepath: Union[str, Path]) -> Hypergraph:
        """
        Import hypergraph from HDF5 format
        
        Args:
            filepath: Input file path
            
        Returns:
            Hypergraph instance
        """
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py required for HDF5 support. Install with: pip install h5py")
        
        filepath = Path(filepath)
        
        with h5py.File(filepath, 'r') as f:
            # Validate format
            if f.attrs.get('format') != 'anant_hypergraph':
                raise ValueError("Invalid HDF5 format")
            
            # Read datasets
            node_ids = [n.decode() for n in f['node_ids'][:]]
            edge_ids = [e.decode() for e in f['edge_ids'][:]]
            weights = f['weights'][:]
            
            # Create incidence DataFrame
            incidences = []
            for i in range(len(node_ids)):
                incidences.append({
                    "node_id": node_ids[i],
                    "edge_id": edge_ids[i],
                    "weight": float(weights[i])
                })
            
            if incidences:
                incidence_df = pl.DataFrame(incidences)
                hg = Hypergraph()
                
                # Add edges from incidence data
                for edge_id in incidence_df["edge_id"].unique():
                    edge_data = incidence_df.filter(pl.col("edge_id") == edge_id)
                    nodes = edge_data["node_id"].to_list()
                    edge_weights = edge_data["weight"].to_list()
                    weight = edge_weights[0] if len(set(edge_weights)) == 1 else 1.0
                    hg.add_edge(edge_id, nodes, weight=weight)
            else:
                hg = Hypergraph()
        
        return hg
    
    # ==================== Generic Export/Import ====================
    
    def export(self, hg: Hypergraph, filepath: Union[str, Path], 
               format_type: Optional[str] = None, **kwargs) -> None:
        """
        Generic export method that auto-detects format from file extension
        
        Args:
            hg: Hypergraph to export
            filepath: Output file path
            format_type: Override auto-detection
            **kwargs: Format-specific arguments
        """
        filepath = Path(filepath)
        
        if format_type is None:
            format_type = filepath.suffix.lower().lstrip('.')
        
        if format_type == 'json':
            self.export_json(hg, filepath)
        elif format_type == 'csv':
            # Extract csv_format from kwargs, default to 'incidences'
            csv_format = kwargs.pop('csv_format', 'incidences')
            self.export_csv(hg, filepath, format_type=csv_format, **kwargs)
        elif format_type == 'graphml':
            self.export_graphml(hg, filepath)
        elif format_type == 'gml':
            self.export_gml(hg, filepath)
        elif format_type == 'gexf':
            self.export_gexf(hg, filepath)
        elif format_type == 'h5' or format_type == 'hdf5':
            self.export_hdf5(hg, filepath)
        elif format_type == 'parquet':
            hg.incidences.data.write_parquet(filepath)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def import_file(self, filepath: Union[str, Path], 
                    format_type: Optional[str] = None, **kwargs) -> Hypergraph:
        """
        Generic import method that auto-detects format from file extension
        
        Args:
            filepath: Input file path
            format_type: Override auto-detection
            **kwargs: Format-specific arguments
            
        Returns:
            Hypergraph instance
        """
        filepath = Path(filepath)
        
        if format_type is None:
            format_type = filepath.suffix.lower().lstrip('.')
        
        if format_type == 'json':
            return self.import_json(filepath)
        elif format_type == 'csv':
            # Extract csv_format from kwargs, default to 'incidences'
            csv_format = kwargs.pop('csv_format', 'incidences')
            return self.import_csv(filepath, format_type=csv_format, **kwargs)
        elif format_type == 'h5' or format_type == 'hdf5':
            return self.import_hdf5(filepath)
        elif format_type == 'parquet':
            incidence_df = pl.read_parquet(filepath)
            hg = Hypergraph()
            if not incidence_df.is_empty():
                for edge_id in incidence_df["edge_id"].unique():
                    edge_data = incidence_df.filter(pl.col("edge_id") == edge_id)
                    nodes = edge_data["node_id"].to_list()
                    weights = edge_data["weight"].to_list() if "weight" in edge_data.columns else [1.0] * len(nodes)
                    weight = weights[0] if len(set(weights)) == 1 else 1.0
                    hg.add_edge(edge_id, nodes, weight=weight)
            return hg
        else:
            raise ValueError(f"Unsupported format: {format_type}")


# Convenience functions
def export_hypergraph(hg: Hypergraph, filepath: Union[str, Path], 
                     format_type: Optional[str] = None, 
                     config: Optional[ImportExportConfig] = None, **kwargs) -> None:
    """Convenience function for exporting hypergraphs"""
    formats = EnhancedFileFormats(config)
    formats.export(hg, filepath, format_type, **kwargs)


def import_hypergraph(filepath: Union[str, Path], 
                     format_type: Optional[str] = None,
                     config: Optional[ImportExportConfig] = None, **kwargs) -> Hypergraph:
    """Convenience function for importing hypergraphs"""
    formats = EnhancedFileFormats(config)
    return formats.import_file(filepath, format_type, **kwargs)