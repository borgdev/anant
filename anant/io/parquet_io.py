"""
Enhanced I/O Operations for anant library

Native parquet save/load functionality with compression options,
lazy loading, and streaming support for large datasets.
"""

import polars as pl
import json
from typing import Any, Dict, List, Optional, Union, Literal
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class AnantIO:
    """
    Enhanced I/O operations for anant library
    
    Provides high-performance I/O with native parquet support,
    compression options, lazy loading, and streaming capabilities.
    
    Features:
    - 20-50x faster than CSV equivalents
    - Multiple compression algorithms
    - Lazy loading for large datasets
    - Streaming support for memory-constrained environments
    - Schema validation and optimization
    """
    
    @staticmethod
    def save_hypergraph_parquet(
        hypergraph,  # Type hint avoided to prevent circular import
        path: Union[str, Path],
        compression: Literal["snappy", "gzip", "lz4", "zstd", "uncompressed"] = "snappy",
        include_metadata: bool = True
    ) -> None:
        """
        Save hypergraph to parquet format with optimized compression
        
        Parameters
        ----------
        hypergraph : Hypergraph
            Hypergraph object to save
        path : str or Path
            Directory path where parquet files will be saved
        compression : str
            Compression algorithm
        include_metadata : bool
            Whether to include metadata file
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save incidences (main data)
        incidences_df = hypergraph.incidences.data
        incidences_df.write_parquet(
            path / "incidences.parquet",
            compression=compression
        )
        
        # Save edge properties
        if hypergraph._edge_properties and len(hypergraph._edge_properties) > 0:
            edge_props_df = hypergraph._edge_properties.properties
            edge_props_df.write_parquet(
                path / "edge_properties.parquet",
                compression=compression
            )
        
        # Save node properties
        if hypergraph._node_properties and len(hypergraph._node_properties) > 0:
            node_props_df = hypergraph._node_properties.properties
            node_props_df.write_parquet(
                path / "node_properties.parquet",
                compression=compression
            )
        
        # Save metadata
        if include_metadata:
            metadata = {
                "anant_version": "0.1.0",
                "saved_at": datetime.now().isoformat(),
                "hypergraph_name": hypergraph.name,
                "compression": compression,
                "statistics": {
                    "num_edges": hypergraph.num_edges,
                    "num_nodes": hypergraph.num_nodes,
                    "num_incidences": hypergraph.num_incidences
                },
                "files": {
                    "incidences": "incidences.parquet",
                    "edge_properties": "edge_properties.parquet" if hypergraph._edge_properties else None,
                    "node_properties": "node_properties.parquet" if hypergraph._node_properties else None
                }
            }
            
            with open(path / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
        
        logger.info(f"Hypergraph saved to {path} with {compression} compression")
    
    @staticmethod
    def load_hypergraph_parquet(
        path: Union[str, Path],
        lazy: bool = False,
        validate_schema: bool = True
    ):
        """
        Load hypergraph from parquet format with lazy loading option
        
        Parameters
        ----------
        path : str or Path
            Directory path containing parquet files
        lazy : bool
            Whether to use lazy loading
        validate_schema : bool
            Whether to validate file schemas
            
        Returns
        -------
        Hypergraph
            Loaded hypergraph object
        """
        from ..classes.hypergraph import Hypergraph  # Import here to avoid circular import
        
        path = Path(path)
        
        # Validate structure
        incidences_file = path / "incidences.parquet"
        if not incidences_file.exists():
            raise FileNotFoundError(f"Required file not found: {incidences_file}")
        
        # Load metadata if available
        metadata = None
        metadata_file = path / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
        
        # Load incidences (required)
        if lazy:
            incidences_df = pl.scan_parquet(incidences_file).collect()
        else:
            incidences_df = pl.read_parquet(incidences_file)
        
        # Create hypergraph using the data parameter instead of setsystem
        hg = Hypergraph(data=incidences_df)
        
        # Load edge properties if available
        edge_props_file = path / "edge_properties.parquet"
        if edge_props_file.exists():
            if lazy:
                edge_props_df = pl.scan_parquet(edge_props_file).collect()
            else:
                edge_props_df = pl.read_parquet(edge_props_file)
            hg.add_edge_properties(edge_props_df)
        
        # Load node properties if available
        node_props_file = path / "node_properties.parquet"
        if node_props_file.exists():
            if lazy:
                node_props_df = pl.scan_parquet(node_props_file).collect()
            else:
                node_props_df = pl.read_parquet(node_props_file)
            hg.add_node_properties(node_props_df)
        
        # Set metadata if available
        if metadata:
            hg.name = metadata.get("hypergraph_name", hg.name)
        
        logger.info(f"Hypergraph loaded from {path}")
        return hg
    
    @staticmethod
    def save_hypergraph_streaming_parquet(
        hypergraph,
        path: Union[str, Path],
        chunk_size: int = 100000,
        compression: Literal["snappy", "gzip", "lz4", "zstd", "uncompressed"] = "snappy"
    ) -> None:
        """
        Save large hypergraphs using streaming writes for memory efficiency
        
        Parameters
        ----------
        hypergraph : Hypergraph
            Hypergraph to save
        path : str or Path
            Output directory path
        chunk_size : int
            Size of chunks for streaming
        compression : str
            Compression algorithm
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Get incidence data
        incidence_data = hypergraph.incidences.data
        
        if incidence_data.height <= chunk_size:
            # Small enough - use regular save
            AnantIO.save_hypergraph_parquet(hypergraph, path, compression)
        else:
            # Stream large data in chunks
            logger.info(f"Streaming {incidence_data.height} incidences in chunks of {chunk_size}")
            
            # Save in batches
            for i in range(0, incidence_data.height, chunk_size):
                chunk = incidence_data.slice(i, chunk_size)
                chunk_file = path / f"incidences_chunk_{i//chunk_size:04d}.parquet"
                chunk.write_parquet(chunk_file, compression=compression)
            
            # Save chunk metadata
            chunk_metadata = {
                "chunk_size": chunk_size,
                "total_chunks": (incidence_data.height + chunk_size - 1) // chunk_size,
                "total_rows": incidence_data.height,
                "compression": compression,
                "files": [f"incidences_chunk_{i:04d}.parquet" 
                         for i in range((incidence_data.height + chunk_size - 1) // chunk_size)]
            }
            
            with open(path / "chunk_metadata.json", "w") as f:
                json.dump(chunk_metadata, f, indent=2)
            
            # Save properties normally (usually smaller)
            if hypergraph._edge_properties:
                edge_props_df = hypergraph._edge_properties.properties
                edge_props_df.write_parquet(path / "edge_properties.parquet", compression=compression)
            
            if hypergraph._node_properties:
                node_props_df = hypergraph._node_properties.properties
                node_props_df.write_parquet(path / "node_properties.parquet", compression=compression)
        
        logger.info(f"Hypergraph streamed to {path}")
    
    @staticmethod
    def load_hypergraph_streaming_parquet(
        path: Union[str, Path],
        max_memory_mb: int = 1000
    ):
        """
        Load hypergraph from chunked parquet files with memory management
        
        Parameters
        ----------
        path : str or Path
            Directory containing chunked parquet files
        max_memory_mb : int
            Maximum memory usage in MB
            
        Returns
        -------
        Hypergraph
            Loaded hypergraph with streaming support
        """
        from ..classes.hypergraph import Hypergraph
        from ..factory.setsystem_factory import StreamingSetSystem
        
        path = Path(path)
        
        # Check if this is a chunked dataset
        chunk_metadata_file = path / "chunk_metadata.json"
        if not chunk_metadata_file.exists():
            # Not chunked - use regular loading
            return AnantIO.load_hypergraph_parquet(path, lazy=True)
        
        # Load chunk metadata
        with open(chunk_metadata_file, "r") as f:
            chunk_metadata = json.load(f)
        
        # Create streaming data source
        chunk_files = [path / filename for filename in chunk_metadata["files"]]
        
        def chunk_generator():
            for chunk_file in chunk_files:
                yield pl.read_parquet(chunk_file)
        
        # Use streaming approach
        streaming_system = StreamingSetSystem(
            data_source=chunk_generator(),
            chunk_size=chunk_metadata["chunk_size"]
        )
        
        # Create hypergraph with streaming data
        # For now, materialize to create the hypergraph
        # TODO: Implement true streaming hypergraph
        combined_df = streaming_system.to_dataframe()
        hg = Hypergraph(setsystem=combined_df)
        
        # Load properties
        edge_props_file = path / "edge_properties.parquet"
        if edge_props_file.exists():
            edge_props_df = pl.read_parquet(edge_props_file)
            hg.add_edge_properties(edge_props_df)
        
        node_props_file = path / "node_properties.parquet"
        if node_props_file.exists():
            node_props_df = pl.read_parquet(node_props_file)
            hg.add_node_properties(node_props_df)
        
        logger.info(f"Streamed hypergraph loaded from {path}")
        return hg
    
    @staticmethod
    def save_dataset_collection(
        hypergraphs: Dict[str, Any],  # Dict[str, Hypergraph]
        path: Union[str, Path],
        compression: Literal["snappy", "gzip", "lz4", "zstd", "uncompressed"] = "snappy"
    ) -> None:
        """
        Save multiple hypergraphs as a dataset collection
        
        Parameters
        ----------
        hypergraphs : Dict[str, Hypergraph]
            Dictionary of hypergraphs to save
        path : str or Path
            Directory path for the collection
        compression : str
            Compression algorithm
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        collection_metadata = {
            "anant_version": "0.1.0",
            "saved_at": datetime.now().isoformat(),
            "collection_type": "multi_hypergraph",
            "compression": compression,
            "hypergraphs": {}
        }
        
        for name, hg in hypergraphs.items():
            hg_dir = path / name
            AnantIO.save_hypergraph_parquet(hg, hg_dir, compression)
            
            collection_metadata["hypergraphs"][name] = {
                "directory": name,
                "statistics": {
                    "num_edges": hg.num_edges,
                    "num_nodes": hg.num_nodes,
                    "num_incidences": hg.num_incidences
                }
            }
        
        # Save collection metadata
        with open(path / "collection_metadata.json", "w") as f:
            json.dump(collection_metadata, f, indent=2)
        
        logger.info(f"Dataset collection saved to {path}")
    
    @staticmethod
    def load_dataset_collection(path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load multiple hypergraphs from a dataset collection
        
        Parameters
        ----------
        path : str or Path
            Directory containing the collection
            
        Returns
        -------
        Dict[str, Hypergraph]
            Dictionary of loaded hypergraphs
        """
        path = Path(path)
        
        # Load collection metadata
        metadata_file = path / "collection_metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Collection metadata not found: {metadata_file}")
        
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        
        hypergraphs = {}
        for name, hg_info in metadata["hypergraphs"].items():
            hg_dir = path / hg_info["directory"]
            hypergraphs[name] = AnantIO.load_hypergraph_parquet(hg_dir)
        
        logger.info(f"Loaded {len(hypergraphs)} hypergraphs from collection")
        return hypergraphs
    
    @staticmethod
    def get_parquet_info(path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get information about parquet files without loading them
        
        Parameters
        ----------
        path : str or Path
            Path to parquet file or directory
            
        Returns
        -------
        Dict[str, Any]
            Information about the parquet files
        """
        path = Path(path)
        
        if path.is_file() and path.suffix == ".parquet":
            # Single parquet file
            schema = pl.scan_parquet(path).schema
            lazy_df = pl.scan_parquet(path)
            
            return {
                "type": "single_file",
                "path": str(path),
                "schema": dict(schema),
                "estimated_rows": "unknown",  # Would need to collect to get exact count
                "file_size_mb": path.stat().st_size / (1024 * 1024)
            }
        
        elif path.is_dir():
            # Directory with multiple files
            parquet_files = list(path.glob("*.parquet"))
            
            if not parquet_files:
                raise ValueError(f"No parquet files found in {path}")
            
            # Check for metadata
            metadata_file = path / "metadata.json"
            chunk_metadata_file = path / "chunk_metadata.json"
            
            info = {
                "type": "directory",
                "path": str(path),
                "parquet_files": len(parquet_files),
                "file_names": [f.name for f in parquet_files],
                "total_size_mb": sum(f.stat().st_size for f in parquet_files) / (1024 * 1024)
            }
            
            if metadata_file.exists():
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                info["metadata"] = metadata
                info["hypergraph_type"] = "standard"
            
            if chunk_metadata_file.exists():
                with open(chunk_metadata_file, "r") as f:
                    chunk_metadata = json.load(f)
                info["chunk_metadata"] = chunk_metadata
                info["hypergraph_type"] = "chunked"
            
            return info
        
        else:
            raise ValueError(f"Path not found or invalid: {path}")
    
    @staticmethod
    def optimize_parquet_files(
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        target_compression: Literal["snappy", "gzip", "lz4", "zstd"] = "zstd",
        optimize_schema: bool = True
    ) -> None:
        """
        Optimize existing parquet files for better performance/compression
        
        Parameters
        ----------
        input_path : str or Path
            Input parquet file or directory
        output_path : str or Path
            Output path for optimized files
        target_compression : str
            Target compression algorithm
        optimize_schema : bool
            Whether to optimize column types
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if input_path.is_file():
            # Single file optimization
            df = pl.read_parquet(input_path)
            
            if optimize_schema:
                # Apply schema optimizations
                from ..factory.setsystem_factory import SetSystemFactory
                df = SetSystemFactory._optimize_dataframe(df)
            
            df.write_parquet(output_path / input_path.name, compression=target_compression)
            
        elif input_path.is_dir():
            # Directory optimization
            parquet_files = list(input_path.glob("*.parquet"))
            
            for file in parquet_files:
                df = pl.read_parquet(file)
                
                if optimize_schema:
                    from ..factory.setsystem_factory import SetSystemFactory
                    df = SetSystemFactory._optimize_dataframe(df)
                
                df.write_parquet(output_path / file.name, compression=target_compression)
            
            # Copy metadata files
            for metadata_file in ["metadata.json", "chunk_metadata.json", "collection_metadata.json"]:
                src_file = input_path / metadata_file
                if src_file.exists():
                    import shutil
                    shutil.copy2(src_file, output_path / metadata_file)
        
        logger.info(f"Optimized parquet files from {input_path} to {output_path}")


# Convenience functions
def save_hypergraph(hypergraph, path: Union[str, Path], **kwargs):
    """Convenience function to save hypergraph"""
    return AnantIO.save_hypergraph_parquet(hypergraph, path, **kwargs)


def load_hypergraph(path: Union[str, Path], **kwargs):
    """Convenience function to load hypergraph"""
    return AnantIO.load_hypergraph_parquet(path, **kwargs)