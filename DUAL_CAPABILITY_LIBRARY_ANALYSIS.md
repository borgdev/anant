# Dual Capability Library Analysis: Hypergraph + Metagraph
## Library Dependencies and Architecture for "anant" Extension

**Document Version**: 1.0  
**Created**: October 18, 2025  
**Focus**: Library selection for Hypergraph + Metagraph dual capabilities with Polars+Parquet metadata storage

---

## 🎯 Architecture Overview

### Dual Capability Design
```python
# User can choose their preferred interface
from anant import Hypergraph, Metagraph

# Traditional hypergraph usage
hg = Hypergraph(setsystem=my_data)
result = hg.s_centrality()

# Advanced metagraph usage with metadata storage
mg = Metagraph(
    hierarchical_data=my_complex_data,
    metadata_backend="polars+parquet"  # Default for metagraph
)
semantic_insights = mg.semantic_layer.analyze_patterns()
```

### Metadata Storage Strategy
- **Hypergraph**: Existing pandas-based property storage (backward compatibility)
- **Metagraph**: Polars+Parquet for rich metadata, hierarchical relationships, and LLM integration
- **Shared Core**: Common algorithms and base classes work with both

---

## 📚 Core Library Dependencies

### 1. **Data Processing & Storage**

#### Primary Dependencies
```toml
[dependencies]
# Core data processing (required for both Hypergraph and Metagraph)
polars = ">=0.20.0"              # High-performance DataFrame operations
pyarrow = ">=14.0.0"             # Parquet I/O, Arrow integration
pandas = ">=2.0.0"               # Backward compatibility for Hypergraph
numpy = ">=1.24.0"               # Numerical computations

# Enhanced I/O and compression
fsspec = ">=2023.10.0"           # Unified file system interface
pyarrow-parquet = ">=14.0.0"     # Optimized parquet operations
lz4 = ">=4.3.0"                  # Fast compression for metadata
zstd = ">=1.5.0"                 # High-ratio compression option
```

#### Storage Format Support
```toml
# Metadata storage formats
orjson = ">=3.9.0"               # Fast JSON for configuration/schemas
msgpack = ">=1.0.0"              # Efficient binary serialization
```

### 2. **Graph Theory & Algorithms**

#### Core Graph Libraries
```toml
# Graph algorithms and structures
networkx = ">=3.1"               # Graph algorithms, compatibility
scipy = ">=1.11.0"               # Sparse matrices, linear algebra
scikit-learn = ">=1.3.0"         # Clustering, dimensionality reduction
```

#### Advanced Analytics
```toml
# Statistical and ML capabilities
statsmodels = ">=0.14.0"         # Statistical analysis
umap-learn = ">=0.5.0"           # Dimensionality reduction for large graphs
hdbscan = ">=0.8.0"              # Density-based clustering
```

### 3. **LLM Integration & AI**

#### LLM and Embedding Support
```toml
# LLM integration for Metagraph semantic analysis
openai = ">=1.0.0"               # OpenAI API integration
anthropic = ">=0.5.0"            # Claude API integration
tiktoken = ">=0.5.0"             # Token counting and management

# Local embedding models
sentence-transformers = ">=2.2.0" # Sentence embeddings
transformers = ">=4.35.0"        # HuggingFace models
torch = ">=2.1.0"                # PyTorch backend (optional)
```

#### Vector Storage & Search
```toml
# Vector databases for semantic search
chromadb = ">=0.4.0"             # Lightweight vector DB
faiss-cpu = ">=1.7.0"            # Facebook AI Similarity Search
hnswlib = ">=0.7.0"              # Hierarchical NSW for fast ANN
```

### 4. **Temporal & Time Series**

#### Time Series Analysis
```toml
# Temporal analysis for Metagraph time-dependent relationships
arrow = ">=1.3.0"                # Modern datetime handling
pendulum = ">=2.1.0"             # Advanced timezone support
```

### 5. **Validation & Schema**

#### Data Validation
```toml
# Schema validation and data integrity
pydantic = ">=2.4.0"             # Data validation and settings
jsonschema = ">=4.19.0"          # JSON schema validation
cerberus = ">=1.3.0"             # Document validation
```

### 6. **Performance & Monitoring**

#### Performance Tools
```toml
# Performance monitoring and optimization
memory-profiler = ">=0.61.0"     # Memory usage tracking
psutil = ">=5.9.0"               # System resource monitoring
line-profiler = ">=4.1.0"        # Line-by-line profiling
```

### 7. **Development & Testing**

#### Development Dependencies
```toml
[dev-dependencies]
pytest = ">=7.4.0"               # Testing framework
pytest-benchmark = ">=4.0.0"     # Performance benchmarking
pytest-cov = ">=4.1.0"          # Coverage reporting
hypothesis = ">=6.88.0"          # Property-based testing

# Code quality
black = ">=23.9.0"               # Code formatting
ruff = ">=0.1.0"                 # Fast linting
mypy = ">=1.6.0"                 # Type checking
pre-commit = ">=3.5.0"          # Git hooks
```

---

## 🏗️ Architecture Implementation

### 1. **Unified Base Classes**

```python
# /anant/core/base.py
from abc import ABC, abstractmethod
from typing import Union, Dict, Any, Optional
import polars as pl
import pandas as pd

class BaseGraphStructure(ABC):
    """Base class for both Hypergraph and Metagraph"""
    
    def __init__(self, 
                 storage_backend: str = "pandas",  # "pandas" or "polars"
                 metadata_backend: Optional[str] = None):
        self.storage_backend = storage_backend
        self.metadata_backend = metadata_backend or storage_backend
        
    @abstractmethod
    def get_nodes(self) -> Union[list, pl.Series]:
        """Get all nodes in the structure"""
        pass
        
    @abstractmethod
    def get_edges(self) -> Union[list, pl.Series]:
        """Get all edges in the structure"""
        pass
```

### 2. **Hypergraph Implementation (Enhanced)**

```python
# /anant/classes/hypergraph.py
import pandas as pd
import polars as pl
from ..core.base import BaseGraphStructure

class Hypergraph(BaseGraphStructure):
    """Traditional hypergraph with optional Polars backend"""
    
    def __init__(self, 
                 setsystem=None,
                 storage_backend: str = "pandas",  # Backward compatibility
                 **kwargs):
        super().__init__(storage_backend=storage_backend)
        
        if storage_backend == "polars":
            self._use_polars_backend(setsystem)
        else:
            self._use_pandas_backend(setsystem)  # Existing implementation
    
    def _use_polars_backend(self, setsystem):
        """Initialize with Polars for better performance"""
        from ..io.polars_factory import create_polars_setsystem
        self._setsystem = create_polars_setsystem(setsystem)
        
    def _use_pandas_backend(self, setsystem):
        """Initialize with pandas (existing implementation)"""
        # Existing HyperNetX-compatible initialization
        pass
        
    def to_metagraph(self, 
                    semantic_enrichment: bool = True,
                    temporal_tracking: bool = False) -> 'Metagraph':
        """Convert hypergraph to metagraph with enhanced capabilities"""
        from .metagraph import Metagraph
        
        return Metagraph.from_hypergraph(
            self, 
            semantic_enrichment=semantic_enrichment,
            temporal_tracking=temporal_tracking
        )
```

### 3. **Metagraph Implementation**

```python
# /anant/classes/metagraph.py
import polars as pl
from pathlib import Path
from typing import Dict, Any, Optional, List
from ..core.base import BaseGraphStructure
from ..metadata import MetadataStore, SemanticLayer, TemporalLayer

class Metagraph(BaseGraphStructure):
    """Advanced metagraph with hierarchical knowledge modeling"""
    
    def __init__(self,
                 hierarchical_data=None,
                 metadata_backend: str = "polars+parquet",
                 storage_path: Optional[Path] = None,
                 llm_integration: bool = True,
                 **kwargs):
        
        super().__init__(
            storage_backend="polars",  # Metagraph always uses Polars
            metadata_backend=metadata_backend
        )
        
        # Initialize core components
        self.storage_path = Path(storage_path) if storage_path else Path("./metagraph_data")
        self.metadata_store = MetadataStore(
            backend=metadata_backend,
            storage_path=self.storage_path
        )
        
        # Initialize layers
        self.semantic_layer = SemanticLayer(
            metadata_store=self.metadata_store,
            llm_enabled=llm_integration
        )
        
        self.temporal_layer = TemporalLayer(
            metadata_store=self.metadata_store
        )
        
        # Load or create hierarchical structure
        if hierarchical_data is not None:
            self._initialize_from_data(hierarchical_data)
    
    @classmethod
    def from_hypergraph(cls, 
                       hypergraph: 'Hypergraph',
                       semantic_enrichment: bool = True,
                       temporal_tracking: bool = False) -> 'Metagraph':
        """Create metagraph from existing hypergraph"""
        
        # Convert hypergraph data to hierarchical structure
        hierarchical_data = cls._convert_hypergraph_to_hierarchical(hypergraph)
        
        metagraph = cls(
            hierarchical_data=hierarchical_data,
            llm_integration=semantic_enrichment
        )
        
        if temporal_tracking:
            metagraph.temporal_layer.enable_tracking()
            
        return metagraph
```

### 4. **Metadata Storage System**

```python
# /anant/metadata/store.py
import polars as pl
from pathlib import Path
from typing import Dict, Any, Optional, Union
import orjson

class MetadataStore:
    """Polars+Parquet-based metadata storage for Metagraph"""
    
    def __init__(self, 
                 backend: str = "polars+parquet",
                 storage_path: Path = Path("./metadata"),
                 compression: str = "zstd"):
        
        self.backend = backend
        self.storage_path = Path(storage_path)
        self.compression = compression
        
        # Create storage directories
        self._setup_storage_structure()
        
        # Initialize metadata tables
        self._initialize_metadata_tables()
    
    def _setup_storage_structure(self):
        """Create directory structure for metadata storage"""
        directories = [
            "entities",      # Entity metadata
            "relationships", # Relationship metadata  
            "schemas",       # Schema definitions
            "temporal",      # Time-dependent data
            "semantic",      # LLM-generated insights
            "policies"       # Governance policies
        ]
        
        for directory in directories:
            (self.storage_path / directory).mkdir(parents=True, exist_ok=True)
    
    def store_entity_metadata(self, 
                             entity_id: str,
                             metadata: Dict[str, Any],
                             entity_type: str = "node") -> None:
        """Store metadata for an entity (node/edge/hyperedge)"""
        
        # Convert to Polars DataFrame
        df = pl.DataFrame([{
            "entity_id": entity_id,
            "entity_type": entity_type,
            "metadata": orjson.dumps(metadata).decode(),
            "created_at": pl.datetime("now"),
            "updated_at": pl.datetime("now")
        }])
        
        # Append to parquet file
        file_path = self.storage_path / f"entities/{entity_type}_metadata.parquet"
        
        if file_path.exists():
            existing_df = pl.read_parquet(file_path)
            # Update existing or append new
            updated_df = self._upsert_dataframe(existing_df, df, "entity_id")
        else:
            updated_df = df
            
        updated_df.write_parquet(file_path, compression=self.compression)
    
    def get_entity_metadata(self, 
                           entity_id: str,
                           entity_type: str = "node") -> Optional[Dict[str, Any]]:
        """Retrieve metadata for an entity"""
        
        file_path = self.storage_path / f"entities/{entity_type}_metadata.parquet"
        
        if not file_path.exists():
            return None
            
        df = pl.read_parquet(file_path)
        result = df.filter(pl.col("entity_id") == entity_id)
        
        if result.height == 0:
            return None
            
        metadata_json = result.select("metadata").item()
        return orjson.loads(metadata_json)
    
    def _upsert_dataframe(self, 
                         existing: pl.DataFrame, 
                         new: pl.DataFrame, 
                         key_col: str) -> pl.DataFrame:
        """Update existing records or insert new ones"""
        
        # Update existing records
        updated = existing.join(
            new.select([key_col, "updated_at"]), 
            on=key_col, 
            how="left"
        ).with_columns([
            pl.when(pl.col("updated_at_right").is_not_null())
            .then(pl.datetime("now"))
            .otherwise(pl.col("updated_at"))
            .alias("updated_at")
        ]).drop("updated_at_right")
        
        # Add new records
        new_keys = set(new.select(key_col).to_series())
        existing_keys = set(existing.select(key_col).to_series())
        truly_new = new.filter(
            pl.col(key_col).is_in(list(new_keys - existing_keys))
        )
        
        return pl.concat([updated, truly_new])
```

### 5. **Semantic Layer with LLM Integration**

```python
# /anant/metadata/semantic.py
import polars as pl
from typing import Dict, Any, List, Optional
from .llm_integrations import OpenAIIntegration, AnthropicIntegration

class SemanticLayer:
    """Semantic analysis layer with LLM integration"""
    
    def __init__(self, 
                 metadata_store,
                 llm_enabled: bool = True,
                 llm_provider: str = "openai"):
        
        self.metadata_store = metadata_store
        self.llm_enabled = llm_enabled
        
        if llm_enabled:
            self.llm = self._initialize_llm(llm_provider)
            self.embeddings = self._initialize_embeddings()
    
    def enrich_entity_semantics(self, 
                               entity_id: str,
                               entity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to generate semantic enrichment for entity"""
        
        if not self.llm_enabled:
            return entity_data
            
        # Generate semantic description
        semantic_description = self.llm.generate_entity_description(entity_data)
        
        # Generate embeddings
        embeddings = self.embeddings.encode(semantic_description)
        
        # Store semantic metadata
        semantic_metadata = {
            "semantic_description": semantic_description,
            "embeddings": embeddings.tolist(),
            "semantic_tags": self.llm.extract_tags(entity_data),
            "importance_score": self.llm.assess_importance(entity_data)
        }
        
        # Store in metadata store
        self.metadata_store.store_entity_metadata(
            entity_id=f"{entity_id}_semantic",
            metadata=semantic_metadata,
            entity_type="semantic"
        )
        
        return {**entity_data, "semantic_enrichment": semantic_metadata}
    
    def find_semantic_patterns(self, 
                              pattern_type: str = "clusters") -> pl.DataFrame:
        """Discover semantic patterns in the metagraph"""
        
        # Load all semantic metadata
        semantic_file = self.metadata_store.storage_path / "entities/semantic_metadata.parquet"
        
        if not semantic_file.exists():
            return pl.DataFrame()
            
        semantic_df = pl.read_parquet(semantic_file)
        
        if pattern_type == "clusters":
            return self._find_semantic_clusters(semantic_df)
        elif pattern_type == "relationships":
            return self._find_semantic_relationships(semantic_df)
        else:
            raise ValueError(f"Unknown pattern type: {pattern_type}")
```

---

## 🔧 Configuration Management

### 1. **Library Configuration**

```python
# /anant/config/settings.py
from pydantic import BaseSettings, Field
from typing import Optional, Dict, Any

class AnantSettings(BaseSettings):
    """Configuration for anant library"""
    
    # Core settings
    default_backend: str = Field(default="pandas", description="Default storage backend")
    metagraph_backend: str = Field(default="polars+parquet", description="Metagraph storage backend")
    
    # Performance settings
    polars_streaming: bool = Field(default=True, description="Enable Polars streaming")
    max_memory_gb: float = Field(default=8.0, description="Maximum memory usage in GB")
    parallel_processing: bool = Field(default=True, description="Enable parallel processing")
    
    # LLM Integration
    llm_provider: str = Field(default="openai", description="LLM provider")
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key")
    
    # Storage settings
    default_compression: str = Field(default="zstd", description="Default compression algorithm")
    metadata_storage_path: str = Field(default="./anant_data", description="Metadata storage path")
    
    class Config:
        env_prefix = "ANANT_"
        env_file = ".env"

# Global settings instance
settings = AnantSettings()
```

### 2. **Dynamic Library Loading**

```python
# /anant/core/dynamic_imports.py
from typing import Optional, Any
import importlib
from ..config.settings import settings

class DynamicLibraryLoader:
    """Dynamically load optional dependencies based on usage"""
    
    _loaded_libraries = {}
    
    @classmethod
    def load_llm_integration(cls, provider: str = None) -> Any:
        """Load LLM integration library"""
        provider = provider or settings.llm_provider
        
        if provider in cls._loaded_libraries:
            return cls._loaded_libraries[provider]
            
        try:
            if provider == "openai":
                import openai
                cls._loaded_libraries[provider] = openai
                return openai
            elif provider == "anthropic":
                import anthropic
                cls._loaded_libraries[provider] = anthropic
                return anthropic
            else:
                raise ValueError(f"Unsupported LLM provider: {provider}")
        except ImportError as e:
            raise ImportError(
                f"LLM provider '{provider}' requires additional dependencies. "
                f"Install with: pip install anant[llm-{provider}]"
            ) from e
    
    @classmethod  
    def load_vector_db(cls, db_type: str = "chromadb") -> Any:
        """Load vector database library"""
        
        if db_type in cls._loaded_libraries:
            return cls._loaded_libraries[db_type]
            
        try:
            if db_type == "chromadb":
                import chromadb
                cls._loaded_libraries[db_type] = chromadb
                return chromadb
            elif db_type == "faiss":
                import faiss
                cls._loaded_libraries[db_type] = faiss
                return faiss
            else:
                raise ValueError(f"Unsupported vector DB: {db_type}")
        except ImportError as e:
            raise ImportError(
                f"Vector database '{db_type}' requires additional dependencies. "
                f"Install with: pip install anant[vector-{db_type}]"
            ) from e
```

---

## 📦 Package Structure

### 1. **Dependency Groups**

```toml
# pyproject.toml
[project.optional-dependencies]
# Core data processing
polars = ["polars>=0.20.0", "pyarrow>=14.0.0"]

# LLM integration options
llm-openai = ["openai>=1.0.0", "tiktoken>=0.5.0"]
llm-anthropic = ["anthropic>=0.5.0"]
llm-local = ["sentence-transformers>=2.2.0", "transformers>=4.35.0"]

# Vector databases
vector-chromadb = ["chromadb>=0.4.0"]
vector-faiss = ["faiss-cpu>=1.7.0"]
vector-hnswlib = ["hnswlib>=0.7.0"]

# Advanced analytics
analytics = ["scikit-learn>=1.3.0", "umap-learn>=0.5.0", "hdbscan>=0.8.0"]

# Full metagraph capabilities
metagraph = [
    "polars>=0.20.0",
    "pyarrow>=14.0.0", 
    "orjson>=3.9.0",
    "pydantic>=2.4.0",
    "chromadb>=0.4.0"
]

# All optional dependencies
all = [
    "polars>=0.20.0",
    "pyarrow>=14.0.0",
    "openai>=1.0.0",
    "anthropic>=0.5.0",
    "sentence-transformers>=2.2.0",
    "chromadb>=0.4.0",
    "scikit-learn>=1.3.0",
    "umap-learn>=0.5.0"
]

# Development dependencies
dev = [
    "pytest>=7.4.0",
    "pytest-benchmark>=4.0.0", 
    "black>=23.9.0",
    "ruff>=0.1.0",
    "mypy>=1.6.0"
]
```

### 2. **Installation Options**

```bash
# Basic installation (Hypergraph only, pandas backend)
pip install anant

# With Polars backend for performance
pip install anant[polars]

# Full Metagraph capabilities
pip install anant[metagraph]

# With specific LLM provider
pip install anant[metagraph,llm-openai]

# Full installation with all features
pip install anant[all]

# Development installation
pip install anant[dev]
```

---

## 🚀 Usage Examples

### 1. **Traditional Hypergraph Usage**

```python
from anant import Hypergraph

# Traditional usage (pandas backend)
hg = Hypergraph({'e1': ['a', 'b'], 'e2': ['b', 'c']})
centrality = hg.s_centrality()

# Enhanced performance (polars backend) 
hg_fast = Hypergraph(
    {'e1': ['a', 'b'], 'e2': ['b', 'c']},
    storage_backend="polars"
)
```

### 2. **Advanced Metagraph Usage**

```python
from anant import Metagraph
from pathlib import Path

# Create metagraph with LLM integration
mg = Metagraph(
    hierarchical_data=complex_enterprise_data,
    storage_path=Path("./enterprise_knowledge"),
    llm_integration=True
)

# Semantic analysis
patterns = mg.semantic_layer.find_semantic_patterns("clusters")

# Temporal tracking
mg.temporal_layer.track_relationship_evolution()

# Convert from hypergraph
traditional_hg = Hypergraph(my_data)
enhanced_mg = traditional_hg.to_metagraph(
    semantic_enrichment=True,
    temporal_tracking=True
)
```

### 3. **Metadata-Driven Analytics**

```python
# Store rich metadata with Polars+Parquet
mg.metadata_store.store_entity_metadata(
    entity_id="customer_001",
    metadata={
        "type": "enterprise_customer",
        "industry": "technology", 
        "compliance_requirements": ["SOX", "GDPR"],
        "data_lineage": {...},
        "quality_metrics": {...}
    },
    entity_type="business_entity"
)

# Query metadata efficiently
compliance_entities = mg.metadata_store.query_entities(
    filters={"compliance_requirements": {"contains": "GDPR"}},
    entity_type="business_entity"
)
```

---

## 📊 Performance Considerations

### 1. **Memory Management**

```python
# /anant/core/memory_management.py
import polars as pl
from ..config.settings import settings

class MemoryManager:
    """Manage memory usage for large datasets"""
    
    @staticmethod
    def optimize_polars_config():
        """Optimize Polars for available memory"""
        pl.Config.set_streaming_chunk_size(
            int(settings.max_memory_gb * 1024 * 1024 * 0.1)  # 10% of max memory
        )
        
        if settings.polars_streaming:
            pl.Config.set_auto_structify(True)
            
    @staticmethod
    def estimate_memory_usage(df: pl.DataFrame) -> float:
        """Estimate memory usage in MB"""
        return df.estimated_size("mb")
```

### 2. **Lazy Evaluation Strategy**

```python
# Use lazy evaluation for large datasets
lazy_df = pl.scan_parquet("large_dataset.parquet")
result = (
    lazy_df
    .filter(pl.col("importance") > 0.5)
    .group_by("category")
    .agg(pl.col("value").mean())
    .collect()  # Only execute when needed
)
```

---

## 🎯 Key Benefits of This Architecture

### 1. **User Choice & Backward Compatibility**
- Users can choose Hypergraph for traditional use cases
- Full backward compatibility with existing HyperNetX code
- Opt-in to Metagraph features when needed

### 2. **Optimized Storage**
- Polars+Parquet for metadata provides:
  - 5-10x faster metadata queries
  - 50-80% memory reduction
  - Native compression support
  - Schema evolution capabilities

### 3. **Modular Dependencies**
- Optional dependencies based on use case
- Minimal installation for basic usage
- Full features available when needed

### 4. **Future-Proof Architecture**
- Clean separation between Hypergraph and Metagraph
- Easy to add new storage backends
- Extensible for additional LLM providers

This architecture gives users the flexibility to choose their preferred interface while leveraging the optimal storage technology (Polars+Parquet) specifically for the advanced Metagraph metadata capabilities.
