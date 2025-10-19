"""
Database Connectivity for Anant Library

Provides direct database integration for hypergraphs:
- SQLite support for lightweight embedded storage
- PostgreSQL integration for production systems
- MongoDB support for document-based storage
- Generic database interface for custom connectors
- Schema management and migration utilities
- Optimized batch operations for large datasets

This module enables persistent storage and retrieval of hypergraphs
from various database systems with automatic schema management.
"""

import sqlite3
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Tuple, AsyncGenerator
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import polars as pl
import numpy as np

from ..classes.hypergraph import Hypergraph


@dataclass
class DatabaseConfig:
    """Configuration for database connections"""
    host: str = "localhost"
    port: int = 5432
    database: str = "hypergraphs"
    username: Optional[str] = None
    password: Optional[str] = None
    schema: str = "public"
    table_prefix: str = "hnx_"
    batch_size: int = 1000
    connection_timeout: int = 30
    enable_ssl: bool = False
    additional_params: Optional[Dict[str, Any]] = None


class DatabaseConnector(ABC):
    """Abstract base class for database connectors"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.connection = None
    
    @abstractmethod
    async def connect(self) -> None:
        """Establish database connection"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close database connection"""
        pass
    
    @abstractmethod
    async def create_schema(self) -> None:
        """Create required database schema"""
        pass
    
    @abstractmethod
    async def save_hypergraph(self, hg: Hypergraph, name: str, 
                             metadata: Optional[Dict[str, Any]] = None) -> str:
        """Save hypergraph to database"""
        pass
    
    @abstractmethod
    async def load_hypergraph(self, name: str) -> Hypergraph:
        """Load hypergraph from database"""
        pass
    
    @abstractmethod
    async def list_hypergraphs(self) -> List[Dict[str, Any]]:
        """List all stored hypergraphs"""
        pass
    
    @abstractmethod
    async def delete_hypergraph(self, name: str) -> bool:
        """Delete hypergraph from database"""
        pass


class SQLiteConnector(DatabaseConnector):
    """SQLite database connector for lightweight storage"""
    
    def __init__(self, config: DatabaseConfig, db_path: Union[str, Path]):
        super().__init__(config)
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
    
    async def connect(self) -> None:
        """Establish SQLite connection"""
        self.connection = sqlite3.connect(
            self.db_path,
            timeout=self.config.connection_timeout,
            check_same_thread=False
        )
        self.connection.row_factory = sqlite3.Row
        await self.create_schema()
    
    async def disconnect(self) -> None:
        """Close SQLite connection"""
        if self.connection:
            self.connection.close()
            self.connection = None
    
    async def create_schema(self) -> None:
        """Create SQLite schema for hypergraphs"""
        cursor = self.connection.cursor()
        
        # Hypergraphs metadata table
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.config.table_prefix}hypergraphs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                num_nodes INTEGER NOT NULL,
                num_edges INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT,
                description TEXT
            )
        """)
        
        # Nodes table
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.config.table_prefix}nodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hypergraph_name TEXT NOT NULL,
                node_id TEXT NOT NULL,
                properties TEXT,
                FOREIGN KEY (hypergraph_name) REFERENCES {self.config.table_prefix}hypergraphs(name) ON DELETE CASCADE,
                UNIQUE(hypergraph_name, node_id)
            )
        """)
        
        # Edges table
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.config.table_prefix}edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hypergraph_name TEXT NOT NULL,
                edge_id TEXT NOT NULL,
                properties TEXT,
                size INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (hypergraph_name) REFERENCES {self.config.table_prefix}hypergraphs(name) ON DELETE CASCADE,
                UNIQUE(hypergraph_name, edge_id)
            )
        """)
        
        # Incidences table
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.config.table_prefix}incidences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hypergraph_name TEXT NOT NULL,
                node_id TEXT NOT NULL,
                edge_id TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                FOREIGN KEY (hypergraph_name) REFERENCES {self.config.table_prefix}hypergraphs(name) ON DELETE CASCADE,
                UNIQUE(hypergraph_name, node_id, edge_id)
            )
        """)
        
        # Create indexes for performance
        cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{self.config.table_prefix}nodes_graph 
            ON {self.config.table_prefix}nodes(hypergraph_name)
        """)
        
        cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{self.config.table_prefix}edges_graph 
            ON {self.config.table_prefix}edges(hypergraph_name)
        """)
        
        cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{self.config.table_prefix}incidences_graph 
            ON {self.config.table_prefix}incidences(hypergraph_name)
        """)
        
        cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{self.config.table_prefix}incidences_node 
            ON {self.config.table_prefix}incidences(node_id)
        """)
        
        cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{self.config.table_prefix}incidences_edge 
            ON {self.config.table_prefix}incidences(edge_id)
        """)
        
        self.connection.commit()
    
    async def save_hypergraph(self, hg: Hypergraph, name: str, 
                             metadata: Optional[Dict[str, Any]] = None) -> str:
        """Save hypergraph to SQLite database"""
        cursor = self.connection.cursor()
        
        try:
            # Start transaction
            cursor.execute("BEGIN TRANSACTION")
            
            # Delete existing hypergraph if it exists
            await self.delete_hypergraph(name)
            
            # Insert hypergraph metadata
            cursor.execute(f"""
                INSERT INTO {self.config.table_prefix}hypergraphs 
                (name, num_nodes, num_edges, metadata, description)
                VALUES (?, ?, ?, ?, ?)
            """, (
                name,
                hg.num_nodes,
                hg.num_edges,
                json.dumps(metadata) if metadata else None,
                getattr(hg, 'description', None)
            ))
            
            # Insert nodes in batches
            nodes_data = [(name, str(node), None) for node in hg.nodes]
            for i in range(0, len(nodes_data), self.config.batch_size):
                batch = nodes_data[i:i + self.config.batch_size]
                cursor.executemany(f"""
                    INSERT INTO {self.config.table_prefix}nodes 
                    (hypergraph_name, node_id, properties)
                    VALUES (?, ?, ?)
                """, batch)
            
            # Insert edges in batches
            edges_data = []
            for edge_id in hg.edges:
                edge_size = hg.get_edge_size(edge_id)
                edges_data.append((name, str(edge_id), None, edge_size))
            
            for i in range(0, len(edges_data), self.config.batch_size):
                batch = edges_data[i:i + self.config.batch_size]
                cursor.executemany(f"""
                    INSERT INTO {self.config.table_prefix}edges 
                    (hypergraph_name, edge_id, properties, size)
                    VALUES (?, ?, ?, ?)
                """, batch)
            
            # Insert incidences in batches
            incidence_data = hg.incidences.data
            incidences_list = []
            for row in incidence_data.iter_rows(named=True):
                incidences_list.append((
                    name,
                    str(row["node_id"]),
                    str(row["edge_id"]),
                    float(row.get("weight", 1.0))
                ))
            
            for i in range(0, len(incidences_list), self.config.batch_size):
                batch = incidences_list[i:i + self.config.batch_size]
                cursor.executemany(f"""
                    INSERT INTO {self.config.table_prefix}incidences 
                    (hypergraph_name, node_id, edge_id, weight)
                    VALUES (?, ?, ?, ?)
                """, batch)
            
            # Commit transaction
            cursor.execute("COMMIT")
            return name
            
        except Exception as e:
            cursor.execute("ROLLBACK")
            raise e
    
    async def load_hypergraph(self, name: str) -> Hypergraph:
        """Load hypergraph from SQLite database"""
        cursor = self.connection.cursor()
        
        # Check if hypergraph exists
        cursor.execute(f"""
            SELECT * FROM {self.config.table_prefix}hypergraphs 
            WHERE name = ?
        """, (name,))
        
        hg_row = cursor.fetchone()
        if not hg_row:
            raise ValueError(f"Hypergraph '{name}' not found")
        
        # Load incidences
        cursor.execute(f"""
            SELECT node_id, edge_id, weight 
            FROM {self.config.table_prefix}incidences 
            WHERE hypergraph_name = ?
        """, (name,))
        
        incidences = []
        for row in cursor.fetchall():
            incidences.append({
                "node_id": row["node_id"],
                "edge_id": row["edge_id"],
                "weight": float(row["weight"])
            })
        
        # Create hypergraph
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
        if hg_row["description"]:
            hg.description = hg_row["description"]
        
        return hg
    
    async def list_hypergraphs(self) -> List[Dict[str, Any]]:
        """List all stored hypergraphs"""
        cursor = self.connection.cursor()
        
        cursor.execute(f"""
            SELECT name, num_nodes, num_edges, created_at, updated_at, 
                   metadata, description
            FROM {self.config.table_prefix}hypergraphs
            ORDER BY updated_at DESC
        """)
        
        hypergraphs = []
        for row in cursor.fetchall():
            hg_info = {
                "name": row["name"],
                "num_nodes": row["num_nodes"],
                "num_edges": row["num_edges"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "description": row["description"]
            }
            
            if row["metadata"]:
                try:
                    hg_info["metadata"] = json.loads(row["metadata"])
                except json.JSONDecodeError:
                    hg_info["metadata"] = {}
            
            hypergraphs.append(hg_info)
        
        return hypergraphs
    
    async def delete_hypergraph(self, name: str) -> bool:
        """Delete hypergraph from SQLite database"""
        cursor = self.connection.cursor()
        
        cursor.execute(f"""
            DELETE FROM {self.config.table_prefix}hypergraphs 
            WHERE name = ?
        """, (name,))
        
        deleted = cursor.rowcount > 0
        self.connection.commit()
        return deleted


class PostgreSQLConnector(DatabaseConnector):
    """PostgreSQL database connector for production systems"""
    
    def __init__(self, config: DatabaseConfig):
        super().__init__(config)
        self._pool = None
    
    async def connect(self) -> None:
        """Establish PostgreSQL connection pool"""
        try:
            import asyncpg
        except ImportError:
            raise ImportError("asyncpg required for PostgreSQL support. Install with: pip install asyncpg")
        
        # Build connection URL
        conn_url = f"postgresql://{self.config.username}:{self.config.password}@{self.config.host}:{self.config.port}/{self.config.database}"
        
        # Create connection pool
        self._pool = await asyncpg.create_pool(
            conn_url,
            min_size=1,
            max_size=10,
            command_timeout=self.config.connection_timeout
        )
        
        await self.create_schema()
    
    async def disconnect(self) -> None:
        """Close PostgreSQL connection pool"""
        if self._pool:
            await self._pool.close()
            self._pool = None
    
    async def create_schema(self) -> None:
        """Create PostgreSQL schema for hypergraphs"""
        async with self._pool.acquire() as conn:
            # Create schema if it doesn't exist
            await conn.execute(f"CREATE SCHEMA IF NOT EXISTS {self.config.schema}")
            
            # Hypergraphs metadata table
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.config.schema}.{self.config.table_prefix}hypergraphs (
                    id SERIAL PRIMARY KEY,
                    name TEXT UNIQUE NOT NULL,
                    num_nodes INTEGER NOT NULL,
                    num_edges INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata JSONB,
                    description TEXT
                )
            """)
            
            # Nodes table
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.config.schema}.{self.config.table_prefix}nodes (
                    id SERIAL PRIMARY KEY,
                    hypergraph_name TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    properties JSONB,
                    FOREIGN KEY (hypergraph_name) REFERENCES {self.config.schema}.{self.config.table_prefix}hypergraphs(name) ON DELETE CASCADE,
                    UNIQUE(hypergraph_name, node_id)
                )
            """)
            
            # Edges table
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.config.schema}.{self.config.table_prefix}edges (
                    id SERIAL PRIMARY KEY,
                    hypergraph_name TEXT NOT NULL,
                    edge_id TEXT NOT NULL,
                    properties JSONB,
                    size INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY (hypergraph_name) REFERENCES {self.config.schema}.{self.config.table_prefix}hypergraphs(name) ON DELETE CASCADE,
                    UNIQUE(hypergraph_name, edge_id)
                )
            """)
            
            # Incidences table
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.config.schema}.{self.config.table_prefix}incidences (
                    id SERIAL PRIMARY KEY,
                    hypergraph_name TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    edge_id TEXT NOT NULL,
                    weight REAL DEFAULT 1.0,
                    FOREIGN KEY (hypergraph_name) REFERENCES {self.config.schema}.{self.config.table_prefix}hypergraphs(name) ON DELETE CASCADE,
                    UNIQUE(hypergraph_name, node_id, edge_id)
                )
            """)
            
            # Create indexes for performance
            indexes = [
                f"CREATE INDEX IF NOT EXISTS idx_{self.config.table_prefix}nodes_graph ON {self.config.schema}.{self.config.table_prefix}nodes(hypergraph_name)",
                f"CREATE INDEX IF NOT EXISTS idx_{self.config.table_prefix}edges_graph ON {self.config.schema}.{self.config.table_prefix}edges(hypergraph_name)",
                f"CREATE INDEX IF NOT EXISTS idx_{self.config.table_prefix}incidences_graph ON {self.config.schema}.{self.config.table_prefix}incidences(hypergraph_name)",
                f"CREATE INDEX IF NOT EXISTS idx_{self.config.table_prefix}incidences_node ON {self.config.schema}.{self.config.table_prefix}incidences(node_id)",
                f"CREATE INDEX IF NOT EXISTS idx_{self.config.table_prefix}incidences_edge ON {self.config.schema}.{self.config.table_prefix}incidences(edge_id)"
            ]
            
            for index_sql in indexes:
                await conn.execute(index_sql)
    
    async def save_hypergraph(self, hg: Hypergraph, name: str, 
                             metadata: Optional[Dict[str, Any]] = None) -> str:
        """Save hypergraph to PostgreSQL database"""
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                # Delete existing hypergraph if it exists
                await self.delete_hypergraph(name)
                
                # Insert hypergraph metadata
                await conn.execute(f"""
                    INSERT INTO {self.config.schema}.{self.config.table_prefix}hypergraphs 
                    (name, num_nodes, num_edges, metadata, description)
                    VALUES ($1, $2, $3, $4, $5)
                """, name, hg.num_nodes, hg.num_edges, 
                   json.dumps(metadata) if metadata else None,
                   getattr(hg, 'description', None))
                
                # Insert nodes in batches
                nodes_data = [(name, str(node), None) for node in hg.nodes]
                await conn.executemany(f"""
                    INSERT INTO {self.config.schema}.{self.config.table_prefix}nodes 
                    (hypergraph_name, node_id, properties)
                    VALUES ($1, $2, $3)
                """, nodes_data)
                
                # Insert edges in batches
                edges_data = []
                for edge_id in hg.edges:
                    edge_size = hg.get_edge_size(edge_id)
                    edges_data.append((name, str(edge_id), None, edge_size))
                
                await conn.executemany(f"""
                    INSERT INTO {self.config.schema}.{self.config.table_prefix}edges 
                    (hypergraph_name, edge_id, properties, size)
                    VALUES ($1, $2, $3, $4)
                """, edges_data)
                
                # Insert incidences in batches
                incidence_data = hg.incidences.data
                incidences_list = []
                for row in incidence_data.iter_rows(named=True):
                    incidences_list.append((
                        name,
                        str(row["node_id"]),
                        str(row["edge_id"]),
                        float(row.get("weight", 1.0))
                    ))
                
                await conn.executemany(f"""
                    INSERT INTO {self.config.schema}.{self.config.table_prefix}incidences 
                    (hypergraph_name, node_id, edge_id, weight)
                    VALUES ($1, $2, $3, $4)
                """, incidences_list)
        
        return name
    
    async def load_hypergraph(self, name: str) -> Hypergraph:
        """Load hypergraph from PostgreSQL database"""
        async with self._pool.acquire() as conn:
            # Check if hypergraph exists
            hg_row = await conn.fetchrow(f"""
                SELECT * FROM {self.config.schema}.{self.config.table_prefix}hypergraphs 
                WHERE name = $1
            """, name)
            
            if not hg_row:
                raise ValueError(f"Hypergraph '{name}' not found")
            
            # Load incidences
            incidence_rows = await conn.fetch(f"""
                SELECT node_id, edge_id, weight 
                FROM {self.config.schema}.{self.config.table_prefix}incidences 
                WHERE hypergraph_name = $1
            """, name)
            
            incidences = []
            for row in incidence_rows:
                incidences.append({
                    "node_id": row["node_id"],
                    "edge_id": row["edge_id"],
                    "weight": float(row["weight"])
                })
            
            # Create hypergraph
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
            if hg_row["description"]:
                hg.description = hg_row["description"]
            
            return hg
    
    async def list_hypergraphs(self) -> List[Dict[str, Any]]:
        """List all stored hypergraphs"""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(f"""
                SELECT name, num_nodes, num_edges, created_at, updated_at, 
                       metadata, description
                FROM {self.config.schema}.{self.config.table_prefix}hypergraphs
                ORDER BY updated_at DESC
            """)
            
            hypergraphs = []
            for row in rows:
                hg_info = {
                    "name": row["name"],
                    "num_nodes": row["num_nodes"],
                    "num_edges": row["num_edges"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "description": row["description"],
                    "metadata": row["metadata"] or {}
                }
                hypergraphs.append(hg_info)
            
            return hypergraphs
    
    async def delete_hypergraph(self, name: str) -> bool:
        """Delete hypergraph from PostgreSQL database"""
        async with self._pool.acquire() as conn:
            result = await conn.execute(f"""
                DELETE FROM {self.config.schema}.{self.config.table_prefix}hypergraphs 
                WHERE name = $1
            """, name)
            
            return result != "DELETE 0"


class MongoDBConnector(DatabaseConnector):
    """MongoDB connector for document-based storage"""
    
    def __init__(self, config: DatabaseConfig):
        super().__init__(config)
        self.client = None
        self.db = None
    
    async def connect(self) -> None:
        """Establish MongoDB connection"""
        try:
            from motor.motor_asyncio import AsyncIOMotorClient
        except ImportError:
            raise ImportError("motor required for MongoDB support. Install with: pip install motor")
        
        # Build connection URL
        if self.config.username and self.config.password:
            conn_url = f"mongodb://{self.config.username}:{self.config.password}@{self.config.host}:{self.config.port}/{self.config.database}"
        else:
            conn_url = f"mongodb://{self.config.host}:{self.config.port}"
        
        self.client = AsyncIOMotorClient(
            conn_url,
            serverSelectionTimeoutMS=self.config.connection_timeout * 1000
        )
        self.db = self.client[self.config.database]
        
        await self.create_schema()
    
    async def disconnect(self) -> None:
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            self.client = None
            self.db = None
    
    async def create_schema(self) -> None:
        """Create MongoDB collections and indexes"""
        # Create collections
        collections = [
            f"{self.config.table_prefix}hypergraphs",
            f"{self.config.table_prefix}incidences"
        ]
        
        for collection_name in collections:
            if collection_name not in await self.db.list_collection_names():
                await self.db.create_collection(collection_name)
        
        # Create indexes
        hypergraphs_collection = self.db[f"{self.config.table_prefix}hypergraphs"]
        await hypergraphs_collection.create_index("name", unique=True)
        
        incidences_collection = self.db[f"{self.config.table_prefix}incidences"]
        await incidences_collection.create_index([("hypergraph_name", 1), ("edge_id", 1)])
        await incidences_collection.create_index([("hypergraph_name", 1), ("node_id", 1)])
    
    async def save_hypergraph(self, hg: Hypergraph, name: str, 
                             metadata: Optional[Dict[str, Any]] = None) -> str:
        """Save hypergraph to MongoDB"""
        # Delete existing hypergraph if it exists
        await self.delete_hypergraph(name)
        
        # Prepare hypergraph document
        hg_doc = {
            "name": name,
            "num_nodes": hg.num_nodes,
            "num_edges": hg.num_edges,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "metadata": metadata or {},
            "description": getattr(hg, 'description', None)
        }
        
        # Insert hypergraph metadata
        hypergraphs_collection = self.db[f"{self.config.table_prefix}hypergraphs"]
        await hypergraphs_collection.insert_one(hg_doc)
        
        # Prepare incidences
        incidence_docs = []
        incidence_data = hg.incidences.data
        for row in incidence_data.iter_rows(named=True):
            incidence_docs.append({
                "hypergraph_name": name,
                "node_id": str(row["node_id"]),
                "edge_id": str(row["edge_id"]),
                "weight": float(row.get("weight", 1.0))
            })
        
        # Insert incidences in batches
        if incidence_docs:
            incidences_collection = self.db[f"{self.config.table_prefix}incidences"]
            for i in range(0, len(incidence_docs), self.config.batch_size):
                batch = incidence_docs[i:i + self.config.batch_size]
                await incidences_collection.insert_many(batch)
        
        return name
    
    async def load_hypergraph(self, name: str) -> Hypergraph:
        """Load hypergraph from MongoDB"""
        # Check if hypergraph exists
        hypergraphs_collection = self.db[f"{self.config.table_prefix}hypergraphs"]
        hg_doc = await hypergraphs_collection.find_one({"name": name})
        
        if not hg_doc:
            raise ValueError(f"Hypergraph '{name}' not found")
        
        # Load incidences
        incidences_collection = self.db[f"{self.config.table_prefix}incidences"]
        incidence_docs = await incidences_collection.find(
            {"hypergraph_name": name}
        ).to_list(length=None)
        
        incidences = []
        for doc in incidence_docs:
            incidences.append({
                "node_id": doc["node_id"],
                "edge_id": doc["edge_id"],
                "weight": float(doc["weight"])
            })
        
        # Create hypergraph
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
        if hg_doc.get("description"):
            hg.description = hg_doc["description"]
        
        return hg
    
    async def list_hypergraphs(self) -> List[Dict[str, Any]]:
        """List all stored hypergraphs"""
        hypergraphs_collection = self.db[f"{self.config.table_prefix}hypergraphs"]
        
        cursor = hypergraphs_collection.find(
            {},
            {"_id": 0, "name": 1, "num_nodes": 1, "num_edges": 1, 
             "created_at": 1, "updated_at": 1, "metadata": 1, "description": 1}
        ).sort("updated_at", -1)
        
        hypergraphs = []
        async for doc in cursor:
            hypergraphs.append(doc)
        
        return hypergraphs
    
    async def delete_hypergraph(self, name: str) -> bool:
        """Delete hypergraph from MongoDB"""
        # Delete hypergraph metadata
        hypergraphs_collection = self.db[f"{self.config.table_prefix}hypergraphs"]
        hg_result = await hypergraphs_collection.delete_one({"name": name})
        
        # Delete incidences
        incidences_collection = self.db[f"{self.config.table_prefix}incidences"]
        await incidences_collection.delete_many({"hypergraph_name": name})
        
        return hg_result.deleted_count > 0


class DatabaseManager:
    """High-level database manager for hypergraphs"""
    
    def __init__(self, connector: DatabaseConnector):
        self.connector = connector
        self._connected = False
    
    async def __aenter__(self):
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()
    
    async def connect(self) -> None:
        """Connect to database"""
        if not self._connected:
            await self.connector.connect()
            self._connected = True
    
    async def disconnect(self) -> None:
        """Disconnect from database"""
        if self._connected:
            await self.connector.disconnect()
            self._connected = False
    
    async def save(self, hg: Hypergraph, name: str, 
                   metadata: Optional[Dict[str, Any]] = None) -> str:
        """Save hypergraph to database"""
        if not self._connected:
            await self.connect()
        return await self.connector.save_hypergraph(hg, name, metadata)
    
    async def load(self, name: str) -> Hypergraph:
        """Load hypergraph from database"""
        if not self._connected:
            await self.connect()
        return await self.connector.load_hypergraph(name)
    
    async def list(self) -> List[Dict[str, Any]]:
        """List all stored hypergraphs"""
        if not self._connected:
            await self.connect()
        return await self.connector.list_hypergraphs()
    
    async def delete(self, name: str) -> bool:
        """Delete hypergraph from database"""
        if not self._connected:
            await self.connect()
        return await self.connector.delete_hypergraph(name)


# Convenience functions
def create_sqlite_manager(db_path: Union[str, Path], 
                         config: Optional[DatabaseConfig] = None) -> DatabaseManager:
    """Create SQLite database manager"""
    config = config or DatabaseConfig()
    connector = SQLiteConnector(config, db_path)
    return DatabaseManager(connector)


def create_postgresql_manager(host: str, database: str, username: str, password: str,
                             port: int = 5432, 
                             config: Optional[DatabaseConfig] = None) -> DatabaseManager:
    """Create PostgreSQL database manager"""
    if config is None:
        config = DatabaseConfig()
    config.host = host
    config.port = port
    config.database = database
    config.username = username
    config.password = password
    
    connector = PostgreSQLConnector(config)
    return DatabaseManager(connector)


def create_mongodb_manager(host: str, database: str, 
                          username: Optional[str] = None, password: Optional[str] = None,
                          port: int = 27017,
                          config: Optional[DatabaseConfig] = None) -> DatabaseManager:
    """Create MongoDB database manager"""
    if config is None:
        config = DatabaseConfig()
    config.host = host
    config.port = port
    config.database = database
    config.username = username
    config.password = password
    
    connector = MongoDBConnector(config)
    return DatabaseManager(connector)