"""
Cluster Manager - Central Cluster Coordination and Node Management

Manages the distributed cluster topology, node discovery, registration,
health monitoring, and resource allocation across the compute cluster.
"""

import asyncio
import time
import threading
import socket
import uuid
from typing import Dict, List, Optional, Set, Callable, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
import platform

# Optional imports with graceful fallbacks
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

try:
    import docker
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False
    docker = None

logger = logging.getLogger(__name__)


class NodeStatus(Enum):
    """Status of cluster nodes."""
    UNKNOWN = "unknown"
    STARTING = "starting"
    ACTIVE = "active"
    BUSY = "busy"
    IDLE = "idle"
    OVERLOADED = "overloaded"
    DRAINING = "draining"
    FAILED = "failed"
    OFFLINE = "offline"


class NodeType(Enum):
    """Types of cluster nodes."""
    COORDINATOR = "coordinator"
    WORKER = "worker"
    STORAGE = "storage"
    GATEWAY = "gateway"
    HYBRID = "hybrid"


@dataclass
class ResourceCapacity:
    """Resource capacity information for a node."""
    cpu_cores: int = 0
    memory_gb: float = 0.0
    disk_gb: float = 0.0
    gpu_count: int = 0
    gpu_memory_gb: float = 0.0
    network_bandwidth_mbps: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'cpu_cores': self.cpu_cores,
            'memory_gb': self.memory_gb,
            'disk_gb': self.disk_gb,
            'gpu_count': self.gpu_count,
            'gpu_memory_gb': self.gpu_memory_gb,
            'network_bandwidth_mbps': self.network_bandwidth_mbps
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ResourceCapacity':
        return cls(**data)


@dataclass
class ResourceUsage:
    """Current resource usage for a node."""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    disk_percent: float = 0.0
    gpu_percent: float = 0.0
    gpu_memory_percent: float = 0.0
    network_io_mbps: float = 0.0
    active_tasks: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'disk_percent': self.disk_percent,
            'gpu_percent': self.gpu_percent,
            'gpu_memory_percent': self.gpu_memory_percent,
            'network_io_mbps': self.network_io_mbps,
            'active_tasks': self.active_tasks
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ResourceUsage':
        return cls(**data)


@dataclass
class NodeInfo:
    """Comprehensive information about a cluster node."""
    node_id: str
    node_type: NodeType
    hostname: str
    ip_address: str
    port: int
    status: NodeStatus = NodeStatus.UNKNOWN
    capacity: ResourceCapacity = field(default_factory=ResourceCapacity)
    usage: ResourceUsage = field(default_factory=ResourceUsage)
    last_heartbeat: float = field(default_factory=time.time)
    joined_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'node_id': self.node_id,
            'node_type': self.node_type.value,
            'hostname': self.hostname,
            'ip_address': self.ip_address,
            'port': self.port,
            'status': self.status.value,
            'capacity': self.capacity.to_dict(),
            'usage': self.usage.to_dict(),
            'last_heartbeat': self.last_heartbeat,
            'joined_at': self.joined_at,
            'metadata': self.metadata,
            'tags': list(self.tags)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NodeInfo':
        return cls(
            node_id=data['node_id'],
            node_type=NodeType(data['node_type']),
            hostname=data['hostname'],
            ip_address=data['ip_address'],
            port=data['port'],
            status=NodeStatus(data['status']),
            capacity=ResourceCapacity.from_dict(data['capacity']),
            usage=ResourceUsage.from_dict(data['usage']),
            last_heartbeat=data['last_heartbeat'],
            joined_at=data['joined_at'],
            metadata=data['metadata'],
            tags=set(data['tags'])
        )


@dataclass
class ClusterConfig:
    """Configuration for cluster management."""
    cluster_name: str = "anant-cluster"
    coordinator_port: int = 8888
    heartbeat_interval: float = 30.0
    heartbeat_timeout: float = 90.0
    discovery_port: int = 8889
    max_nodes: int = 1000
    auto_scaling: bool = True
    load_balancing: bool = True
    fault_tolerance: bool = True
    security_enabled: bool = False
    encryption_key: Optional[str] = None


class ClusterManager:
    """
    Central cluster manager for coordinating distributed nodes.
    
    Handles node discovery, registration, health monitoring, load balancing,
    and resource allocation across the compute cluster.
    """
    
    def __init__(self, config: Optional[ClusterConfig] = None):
        """
        Initialize cluster manager.
        
        Args:
            config: Cluster configuration
        """
        self.config = config or ClusterConfig()
        self.nodes: Dict[str, NodeInfo] = {}
        self.node_groups: Dict[str, Set[str]] = {}  # group_name -> node_ids
        self._lock = threading.RLock()
        self._running = False
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._discovery_thread: Optional[threading.Thread] = None
        
        # Event callbacks
        self._node_joined_callbacks: List[Callable[[NodeInfo], None]] = []
        self._node_left_callbacks: List[Callable[[NodeInfo], None]] = []
        self._node_failed_callbacks: List[Callable[[NodeInfo], None]] = []
        
        # Statistics
        self._stats = {
            'nodes_joined': 0,
            'nodes_left': 0,
            'nodes_failed': 0,
            'heartbeats_received': 0,
            'heartbeats_missed': 0
        }
        
        # Get local node info
        self._local_node = self._create_local_node_info()
        
    def _create_local_node_info(self) -> NodeInfo:
        """Create node info for the local coordinator."""
        node_id = f"coordinator-{uuid.uuid4().hex[:8]}"
        hostname = socket.gethostname()
        
        try:
            # Get local IP address
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip_address = s.getsockname()[0]
            s.close()
        except Exception:
            ip_address = "127.0.0.1"
            
        # Get system resources
        capacity = self._get_system_capacity()
        
        return NodeInfo(
            node_id=node_id,
            node_type=NodeType.COORDINATOR,
            hostname=hostname,
            ip_address=ip_address,
            port=self.config.coordinator_port,
            status=NodeStatus.ACTIVE,
            capacity=capacity,
            metadata={
                'platform': platform.platform(),
                'python_version': platform.python_version(),
                'coordinator': True
            }
        )
        
    def _get_system_capacity(self) -> ResourceCapacity:
        """Get system resource capacity."""
        capacity = ResourceCapacity()
        
        if PSUTIL_AVAILABLE:
            try:
                # CPU information
                capacity.cpu_cores = psutil.cpu_count(logical=True)
                
                # Memory information
                memory = psutil.virtual_memory()
                capacity.memory_gb = memory.total / (1024**3)
                
                # Disk information
                disk = psutil.disk_usage('/')
                capacity.disk_gb = disk.total / (1024**3)
                
                # Network information (estimate)
                capacity.network_bandwidth_mbps = 1000.0  # Default 1Gbps
                
                # GPU information (basic detection)
                try:
                    import nvidia_ml_py3 as nvml
                    nvml.nvmlInit()
                    capacity.gpu_count = nvml.nvmlDeviceGetCount()
                    
                    if capacity.gpu_count > 0:
                        handle = nvml.nvmlDeviceGetHandleByIndex(0)
                        info = nvml.nvmlDeviceGetMemoryInfo(handle)
                        capacity.gpu_memory_gb = info.total / (1024**3)
                        
                except ImportError:
                    pass
                    
            except Exception as e:
                logger.warning(f"Failed to get system capacity: {e}")
                
        return capacity
        
    def start(self) -> bool:
        """
        Start the cluster manager.
        
        Returns:
            True if started successfully
        """
        if self._running:
            return True
            
        try:
            # Register local coordinator node
            self.register_node(self._local_node)
            
            # Start background threads
            self._running = True
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_monitor, daemon=True
            )
            self._heartbeat_thread.start()
            
            self._discovery_thread = threading.Thread(
                target=self._discovery_service, daemon=True
            )
            self._discovery_thread.start()
            
            logger.info(f"Cluster manager started: {self._local_node.node_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start cluster manager: {e}")
            self._running = False
            return False
            
    def stop(self):
        """Stop the cluster manager."""
        self._running = False
        
        # Wait for threads to finish
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=5.0)
            
        if self._discovery_thread and self._discovery_thread.is_alive():
            self._discovery_thread.join(timeout=5.0)
            
        logger.info("Cluster manager stopped")
        
    def register_node(self, node_info: NodeInfo) -> bool:
        """
        Register a new node with the cluster.
        
        Args:
            node_info: Information about the node to register
            
        Returns:
            True if registration successful
        """
        with self._lock:
            if len(self.nodes) >= self.config.max_nodes:
                logger.warning(f"Cluster at maximum capacity: {self.config.max_nodes}")
                return False
                
            if node_info.node_id in self.nodes:
                logger.info(f"Node already registered: {node_info.node_id}")
                # Update existing node info
                self.nodes[node_info.node_id] = node_info
            else:
                # New node registration
                node_info.joined_at = time.time()
                node_info.last_heartbeat = time.time()
                self.nodes[node_info.node_id] = node_info
                self._stats['nodes_joined'] += 1
                
                # Trigger callbacks
                for callback in self._node_joined_callbacks:
                    try:
                        callback(node_info)
                    except Exception as e:
                        logger.warning(f"Node joined callback failed: {e}")
                        
                logger.info(f"Node registered: {node_info.node_id} ({node_info.node_type.value})")
                
            return True
            
    def unregister_node(self, node_id: str) -> bool:
        """
        Unregister a node from the cluster.
        
        Args:
            node_id: ID of the node to unregister
            
        Returns:
            True if unregistration successful
        """
        with self._lock:
            if node_id not in self.nodes:
                return False
                
            node_info = self.nodes[node_id]
            del self.nodes[node_id]
            self._stats['nodes_left'] += 1
            
            # Remove from node groups
            for group_nodes in self.node_groups.values():
                group_nodes.discard(node_id)
                
            # Trigger callbacks
            for callback in self._node_left_callbacks:
                try:
                    callback(node_info)
                except Exception as e:
                    logger.warning(f"Node left callback failed: {e}")
                    
            logger.info(f"Node unregistered: {node_id}")
            return True
            
    def update_node_status(self, node_id: str, status: NodeStatus) -> bool:
        """
        Update the status of a node.
        
        Args:
            node_id: ID of the node
            status: New status
            
        Returns:
            True if update successful
        """
        with self._lock:
            if node_id not in self.nodes:
                return False
                
            old_status = self.nodes[node_id].status
            self.nodes[node_id].status = status
            self.nodes[node_id].last_heartbeat = time.time()
            
            # Log status changes
            if old_status != status:
                logger.info(f"Node {node_id} status changed: {old_status.value} -> {status.value}")
                
                # Handle failures
                if status == NodeStatus.FAILED:
                    self._handle_node_failure(self.nodes[node_id])
                    
            return True
            
    def update_node_usage(self, node_id: str, usage: ResourceUsage) -> bool:
        """
        Update resource usage for a node.
        
        Args:
            node_id: ID of the node
            usage: Current resource usage
            
        Returns:
            True if update successful
        """
        with self._lock:
            if node_id not in self.nodes:
                return False
                
            self.nodes[node_id].usage = usage
            self.nodes[node_id].last_heartbeat = time.time()
            
            # Check for overload conditions
            if (usage.cpu_percent > 90 or usage.memory_percent > 90):
                if self.nodes[node_id].status != NodeStatus.OVERLOADED:
                    self.update_node_status(node_id, NodeStatus.OVERLOADED)
            elif self.nodes[node_id].status == NodeStatus.OVERLOADED:
                if usage.cpu_percent < 70 and usage.memory_percent < 70:
                    self.update_node_status(node_id, NodeStatus.ACTIVE)
                    
            return True
            
    def get_nodes(self, node_type: Optional[NodeType] = None, 
                 status: Optional[NodeStatus] = None,
                 tags: Optional[Set[str]] = None) -> List[NodeInfo]:
        """
        Get nodes matching criteria.
        
        Args:
            node_type: Filter by node type
            status: Filter by status
            tags: Filter by tags (node must have all tags)
            
        Returns:
            List of matching nodes
        """
        with self._lock:
            nodes = list(self.nodes.values())
            
            if node_type:
                nodes = [n for n in nodes if n.node_type == node_type]
                
            if status:
                nodes = [n for n in nodes if n.status == status]
                
            if tags:
                nodes = [n for n in nodes if tags.issubset(n.tags)]
                
            return nodes
            
    def get_available_nodes(self, min_cpu_cores: int = 0,
                          min_memory_gb: float = 0.0,
                          max_cpu_usage: float = 80.0,
                          max_memory_usage: float = 80.0) -> List[NodeInfo]:
        """
        Get nodes available for new tasks.
        
        Args:
            min_cpu_cores: Minimum CPU cores required
            min_memory_gb: Minimum memory required
            max_cpu_usage: Maximum acceptable CPU usage
            max_memory_usage: Maximum acceptable memory usage
            
        Returns:
            List of available nodes
        """
        available = []
        
        with self._lock:
            for node in self.nodes.values():
                if (node.status in [NodeStatus.ACTIVE, NodeStatus.IDLE] and
                    node.capacity.cpu_cores >= min_cpu_cores and
                    node.capacity.memory_gb >= min_memory_gb and
                    node.usage.cpu_percent <= max_cpu_usage and
                    node.usage.memory_percent <= max_memory_usage):
                    available.append(node)
                    
        # Sort by resource availability (lowest usage first)
        available.sort(key=lambda n: n.usage.cpu_percent + n.usage.memory_percent)
        return available
        
    def get_cluster_stats(self) -> Dict[str, Any]:
        """Get cluster statistics."""
        with self._lock:
            node_counts = {}
            for status in NodeStatus:
                node_counts[status.value] = len([
                    n for n in self.nodes.values() if n.status == status
                ])
                
            total_capacity = ResourceCapacity()
            total_usage = ResourceUsage()
            
            for node in self.nodes.values():
                total_capacity.cpu_cores += node.capacity.cpu_cores
                total_capacity.memory_gb += node.capacity.memory_gb
                total_capacity.disk_gb += node.capacity.disk_gb
                total_capacity.gpu_count += node.capacity.gpu_count
                
                total_usage.cpu_percent += node.usage.cpu_percent
                total_usage.memory_percent += node.usage.memory_percent
                total_usage.active_tasks += node.usage.active_tasks
                
            node_count = len(self.nodes)
            if node_count > 0:
                total_usage.cpu_percent /= node_count
                total_usage.memory_percent /= node_count
                
            return {
                'cluster_name': self.config.cluster_name,
                'coordinator_node': self._local_node.node_id,
                'total_nodes': node_count,
                'node_counts': node_counts,
                'total_capacity': total_capacity.to_dict(),
                'average_usage': total_usage.to_dict(),
                'stats': self._stats.copy()
            }
            
    def _heartbeat_monitor(self):
        """Background thread to monitor node heartbeats."""
        while self._running:
            try:
                current_time = time.time()
                failed_nodes = []
                
                with self._lock:
                    for node_id, node_info in self.nodes.items():
                        time_since_heartbeat = current_time - node_info.last_heartbeat
                        
                        if time_since_heartbeat > self.config.heartbeat_timeout:
                            if node_info.status != NodeStatus.FAILED:
                                failed_nodes.append(node_id)
                                self._stats['heartbeats_missed'] += 1
                                
                # Handle failed nodes outside the lock
                for node_id in failed_nodes:
                    self.update_node_status(node_id, NodeStatus.FAILED)
                    
                time.sleep(self.config.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Heartbeat monitor error: {e}")
                time.sleep(5.0)
                
    def _discovery_service(self):
        """Background service for node discovery."""
        # This would implement UDP-based node discovery
        # For now, it's a placeholder
        while self._running:
            try:
                # Node discovery logic would go here
                time.sleep(30.0)
            except Exception as e:
                logger.error(f"Discovery service error: {e}")
                time.sleep(5.0)
                
    def _handle_node_failure(self, node_info: NodeInfo):
        """Handle node failure."""
        self._stats['nodes_failed'] += 1
        
        # Trigger failure callbacks
        for callback in self._node_failed_callbacks:
            try:
                callback(node_info)
            except Exception as e:
                logger.warning(f"Node failed callback error: {e}")
                
        logger.warning(f"Node failed: {node_info.node_id}")
        
    def add_node_joined_callback(self, callback: Callable[[NodeInfo], None]):
        """Add callback for node joined events."""
        self._node_joined_callbacks.append(callback)
        
    def add_node_left_callback(self, callback: Callable[[NodeInfo], None]):
        """Add callback for node left events."""
        self._node_left_callbacks.append(callback)
        
    def add_node_failed_callback(self, callback: Callable[[NodeInfo], None]):
        """Add callback for node failed events."""
        self._node_failed_callbacks.append(callback)
        
    def create_node_group(self, group_name: str, node_ids: List[str]) -> bool:
        """
        Create a named group of nodes.
        
        Args:
            group_name: Name of the group
            node_ids: List of node IDs to include
            
        Returns:
            True if group created successfully
        """
        with self._lock:
            # Validate all nodes exist
            for node_id in node_ids:
                if node_id not in self.nodes:
                    logger.warning(f"Node {node_id} not found for group {group_name}")
                    return False
                    
            self.node_groups[group_name] = set(node_ids)
            logger.info(f"Created node group '{group_name}' with {len(node_ids)} nodes")
            return True
            
    def get_node_group(self, group_name: str) -> List[NodeInfo]:
        """Get nodes in a named group."""
        with self._lock:
            if group_name not in self.node_groups:
                return []
                
            return [self.nodes[node_id] for node_id in self.node_groups[group_name]
                   if node_id in self.nodes]