"""
Service-Specific Repositories (Simplified)
=========================================

Contains only the working repositories for models that exist.
"""

import logging
from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime, timedelta
from sqlalchemy import select, and_, or_, func, desc, delete
from sqlalchemy.orm import selectinload, joinedload

from .repositories import (
    BaseRepository, 
    SearchableRepository, 
    TimestampedRepository, 
    UserOwnedRepository,
    VersionedRepository
)

# Create logger for this module
logger = logging.getLogger(__name__)
from .models import (
    User, Concept, ConceptRelation, Knowledge, SystemMetrics, 
    Ontology, KnowledgeGraph
)
from .connection import DatabaseManager


class UserRepository(TimestampedRepository[User, UUID]):
    """Repository for User management"""
    
    def __init__(self, db_manager: DatabaseManager):
        super().__init__(db_manager, User)
    
    async def get_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        return await self.find_one_by(username=username)
    
    async def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        return await self.find_one_by(email=email)
    
    async def get_active_users(self) -> List[User]:
        """Get all active users"""
        return await self.find_by(is_active=True)
    
    async def authenticate_user(self, username: str, password_hash: str) -> Optional[User]:
        """Authenticate user credentials"""
        user = await self.get_by_username(username)
        if user and user.password_hash == password_hash and user.is_active:
            return user
        return None


class KnowledgeRepository(SearchableRepository[Knowledge, UUID], UserOwnedRepository[Knowledge, UUID]):
    """Repository for Knowledge management"""
    
    def __init__(self, db_manager: DatabaseManager):
        super().__init__(db_manager, Knowledge)
    
    async def search_by_content(self, query: str, limit: int = 10) -> List[Knowledge]:
        """Full-text search in knowledge content"""
        async with await self.get_session() as session:
            stmt = (
                select(self.model_class)
                .where(self.model_class.content.match(query))
                .order_by(desc(self.model_class.updated_at))
                .limit(limit)
            )
            result = await session.execute(stmt)
            return result.scalars().all()
    
    async def get_by_category(self, category: str) -> List[Knowledge]:
        """Get knowledge by category"""
        return await self.find_by(category=category)
    
    async def get_by_tags(self, tags: List[str]) -> List[Knowledge]:
        """Get knowledge by tags"""
        async with await self.get_session() as session:
            stmt = (
                select(self.model_class)
                .where(self.model_class.tags.overlap(tags))
                .order_by(desc(self.model_class.updated_at))
            )
            result = await session.execute(stmt)
            return result.scalars().all()
    
    async def get_public_knowledge(self) -> List[Knowledge]:
        """Get all public knowledge"""
        return await self.find_by(is_public=True)


class SystemMetricsRepository(TimestampedRepository[SystemMetrics, UUID]):
    """Repository for SystemMetrics management"""
    
    def __init__(self, db_manager: DatabaseManager):
        super().__init__(db_manager, SystemMetrics)
    
    async def get_by_service(self, service_name: str, hours: int = 24) -> List[SystemMetrics]:
        """Get metrics for a specific service within time range"""
        since = datetime.utcnow() - timedelta(hours=hours)
        async with await self.get_session() as session:
            stmt = (
                select(self.model_class)
                .where(
                    and_(
                        self.model_class.service_name == service_name,
                        self.model_class.timestamp >= since
                    )
                )
                .order_by(desc(self.model_class.timestamp))
            )
            result = await session.execute(stmt)
            return result.scalars().all()
    
    async def get_by_metric_type(self, metric_type: str, hours: int = 24) -> List[SystemMetrics]:
        """Get metrics by type within time range"""
        since = datetime.utcnow() - timedelta(hours=hours)
        async with await self.get_session() as session:
            stmt = (
                select(self.model_class)
                .where(
                    and_(
                        self.model_class.metric_type == metric_type,
                        self.model_class.timestamp >= since
                    )
                )
                .order_by(desc(self.model_class.timestamp))
            )
            result = await session.execute(stmt)
            return result.scalars().all()
    
    async def get_aggregated_metrics(
        self, 
        service_name: str, 
        metric_type: str,
        hours: int = 24
    ) -> Dict[str, Any]:
        """Get aggregated metrics for service and type"""
        since = datetime.utcnow() - timedelta(hours=hours)
        async with await self.get_session() as session:
            stmt = (
                select(
                    func.avg(self.model_class.value).label('avg_value'),
                    func.min(self.model_class.value).label('min_value'),
                    func.max(self.model_class.value).label('max_value'),
                    func.count(self.model_class.id).label('count')
                )
                .where(
                    and_(
                        self.model_class.service_name == service_name,
                        self.model_class.metric_type == metric_type,
                        self.model_class.timestamp >= since
                    )
                )
            )
            result = await session.execute(stmt)
            row = result.first()
            return {
                'avg_value': float(row.avg_value) if row.avg_value else 0,
                'min_value': float(row.min_value) if row.min_value else 0,
                'max_value': float(row.max_value) if row.max_value else 0,
                'count': row.count
            }


class OntologyRepository(TimestampedRepository[Ontology, UUID]):
    """Repository for Ontology management"""
    
    def __init__(self, db_manager: DatabaseManager):
        super().__init__(db_manager, Ontology)
    
    async def get_by_name(self, name: str) -> Optional[Ontology]:
        """Get ontology by name"""
        return await self.find_one_by(name=name)
    
    async def get_by_type(self, ontology_type: str) -> List[Ontology]:
        """Get ontologies by type"""
        return await self.find_by(ontology_type=ontology_type)
    
    async def get_active_ontologies(self) -> List[Ontology]:
        """Get all active ontologies"""
        return await self.find_by(status='active')


class KnowledgeGraphRepository(TimestampedRepository[KnowledgeGraph, UUID]):
    """Repository for KnowledgeGraph management"""
    
    def __init__(self, db_manager: DatabaseManager):
        super().__init__(db_manager, KnowledgeGraph)
    
    async def get_by_name(self, name: str) -> Optional[KnowledgeGraph]:
        """Get knowledge graph by name"""
        return await self.find_one_by(name=name)
    
    async def get_by_ontology(self, ontology_id: UUID) -> List[KnowledgeGraph]:
        """Get knowledge graphs by ontology"""
        return await self.find_by(ontology_id=ontology_id)
    
    async def get_by_type(self, graph_type: str) -> List[KnowledgeGraph]:
        """Get knowledge graphs by type"""
        return await self.find_by(graph_type=graph_type)
    
    async def get_active_graphs(self) -> List[KnowledgeGraph]:
        """Get all active knowledge graphs"""
        return await self.find_by(status='active')