"""
Schema.org Hierarchical Knowledge Graph Test
==========================================

This script demonstrates the integration between anant library and anant_integration
to create a hierarchical knowledge graph from schema.org ontology.

Features:
- Load schema.org ontology from complete.jsonld
- Create 'schemaorg_meta' metagraph with anant library
- Generate realistic test data based on schema.org classes
- Store data in anant metagraph and metadata in PostgreSQL
- Test hierarchical queries and relationships
"""

import asyncio
import json
import csv
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
import uuid
import random

# anant library imports
from anant.kg.hierarchical import HierarchicalKnowledgeGraph

# anant_integration imports
import sys
sys.path.append('/home/amansingh/dev/ai/anant/anant_integration')

from database.connection import DatabaseManager
from database.models import KnowledgeGraph, Ontology, Concept, ConceptRelation
from database.service_repositories import KnowledgeGraphRepository, OntologyRepository

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SchemaOrgKnowledgeGraphTest:
    """Test class for schema.org hierarchical knowledge graph integration"""
    
    def __init__(self):
        self.db_manager = None
        self.hierarchical_kg = None
        self.ontology_data = None
        self.test_data_path = "/home/amansingh/dev/ai/anant/schemaorg_test_data.csv"
        self.classes = {}
        self.properties = {}
        self.hierarchy = {}
        
    async def initialize(self):
        """Initialize database connection and create metagraph"""
        logger.info("🔄 Initializing database connection...")
        self.db_manager = DatabaseManager()
        await self.db_manager.initialize()
        
        logger.info("🔄 Creating hierarchical knowledge graph with anant...")
        self.hierarchical_kg = HierarchicalKnowledgeGraph(
            name="schemaorg_meta", 
            enable_semantic_reasoning=True,
            enable_temporal_tracking=False
        )
        
        logger.info("✅ Initialization complete")
    
    def load_schema_org_ontology(self):
        """Load and analyze schema.org ontology structure"""
        logger.info("🔄 Loading schema.org ontology...")
        
        ontology_path = "/home/amansingh/dev/ai/anant/schamaorg/complete.jsonld"
        with open(ontology_path, 'r', encoding='utf-8') as f:
            self.ontology_data = json.load(f)
        
        # Extract classes and properties
        graph = self.ontology_data.get('@graph', [])
        
        for item in graph:
            item_type = item.get('@type')
            item_id = item.get('@id', '')
            
            if item_type == 'rdfs:Class' and item_id.startswith('schema:'):
                class_name = item_id.replace('schema:', '')
                self.classes[class_name] = {
                    'id': item_id,
                    'label': item.get('rdfs:label', class_name),
                    'comment': item.get('rdfs:comment', ''),
                    'subClassOf': item.get('rdfs:subClassOf', {}),
                    'supersededBy': item.get('schema:supersededBy', {}),
                }
                
            elif item_type == 'rdf:Property' and item_id.startswith('schema:'):
                prop_name = item_id.replace('schema:', '')
                self.properties[prop_name] = {
                    'id': item_id,
                    'label': item.get('rdfs:label', prop_name),
                    'comment': item.get('rdfs:comment', ''),
                    'domainIncludes': item.get('schema:domainIncludes', []),
                    'rangeIncludes': item.get('schema:rangeIncludes', []),
                    'subPropertyOf': item.get('rdfs:subPropertyOf', {}),
                }
        
        logger.info(f"📊 Found {len(self.classes)} classes and {len(self.properties)} properties")
        self._build_hierarchy()
        
    def _build_hierarchy(self):
        """Build class hierarchy from subClassOf relationships"""
        logger.info("🔄 Building class hierarchy...")
        
        for class_name, class_data in self.classes.items():
            subclass_of = class_data.get('subClassOf')
            if subclass_of:
                if isinstance(subclass_of, dict):
                    parent_id = subclass_of.get('@id', '')
                    if parent_id.startswith('schema:'):
                        parent_class = parent_id.replace('schema:', '')
                        if parent_class not in self.hierarchy:
                            self.hierarchy[parent_class] = []
                        self.hierarchy[parent_class].append(class_name)
                elif isinstance(subclass_of, list):
                    for parent in subclass_of:
                        if isinstance(parent, dict):
                            parent_id = parent.get('@id', '')
                            if parent_id.startswith('schema:'):
                                parent_class = parent_id.replace('schema:', '')
                                if parent_class not in self.hierarchy:
                                    self.hierarchy[parent_class] = []
                                self.hierarchy[parent_class].append(class_name)
        
        logger.info(f"🏗️ Built hierarchy with {len(self.hierarchy)} parent classes")
        
    def generate_test_data(self):
        """Generate realistic test data based on schema.org classes"""
        logger.info("🔄 Generating test data...")
        
        # Focus on common schema.org classes for realistic data
        test_entities = []
        
        # Generate Person entities
        for i in range(50):
            entity = {
                'entity_id': f"person_{i}",
                'entity_type': 'Person',
                'name': f"Person {i}",
                'email': f"person{i}@example.com",
                'telephone': f"+1-555-{1000+i}",
                'description': f"Test person entity {i}",
                'created_at': datetime.now().isoformat(),
                'properties': json.dumps({
                    'givenName': f"Given{i}",
                    'familyName': f"Family{i}",
                    'jobTitle': random.choice(['Engineer', 'Manager', 'Analyst', 'Designer']),
                    'worksFor': f"organization_{i//10}"
                })
            }
            test_entities.append(entity)
        
        # Generate Organization entities
        for i in range(10):
            entity = {
                'entity_id': f"organization_{i}",
                'entity_type': 'Organization',
                'name': f"Company {i}",
                'email': f"contact@company{i}.com",
                'telephone': f"+1-555-{2000+i}",
                'description': f"Test organization entity {i}",
                'created_at': datetime.now().isoformat(),
                'properties': json.dumps({
                    'legalName': f"Company {i} Inc.",
                    'foundingDate': f"201{i%10}-01-01",
                    'numberOfEmployees': random.randint(10, 1000),
                    'industry': random.choice(['Technology', 'Finance', 'Healthcare', 'Education'])
                })
            }
            test_entities.append(entity)
            
        # Generate Product entities
        for i in range(30):
            entity = {
                'entity_id': f"product_{i}",
                'entity_type': 'Product',
                'name': f"Product {i}",
                'email': f"support@product{i}.com",
                'telephone': f"+1-555-{3000+i}",
                'description': f"Test product entity {i}",
                'created_at': datetime.now().isoformat(),
                'properties': json.dumps({
                    'brand': f"Brand {i//10}",
                    'model': f"Model {i}",
                    'price': random.randint(100, 5000),
                    'manufacturer': f"organization_{i//5}"
                })
            }
            test_entities.append(entity)
            
        # Generate Place entities
        for i in range(20):
            entity = {
                'entity_id': f"place_{i}",
                'entity_type': 'Place',
                'name': f"Place {i}",
                'email': f"info@place{i}.com",
                'telephone': f"+1-555-{4000+i}",
                'description': f"Test place entity {i}",
                'created_at': datetime.now().isoformat(),
                'properties': json.dumps({
                    'addressLocality': f"City {i}",
                    'addressRegion': random.choice(['CA', 'NY', 'TX', 'FL']),
                    'postalCode': f"{10000+i}",
                    'addressCountry': 'US'
                })
            }
            test_entities.append(entity)
        
        # Save to CSV
        fieldnames = ['entity_id', 'entity_type', 'name', 'email', 'telephone', 'description', 'created_at', 'properties']
        
        with open(self.test_data_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(test_entities)
        
        logger.info(f"📊 Generated {len(test_entities)} test entities and saved to {self.test_data_path}")
        return test_entities
    
    async def create_ontology_in_db(self):
        """Create ontology record in PostgreSQL database"""
        logger.info("🔄 Creating ontology record in database...")
        
        ontology_repo = OntologyRepository(self.db_manager)
        
        ontology_data = {
            'name': 'schema.org',
            'version': '1.0',
            'description': 'Schema.org ontology for structured data markup',
            'ontology_type': 'UMLS',  # Using UMLS as closest match
            'source_url': 'https://schema.org/',
            'namespace': 'https://schema.org/',
            'metadata': {
                'classes_count': len(self.classes),
                'properties_count': len(self.properties),
                'hierarchy_depth': self._calculate_hierarchy_depth(),
                'loaded_at': datetime.now().isoformat()
            }
        }
        
        ontology = await ontology_repo.create(ontology_data)
        logger.info(f"✅ Created ontology record with ID: {ontology.id}")
        return ontology
    
    async def create_knowledge_graph_in_db(self, ontology_id: str):
        """Create knowledge graph record in PostgreSQL database"""
        logger.info("🔄 Creating knowledge graph record in database...")
        
        kg_repo = KnowledgeGraphRepository(self.db_manager)
        
        kg_data = {
            'name': 'schemaorg_meta',
            'description': 'Hierarchical knowledge graph based on schema.org ontology',
            'graph_type': 'HIERARCHICAL',
            'ontology_id': ontology_id,
            'metadata': {
                'hierarchical_kg_name': 'schemaorg_meta',
                'anant_version': '1.0',
                'created_with': 'anant_integration',
                'schema_org_version': 'latest'
            }
        }
        
        kg = await kg_repo.create(kg_data)
        logger.info(f"✅ Created knowledge graph record with ID: {kg.id}")
        return kg
    
    def load_data_to_hierarchical_kg(self, test_entities: List[Dict]):
        """Load test data into anant hierarchical knowledge graph"""
        logger.info("🔄 Loading data into hierarchical knowledge graph...")
        
        # First, create hierarchical levels based on schema.org structure
        logger.info("🔄 Creating hierarchical levels...")
        
        # Create top-level for schema.org types
        self.hierarchical_kg.create_level("types", "Schema.org Types", "Top-level schema.org types", 0)
        
        # Create level for instances  
        self.hierarchical_kg.create_level("instances", "Entity Instances", "Specific instances of schema.org types", 1)
        
        # Add nodes for each entity
        nodes_added = 0
        edges_added = 0
        
        for entity in test_entities:
            # Add main entity to instances level
            node_id = entity['entity_id']
            entity_properties = {
                'name': entity['name'],
                'type': entity['entity_type'],
                'email': entity.get('email', ''),
                'telephone': entity.get('telephone', ''),
                'description': entity['description']
            }
            
            # Parse additional properties
            try:
                additional_props = json.loads(entity['properties'])
                entity_properties.update(additional_props)
            except json.JSONDecodeError:
                logger.warning(f"Could not parse properties for entity {node_id}")
            
            # Add entity to hierarchical KG
            success = self.hierarchical_kg.add_entity_to_level(
                entity_id=node_id,
                entity_type=entity['entity_type'],
                properties=entity_properties,
                level_id="instances"
            )
            
            if success:
                nodes_added += 1
                
                # Add type entity to types level if not exists
                type_node_id = f"type_{entity['entity_type']}"
                if not self.hierarchical_kg.has_node(type_node_id):
                    type_properties = {
                        'name': entity['entity_type'],
                        'type': 'Type',
                        'description': f"Schema.org type: {entity['entity_type']}"
                    }
                    type_success = self.hierarchical_kg.add_entity_to_level(
                        entity_id=type_node_id,
                        entity_type='Type',
                        properties=type_properties,
                        level_id="types"
                    )
                    if type_success:
                        nodes_added += 1
                
                # Add cross-level relationship (instanceOf)
                self.hierarchical_kg.add_cross_level_relationship(
                    source_entity=node_id,
                    target_entity=type_node_id,
                    relationship_type='instanceOf'
                )
                edges_added += 1
                
                # Add same-level relationships based on properties
                try:
                    props = json.loads(entity['properties'])
                    
                    if entity['entity_type'] == 'Person':
                        if 'worksFor' in props:
                            org_id = props['worksFor']
                            self.hierarchical_kg.add_relationship(
                                source_entity=node_id,
                                target_entity=org_id, 
                                relationship_type='worksFor'
                            )
                            edges_added += 1
                            
                    elif entity['entity_type'] == 'Product':
                        if 'manufacturer' in props:
                            manufacturer_id = props['manufacturer']
                            self.hierarchical_kg.add_relationship(
                                source_entity=node_id,
                                target_entity=manufacturer_id,
                                relationship_type='manufacturedBy'
                            )
                            edges_added += 1
                            
                except json.JSONDecodeError:
                    pass
        
        logger.info(f"✅ Loaded {nodes_added} nodes and {edges_added} edges into hierarchical knowledge graph")
        return nodes_added, edges_added
    
    def _get_parent_types(self, entity_type: str) -> List[str]:
        """Get all parent types for a given entity type"""
        parents = []
        for parent, children in self.hierarchy.items():
            if entity_type in children:
                parents.append(parent)
        return parents
    
    def _calculate_hierarchy_depth(self) -> int:
        """Calculate the maximum depth of the class hierarchy"""
        def get_depth(class_name, visited=None):
            if visited is None:
                visited = set()
            if class_name in visited:
                return 0
            visited.add(class_name)
            
            if class_name not in self.hierarchy:
                return 1
            
            max_child_depth = 0
            for child in self.hierarchy[class_name]:
                child_depth = get_depth(child, visited.copy())
                max_child_depth = max(max_child_depth, child_depth)
            
            return max_child_depth + 1
        
        max_depth = 0
        for root_class in self.hierarchy.keys():
            depth = get_depth(root_class)
            max_depth = max(max_depth, depth)
        
        return max_depth
    
    def test_hierarchical_queries(self):
        """Test hierarchical queries on the knowledge graph"""
        logger.info("🔄 Testing hierarchical queries...")
        
        queries = []
        
        # Test 1: Get hierarchy statistics
        hierarchy_stats = self.hierarchical_kg.get_hierarchy_statistics()
        queries.append(f"Hierarchy levels: {len(hierarchy_stats.get('levels', {}))}")
        
        # Test 2: Find entities at specific levels
        instance_entities = self.hierarchical_kg.get_entities_at_level("instances")
        type_entities = self.hierarchical_kg.get_entities_at_level("types")
        queries.append(f"Instance entities: {len(instance_entities)}")
        queries.append(f"Type entities: {len(type_entities)}")
        
        # Test 3: Test cross-level relationships
        cross_level_rels = self.hierarchical_kg.get_cross_level_relationships()
        queries.append(f"Cross-level relationships: {len(cross_level_rels)}")
        
        # Test 4: Test navigation (pick a sample entity)
        if instance_entities:
            sample_entity = instance_entities[0]
            ancestors = self.hierarchical_kg.get_ancestors(sample_entity)
            descendants = self.hierarchical_kg.get_descendants(sample_entity)
            queries.append(f"Sample entity '{sample_entity}' has {len(ancestors)} ancestors, {len(descendants)} descendants")
        
        # Test 5: Overall graph statistics
        total_nodes = self.hierarchical_kg.num_nodes()
        total_edges = self.hierarchical_kg.num_edges()
        queries.append(f"Total graph size: {total_nodes} nodes, {total_edges} edges")
        
        # Test 6: Semantic search (if available)
        try:
            search_results = self.hierarchical_kg.semantic_search("Person")
            queries.append(f"Semantic search for 'Person': {len(search_results)} results")
        except Exception as e:
            queries.append(f"Semantic search not available: {e}")
        
        for query_result in queries:
            logger.info(f"📊 Query result: {query_result}")
        
        return queries
    
    async def run_complete_test(self):
        """Run the complete integration test"""
        logger.info("🚀 Starting complete schema.org hierarchical knowledge graph test")
        
        try:
            # Step 1: Initialize
            await self.initialize()
            
            # Step 2: Load schema.org ontology
            self.load_schema_org_ontology()
            
            # Step 3: Generate test data
            test_entities = self.generate_test_data()
            
            # Step 4: Create ontology in database
            ontology = await self.create_ontology_in_db()
            
            # Step 5: Create knowledge graph in database
            kg = await self.create_knowledge_graph_in_db(ontology.id)
            
            # Step 6: Load data into hierarchical knowledge graph
            nodes_count, edges_count = self.load_data_to_hierarchical_kg(test_entities)
            
            # Step 7: Test hierarchical queries
            query_results = self.test_hierarchical_queries()
            
            # Final summary
            logger.info("="*80)
            logger.info("🎉 INTEGRATION TEST COMPLETE!")
            logger.info("="*80)
            logger.info(f"📊 Schema.org Classes: {len(self.classes)}")
            logger.info(f"📊 Schema.org Properties: {len(self.properties)}")
            logger.info(f"📊 Hierarchy Relationships: {len(self.hierarchy)}")
            logger.info(f"📊 Test Entities Generated: {len(test_entities)}")
            logger.info(f"📊 Hierarchical KG Nodes: {nodes_count}")
            logger.info(f"📊 Hierarchical KG Edges: {edges_count}")
            logger.info(f"📊 Database Ontology ID: {ontology.id}")
            logger.info(f"📊 Database Knowledge Graph ID: {kg.id}")
            logger.info("📊 Query Results:")
            for result in query_results:
                logger.info(f"   • {result}")
            logger.info("="*80)
            logger.info("✅ Integration between anant HierarchicalKnowledgeGraph and anant_integration verified!")
            
        except Exception as e:
            logger.error(f"❌ Test failed: {e}")
            raise
        
        finally:
            if self.db_manager:
                await self.db_manager.close()


async def main():
    """Main function to run the test"""
    test = SchemaOrgKnowledgeGraphTest()
    await test.run_complete_test()


if __name__ == "__main__":
    asyncio.run(main())