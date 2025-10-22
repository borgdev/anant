"""
Schema.org Hierarchical Knowledge Graph Test
==========================================

This test demonstrates the capabilities of anant's HierarchicalKnowledgeGraph
using schema.org entities and relationships.

Features:
- Load schema.org entities from CSV files  
- Create hierarchical levels (Types -> Instances -> Locations)
- Build semantic relationships between entities
- Test hierarchical navigation and cross-level relationships
- Demonstrate advanced querying capabilities
"""

import sys
import os
import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add anant to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from anant.kg.hierarchical import HierarchicalKnowledgeGraph

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SchemaOrgHierarchicalTest:
    """Test class for schema.org hierarchical knowledge graph"""
    
    def __init__(self):
        self.hkg = None
        self.data_dir = Path(__file__).parent / "data_schemaorg"
        self.entities = {
            'persons': [],
            'organizations': [],
            'products': [],
            'places': []
        }
        self.purchase_data = []  # Store purchase relationships
        
    def load_test_data(self):
        """Load all test data from CSV files"""
        logger.info("🔄 Loading test data from CSV files...")
        
        # Load persons
        persons_file = self.data_dir / "persons.csv"
        with open(persons_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            self.entities['persons'] = list(reader)
        
        # Load organizations
        orgs_file = self.data_dir / "organizations.csv"
        with open(orgs_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            self.entities['organizations'] = list(reader)
        
        # Load products
        products_file = self.data_dir / "products.csv"
        with open(products_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            self.entities['products'] = list(reader)
        
        # Load places
        places_file = self.data_dir / "places.csv"
        with open(places_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            self.entities['places'] = list(reader)
        
        # Load person-product purchases
        purchases_file = self.data_dir / "person_product_purchases.csv"
        try:
            with open(purchases_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                self.purchase_data = list(reader)
            logger.info(f"📦 Loaded {len(self.purchase_data)} purchase relationships")
        except FileNotFoundError:
            logger.warning(f"⚠️  Purchase data file not found: {purchases_file}")
            self.purchase_data = []
        
        total_entities = sum(len(entities) for entities in self.entities.values())
        logger.info(f"📊 Loaded {total_entities} entities:")
        for entity_type, entities in self.entities.items():
            logger.info(f"   • {entity_type}: {len(entities)}")
        
        if self.purchase_data:
            logger.info(f"   • purchases: {len(self.purchase_data)}")
    
    def create_hierarchical_structure(self):
        """Create the hierarchical knowledge graph structure"""
        logger.info("🔄 Creating hierarchical knowledge graph...")
        
        # Initialize hierarchical knowledge graph
        self.hkg = HierarchicalKnowledgeGraph(
            name="schemaorg_test",
            enable_semantic_reasoning=True,
            enable_temporal_tracking=False
        )
        
        # Create hierarchical levels
        logger.info("🔄 Creating hierarchical levels...")
        
        # Level 0: Schema.org Types (most abstract)
        self.hkg.create_level("types", "Schema.org Types", 
                            "Top-level schema.org entity types", level_order=0)
        
        # Level 1: Entity Instances (concrete instances)
        self.hkg.create_level("instances", "Entity Instances", 
                            "Specific instances of schema.org types", level_order=1)
        
        # Level 2: Transactions (purchase relationships)
        self.hkg.create_level("transactions", "Transaction Context", 
                            "Purchase transactions and commercial relationships", level_order=2)
        
        # Level 3: Location Context (spatial relationships)
        self.hkg.create_level("locations", "Location Context", 
                            "Physical locations and spatial relationships", level_order=3)
        
        logger.info("✅ Created 4 hierarchical levels")
    
    def load_entities_to_hierarchy(self):
        """Load entities into the hierarchical structure"""
        logger.info("🔄 Loading entities into hierarchical structure...")
        
        nodes_added = 0
        relationships_added = 0
        
        # Add schema.org type entities to types level
        schema_types = ["Person", "Organization", "Product", "Place", "Purchase"]
        for schema_type in schema_types:
            type_id = f"type_{schema_type.lower()}"
            properties = {
                'name': schema_type,
                'type': 'SchemaOrgType',
                'description': f"Schema.org {schema_type} type",
                'schema_url': f"https://schema.org/{schema_type}"
            }
            
            success = self.hkg.add_node_to_level(
                node_id=type_id,
                node_type='SchemaOrgType',
                properties=properties,
                level_id="types"
            )
            if success:
                nodes_added += 1
        
        # Add all entity instances to instances level
        for entity_category, entities in self.entities.items():
            for entity in entities:
                entity_id = entity['entity_id']
                entity_type = entity['entity_type']
                
                # Parse properties
                try:
                    additional_props = json.loads(entity['properties'])
                except json.JSONDecodeError:
                    additional_props = {}
                
                # Combine all properties
                properties = {
                    'name': entity['name'],
                    'type': entity_type,
                    'email': entity.get('email', ''),
                    'telephone': entity.get('telephone', ''),
                    'description': entity['description'],
                    **additional_props
                }
                
                # Add to instances level
                success = self.hkg.add_node_to_level(
                    node_id=entity_id,
                    node_type=entity_type,
                    properties=properties,
                    level_id="instances"
                )
                
                if success:
                    nodes_added += 1
                    
                    # Add cross-level relationship to type
                    type_id = f"type_{entity_type.lower()}"
                    self.hkg.add_cross_level_relationship(
                        source_node=entity_id,
                        target_node=type_id,
                        relationship_type='instanceOf',
                        properties={'created_at': datetime.now().isoformat()}
                    )
                    relationships_added += 1
        
        # Add purchase transactions to transactions level
        for purchase in self.purchase_data:
            purchase_id = purchase['purchase_id']
            
            # Create purchase transaction entity
            transaction_properties = {
                'name': f"Purchase Transaction {purchase_id}",
                'type': 'PurchaseTransaction',
                'transaction_id': purchase_id,
                'person_id': purchase['person_id'],
                'product_id': purchase['product_id'],
                'purchase_date': purchase['purchase_date'],
                'quantity': int(purchase['quantity']),
                'unit_price': float(purchase['unit_price']),
                'total_amount': float(purchase['total_amount']),
                'payment_method': purchase['payment_method'],
                'status': purchase['purchase_status'],
                'notes': purchase['notes']
            }
            
            success = self.hkg.add_node_to_level(
                node_id=purchase_id,
                node_type='PurchaseTransaction',
                properties=transaction_properties,
                level_id="transactions"
            )
            
            if success:
                nodes_added += 1
                
                # Add cross-level relationship to Purchase type
                self.hkg.add_cross_level_relationship(
                    source_node=purchase_id,
                    target_node="type_purchase",
                    relationship_type='instanceOf',
                    properties={'created_at': datetime.now().isoformat()}
                )
                relationships_added += 1
        
        # Add places to locations level (they can exist in both instances and locations)
        for place in self.entities['places']:
            place_id = place['entity_id']
            
            # Parse location properties
            try:
                location_props = json.loads(place['properties'])
            except json.JSONDecodeError:
                location_props = {}
            
            # Create location context entity
            location_context_id = f"location_{place_id}"
            location_properties = {
                'name': f"Location context for {place['name']}",
                'type': 'LocationContext',
                'place_id': place_id,
                'coordinates': location_props.get('coordinates', ''),
                'locality': location_props.get('addressLocality', ''),
                'region': location_props.get('addressRegion', ''),
                'country': location_props.get('addressCountry', ''),
                'postal_code': location_props.get('postalCode', ''),
                'place_type': location_props.get('placeType', 'Unknown')
            }
            
            success = self.hkg.add_node_to_level(
                node_id=location_context_id,
                node_type='LocationContext',
                properties=location_properties,
                level_id="locations"
            )
            
            if success:
                nodes_added += 1
                
                # Link location context to place instance
                self.hkg.add_cross_level_relationship(
                    source_node=location_context_id,
                    target_node=place_id,
                    relationship_type='representsLocation',
                    properties={'created_at': datetime.now().isoformat()}
                )
                relationships_added += 1
        
        logger.info(f"✅ Added {nodes_added} nodes and {relationships_added} cross-level relationships")
        return nodes_added, relationships_added
    
    def create_semantic_relationships(self):
        """Create semantic relationships between entities"""
        logger.info("🔄 Creating semantic relationships...")
        
        relationships_added = 0
        
        # Create relationships based on entity properties
        for person in self.entities['persons']:
            person_id = person['entity_id']
            
            try:
                person_props = json.loads(person['properties'])
                
                # Person works for Organization
                if 'worksFor' in person_props:
                    org_id = person_props['worksFor']
                    self.hkg.add_relationship(
                        source_node=person_id,
                        target_node=org_id,
                        relationship_type='worksFor',
                        properties={'role': person_props.get('jobTitle', 'Employee')}
                    )
                    relationships_added += 1
                
                # Person has address (lives at Place)
                if 'address' in person_props:
                    place_id = person_props['address']
                    self.hkg.add_relationship(
                        source_node=person_id,
                        target_node=place_id,
                        relationship_type='livesAt',
                        properties={'relationship_type': 'residential'}
                    )
                    relationships_added += 1
                    
            except json.JSONDecodeError:
                continue
        
        # Organization relationships
        for organization in self.entities['organizations']:
            org_id = organization['entity_id']
            
            try:
                org_props = json.loads(organization['properties'])
                
                # Organization has address (located at Place)
                if 'address' in org_props:
                    place_id = org_props['address']
                    self.hkg.add_relationship(
                        source_node=org_id,
                        target_node=place_id,
                        relationship_type='locatedAt',
                        properties={'relationship_type': 'business_location'}
                    )
                    relationships_added += 1
                    
            except json.JSONDecodeError:
                continue
        
        # Product relationships
        for product in self.entities['products']:
            product_id = product['entity_id']
            
            try:
                product_props = json.loads(product['properties'])
                
                # Product manufactured by Organization
                if 'manufacturer' in product_props:
                    org_id = product_props['manufacturer']
                    self.hkg.add_relationship(
                        source_node=product_id,
                        target_node=org_id,
                        relationship_type='manufacturedBy',
                        properties={
                            'brand': product_props.get('brand', ''),
                            'model': product_props.get('model', ''),
                            'version': product_props.get('version', '')
                        }
                    )
                    relationships_added += 1
                    
            except json.JSONDecodeError:
                continue
        
        # Create purchase-based relationships
        for purchase in self.purchase_data:
            try:
                purchase_id = purchase['purchase_id']
                person_id = purchase['person_id']
                product_id = purchase['product_id']
                
                # Person purchased Product (via transaction)
                self.hkg.add_relationship(
                    source_node=person_id,
                    target_node=purchase_id,
                    relationship_type='initiatedPurchase',
                    properties={
                        'purchase_date': purchase['purchase_date'],
                        'payment_method': purchase['payment_method'],
                        'total_amount': purchase['total_amount']
                    }
                )
                relationships_added += 1
                
                # Transaction involves Product
                self.hkg.add_relationship(
                    source_node=purchase_id,
                    target_node=product_id,
                    relationship_type='involvesPurchaseOf',
                    properties={
                        'quantity': purchase['quantity'],
                        'unit_price': purchase['unit_price'],
                        'status': purchase['purchase_status']
                    }
                )
                relationships_added += 1
                
                # Direct Person-Product relationship (for easier querying)
                self.hkg.add_relationship(
                    source_node=person_id,
                    target_node=product_id,
                    relationship_type='purchased',
                    properties={
                        'transaction_id': purchase_id,
                        'purchase_date': purchase['purchase_date'],
                        'total_amount': purchase['total_amount'],
                        'quantity': purchase['quantity']
                    }
                )
                relationships_added += 1
                
            except Exception as e:
                logger.warning(f"⚠️  Failed to create purchase relationship for {purchase_id}: {e}")
                continue
        
        logger.info(f"✅ Created {relationships_added} semantic relationships")
        return relationships_added
    
    def test_hierarchical_operations(self):
        """Test hierarchical navigation and queries"""
        logger.info("🔄 Testing hierarchical operations...")
        
        test_results = []
        
        # Test 1: Level statistics
        hierarchy_stats = self.hkg.get_hierarchy_statistics()
        test_results.append(f"Hierarchy levels: {len(hierarchy_stats.get('levels', {}))}")
        
        # Test 2: Entities at each level
        for level_id in ["types", "instances", "transactions", "locations"]:
            entities = self.hkg.get_nodes_at_level(level_id)
            test_results.append(f"Level '{level_id}': {len(entities)} entities")
        
        # Test 3: Cross-level relationships
        cross_level_rels = self.hkg.get_cross_level_relationships()
        test_results.append(f"Cross-level relationships: {len(cross_level_rels)}")
        
        # Test 4: Navigation tests (using sample entities)
        instance_entities = self.hkg.get_nodes_at_level("instances")
        if instance_entities:
            sample_node = instance_entities[0]
            ancestors = self.hkg.get_ancestors(sample_node)
            descendants = self.hkg.get_descendants(sample_node)
            test_results.append(f"Node '{sample_node}': {len(ancestors)} ancestors, {len(descendants)} descendants")
        
        # Test 5: Entity type filtering
        person_entities = [e for e in instance_entities if e.startswith('person_')]
        org_entities = [e for e in instance_entities if e.startswith('org_')]
        product_entities = [e for e in instance_entities if e.startswith('product_')]
        place_entities = [e for e in instance_entities if e.startswith('place_')]
        
        # Transaction entities
        transaction_entities = self.hkg.get_nodes_at_level("transactions")
        purchase_entities = [e for e in transaction_entities if e.startswith('purchase_')]
        
        test_results.append(f"Person nodes: {len(person_entities)}")
        test_results.append(f"Organization nodes: {len(org_entities)}")
        test_results.append(f"Product nodes: {len(product_entities)}")
        test_results.append(f"Place nodes: {len(place_entities)}")
        test_results.append(f"Purchase transaction nodes: {len(purchase_entities)}")
        
        # Test 6: Overall graph statistics
        total_nodes = self.hkg.num_nodes()
        total_edges = self.hkg.num_edges()
        test_results.append(f"Total graph: {total_nodes} nodes, {total_edges} edges")
        
        # Test 7: Semantic search (if available)
        try:
            search_results = self.hkg.semantic_search("TechCorp")
            test_results.append(f"Semantic search 'TechCorp': {len(search_results)} results")
        except Exception as e:
            test_results.append(f"Semantic search error: {str(e)[:50]}...")
        
        # Display results
        logger.info("📊 Hierarchical operation test results:")
        for i, result in enumerate(test_results, 1):
            logger.info(f"   {i}. {result}")
        
        return test_results
    
    def demonstrate_relationships(self):
        """Demonstrate various relationship types and queries"""
        logger.info("🔄 Demonstrating relationship queries...")
        
        # Find all work relationships
        work_relationships = self.hkg.get_cross_level_relationships(relationship_type='worksFor')
        logger.info(f"📊 Found {len(work_relationships)} work relationships")
        
        # Find location relationships
        location_relationships = self.hkg.get_cross_level_relationships(relationship_type='locatedAt')
        logger.info(f"📊 Found {len(location_relationships)} location relationships")
        
        # Find manufacturing relationships
        manufacturing_relationships = self.hkg.get_cross_level_relationships(relationship_type='manufacturedBy')
        logger.info(f"📊 Found {len(manufacturing_relationships)} manufacturing relationships")
        
        # Find purchase relationships
        purchase_relationships = self.hkg.get_cross_level_relationships(relationship_type='purchased')
        logger.info(f"📊 Found {len(purchase_relationships)} purchase relationships")
        
        # Find transaction relationships
        transaction_init_relationships = self.hkg.get_cross_level_relationships(relationship_type='initiatedPurchase')
        logger.info(f"📊 Found {len(transaction_init_relationships)} transaction initiation relationships")
        
        # Sample detailed relationship analysis
        if work_relationships:
            sample_work_rel = work_relationships[0]
            logger.info(f"📋 Sample work relationship: {sample_work_rel['source_node']} -> {sample_work_rel['target_node']}")
        
        if purchase_relationships:
            sample_purchase_rel = purchase_relationships[0]
            logger.info(f"📋 Sample purchase relationship: {sample_purchase_rel['source_node']} -> {sample_purchase_rel['target_node']}")
            
        # Purchase analytics
        if self.purchase_data:
            total_revenue = sum(float(p['total_amount']) for p in self.purchase_data)
            avg_order_value = total_revenue / len(self.purchase_data)
            logger.info(f"📊 Purchase Analytics:")
            logger.info(f"   • Total Revenue: ${total_revenue:.2f}")
            logger.info(f"   • Average Order Value: ${avg_order_value:.2f}")
            logger.info(f"   • Total Transactions: {len(self.purchase_data)}")
            
            # Most popular products
            product_counts = {}
            for purchase in self.purchase_data:
                product_id = purchase['product_id']
                product_counts[product_id] = product_counts.get(product_id, 0) + 1
            
            if product_counts:
                most_popular = max(product_counts, key=product_counts.get)
                logger.info(f"   • Most Popular Product: {most_popular} ({product_counts[most_popular]} purchases)")
    
    def export_graph_summary(self):
        """Export a summary of the graph structure"""
        logger.info("🔄 Exporting graph summary...")
        
        summary = {
            'graph_name': self.hkg.name,
            'creation_time': datetime.now().isoformat(),
            'levels': {},
            'statistics': {
                'total_nodes': self.hkg.num_nodes(),
                'total_edges': self.hkg.num_edges(),
                'cross_level_relationships': len(self.hkg.get_cross_level_relationships())
            }
        }
        
        # Level details
        for level_id in ["types", "instances", "transactions", "locations"]:
            entities = self.hkg.get_nodes_at_level(level_id)
            level_metadata = self.hkg.get_level_metadata(level_id)
            
            summary['levels'][level_id] = {
                'node_count': len(entities),
                'metadata': level_metadata,
                'sample_nodes': entities[:5] if entities else []
            }
        
        # Save summary
        summary_path = self.data_dir / "graph_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Graph summary exported to {summary_path}")
        return summary
    
    def run_complete_test(self):
        """Run the complete hierarchical knowledge graph test"""
        logger.info("🚀 Starting Schema.org Hierarchical Knowledge Graph Test")
        logger.info("="*80)
        
        try:
            # Step 1: Load test data
            self.load_test_data()
            
            # Step 2: Create hierarchical structure
            self.create_hierarchical_structure()
            
            # Step 3: Load entities into hierarchy
            nodes_count, cross_level_rels = self.load_entities_to_hierarchy()
            
            # Step 4: Create semantic relationships
            semantic_rels = self.create_semantic_relationships()
            
            # Step 5: Test hierarchical operations
            test_results = self.test_hierarchical_operations()
            
            # Step 6: Demonstrate relationships
            self.demonstrate_relationships()
            
            # Step 7: Export summary
            summary = self.export_graph_summary()
            
            # Final summary
            logger.info("="*80)
            logger.info("🎉 HIERARCHICAL KNOWLEDGE GRAPH TEST COMPLETE!")
            logger.info("="*80)
            logger.info(f"📊 Total Nodes Loaded: {nodes_count}")
            logger.info(f"📊 Cross-level Relationships: {cross_level_rels}")
            logger.info(f"📊 Semantic Relationships: {semantic_rels}")
            logger.info(f"📊 Final Graph Size: {self.hkg.num_nodes()} nodes, {self.hkg.num_edges()} edges")
            logger.info(f"📊 Hierarchical Levels: {len(summary['levels'])}")
            logger.info("📊 Test Results Summary:")
            for i, result in enumerate(test_results[:5], 1):  # Show first 5 results
                logger.info(f"   • {result}")
            logger.info("="*80)
            logger.info("✅ Schema.org hierarchical knowledge graph successfully created and tested!")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main function to run the test"""
    test = SchemaOrgHierarchicalTest()
    success = test.run_complete_test()
    
    if success:
        logger.info("🎯 All tests passed successfully!")
        return 0
    else:
        logger.error("💥 Some tests failed!")
        return 1


if __name__ == "__main__":
    exit(main())