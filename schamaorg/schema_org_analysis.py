#!/usr/bin/env python3
"""
Schema.org Ontology-Driven Hierarchical Knowledge Graph Analysis
================================================================

This script demonstrates how to:
1. Load the Schema.org ontology from complete.jsonld
2. Parse class definitions, properties, and relationships 
3. Create a hierarchical knowledge graph structure
4. Load data with ontological awareness
"""
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Set, Optional
sys.path.insert(0, '/home/amansingh/dev/ai/anant')

import anant
from anant.kg.hierarchical import HierarchicalKnowledgeGraph
import polars as pl

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("🚀 Schema.org Ontology-Driven Hierarchical Knowledge Graph Analysis")

# Initialize ANANT
anant.setup()

class SchemaOrgOntologyParser:
    """Parser for Schema.org ontology from JSON-LD"""
    
    def __init__(self, jsonld_path: str):
        self.jsonld_path = jsonld_path
        self.classes: Dict[str, Dict[str, Any]] = {}
        self.properties: Dict[str, Dict[str, Any]] = {}
        self.class_hierarchy: Dict[str, List[str]] = {}  # class -> list of subclasses
        
    def load_ontology(self) -> None:
        """Load and parse the Schema.org ontology"""
        logger.info(f"Loading Schema.org ontology from {self.jsonld_path}")
        
        with open(self.jsonld_path, 'r', encoding='utf-8') as f:
            ontology_data = json.load(f)
        
        graph = ontology_data.get('@graph', [])
        logger.info(f"Processing {len(graph)} ontology items")
        
        # First pass: extract classes and properties
        for item in graph:
            item_id = item.get('@id', '')
            item_type = item.get('@type', '')
            
            if item_type == 'rdfs:Class' and item_id.startswith('schema:'):
                self._process_class(item)
            elif item_type == 'rdf:Property' and item_id.startswith('schema:'):
                self._process_property(item)
        
        # Second pass: build class hierarchy
        self._build_class_hierarchy()
        
        logger.info(f"Loaded {len(self.classes)} classes and {len(self.properties)} properties")
        
    def _process_class(self, class_item: Dict[str, Any]) -> None:
        """Process a class definition"""
        class_id = class_item['@id'].replace('schema:', '')
        
        # Handle equivalentClass which can be dict or list
        equivalent_class = class_item.get('owl:equivalentClass', {})
        if isinstance(equivalent_class, list):
            equivalent_class = equivalent_class[0] if equivalent_class else {}
        equiv_id = equivalent_class.get('@id', '') if isinstance(equivalent_class, dict) else ''
        
        self.classes[class_id] = {
            'id': class_id,
            'label': class_item.get('rdfs:label', class_id),
            'comment': class_item.get('rdfs:comment', ''),
            'subclass_of': self._extract_subclass_of(class_item),
            'equivalent_class': equiv_id,
            'properties': []  # Will be populated later
        }
        
    def _process_property(self, prop_item: Dict[str, Any]) -> None:
        """Process a property definition"""
        prop_id = prop_item['@id'].replace('schema:', '')
        
        domain_includes = self._extract_domain_includes(prop_item)
        range_includes = self._extract_range_includes(prop_item)
        
        # Handle subPropertyOf which can be dict or list
        subproperty_of = prop_item.get('rdfs:subPropertyOf', {})
        if isinstance(subproperty_of, list):
            subproperty_of = subproperty_of[0] if subproperty_of else {}
        subprop_id = subproperty_of.get('@id', '') if isinstance(subproperty_of, dict) else ''
        
        self.properties[prop_id] = {
            'id': prop_id,
            'label': prop_item.get('rdfs:label', prop_id),
            'comment': prop_item.get('rdfs:comment', ''),
            'domain_includes': domain_includes,
            'range_includes': range_includes,
            'subproperty_of': subprop_id.replace('schema:', '') if subprop_id else ''
        }
        
    def _extract_subclass_of(self, class_item: Dict[str, Any]) -> str:
        """Extract subclass relationship"""
        subclass_of = class_item.get('rdfs:subClassOf', {})
        if isinstance(subclass_of, list):
            subclass_of = subclass_of[0] if subclass_of else {}
        return subclass_of.get('@id', '').replace('schema:', '') if isinstance(subclass_of, dict) else ''
        
    def _extract_domain_includes(self, prop_item: Dict[str, Any]) -> List[str]:
        """Extract domain includes (classes this property applies to)"""
        domain = prop_item.get('schema:domainIncludes', [])
        if isinstance(domain, dict):
            domain = [domain]
        elif not isinstance(domain, list):
            domain = []
            
        return [d.get('@id', '').replace('schema:', '') for d in domain if d.get('@id', '').startswith('schema:')]
        
    def _extract_range_includes(self, prop_item: Dict[str, Any]) -> List[str]:
        """Extract range includes (types this property can have)"""
        range_inc = prop_item.get('schema:rangeIncludes', [])
        if isinstance(range_inc, dict):
            range_inc = [range_inc]
        elif not isinstance(range_inc, list):
            range_inc = []
            
        return [r.get('@id', '').replace('schema:', '') for r in range_inc if r.get('@id', '').startswith('schema:')]
        
    def _build_class_hierarchy(self) -> None:
        """Build the class hierarchy tree"""
        for class_id, class_info in self.classes.items():
            parent = class_info['subclass_of']
            if parent and parent in self.classes:
                if parent not in self.class_hierarchy:
                    self.class_hierarchy[parent] = []
                self.class_hierarchy[parent].append(class_id)
                
        # Add properties to their domain classes
        for prop_id, prop_info in self.properties.items():
            for domain_class in prop_info['domain_includes']:
                if domain_class in self.classes:
                    self.classes[domain_class]['properties'].append(prop_id)
                    
    def get_core_types(self) -> List[str]:
        """Get the core Schema.org types we're interested in"""
        core_types = ['Person', 'Organization', 'Product', 'Place', 'Thing']
        return [t for t in core_types if t in self.classes]
        
    def get_class_properties(self, class_name: str) -> List[Dict[str, Any]]:
        """Get all properties applicable to a class"""
        if class_name not in self.classes:
            return []
            
        properties = []
        for prop_id in self.classes[class_name]['properties']:
            if prop_id in self.properties:
                properties.append(self.properties[prop_id])
        return properties

# Initialize ontology parser
ontology_path = Path(__file__).parent / "complete.jsonld"
parser = SchemaOrgOntologyParser(str(ontology_path))
parser.load_ontology()

# Create hierarchical knowledge graph with ontology-driven structure
hkg = HierarchicalKnowledgeGraph("schema_org_ontology")
print(f"Created hierarchical knowledge graph: {hkg.name}")

# Create hierarchical knowledge graph with ontology-driven structure
hkg = HierarchicalKnowledgeGraph("schema_org_ontology")
print(f"Created hierarchical knowledge graph: {hkg.name}")

# Create hierarchical levels based on ontology structure
print("\n🏗️ Creating ontology-driven hierarchical levels...")

# Level 1: Core Schema.org Types (highest abstraction)
hkg.add_level("core_types", {
    'level_name': "Core Schema.org Types",
    'level_description': "Fundamental Schema.org class types",
    'level_order': 0
})

# Level 2: Specific Classes (mid-level abstraction)
hkg.add_level("classes", {
    'level_name': "Schema.org Classes", 
    'level_description': "Specific Schema.org class definitions",
    'level_order': 1
})

# Level 3: Properties (detailed level)
hkg.add_level("properties", {
    'level_name': "Schema.org Properties",
    'level_description': "Properties and relationships between classes",
    'level_order': 2
})

# Level 4: Instances (actual data)
hkg.add_level("instances", {
    'level_name': "Entity Instances",
    'level_description': "Actual instances of Schema.org types",
    'level_order': 3
})

print(f"✅ Created {len(hkg.levels)} hierarchical levels")

# Populate core types level
core_types = parser.get_core_types()
print(f"\n📊 Adding {len(core_types)} core types to hierarchy...")

for core_type in core_types:
    class_info = parser.classes[core_type]
    hkg.add_entity_to_level(
        f"type_{core_type.lower()}", 
        "core_class",
        {
            'schema_class': core_type,
            'label': class_info['label'],
            'comment': class_info['comment'],
            'ontology_type': 'core_class'
        },
        "core_types"
    )
    print(f"  Added core type: {core_type}")

# Add subclasses to classes level
print(f"\n🔗 Adding subclasses and their properties...")

added_classes = set()
for core_type in core_types:
    # Add the core class itself to classes level
    if core_type not in added_classes:
        class_info = parser.classes[core_type]
        hkg.add_entity_to_level(
            f"class_{core_type.lower()}",
            "class_definition", 
            {
                'schema_class': core_type,
                'label': class_info['label'],
                'comment': class_info['comment'],
                'parent_class': class_info['subclass_of'],
                'ontology_type': 'class_definition'
            },
            "classes"
        )
        added_classes.add(core_type)
        
        # Add relationship from core type to class
        hkg.add_cross_level_relationship(
            f"type_{core_type.lower()}",
            f"class_{core_type.lower()}",
            "instantiates",
            {'direction': 'downward', 'source_level': 'core_types', 'target_level': 'classes'}
        )
        
        # Also add as semantic edge to the knowledge graph
        hkg.knowledge_graph.add_edge(
            [f"type_{core_type.lower()}", f"class_{core_type.lower()}"],
            data={'relationship_type': 'instantiates', 'direction': 'downward'},
            edge_type='hierarchy_relationship'
        )
    
    # Add subclasses if they exist
    if core_type in parser.class_hierarchy:
        for subclass in parser.class_hierarchy[core_type]:
            if subclass not in added_classes:
                subclass_info = parser.classes[subclass]
                hkg.add_entity_to_level(
                    f"class_{subclass.lower()}",
                    "subclass_definition",
                    {
                        'schema_class': subclass,
                        'label': subclass_info['label'], 
                        'comment': subclass_info['comment'],
                        'parent_class': subclass_info['subclass_of'],
                        'ontology_type': 'subclass_definition'
                    },
                    "classes"
                )
                added_classes.add(subclass)
                print(f"  Added subclass: {subclass} (subclass of {core_type})")
                
                # Add relationship from parent class to subclass
                hkg.add_cross_level_relationship(
                    f"class_{core_type.lower()}",
                    f"class_{subclass.lower()}",
                    "has_subclass",
                    {'direction': 'lateral', 'source_level': 'classes', 'target_level': 'classes'}
                )
                
                # Also add as semantic edge to the knowledge graph
                hkg.knowledge_graph.add_edge(
                    [f"class_{core_type.lower()}", f"class_{subclass.lower()}"],
                    data={'relationship_type': 'has_subclass', 'direction': 'lateral'},
                    edge_type='subclass_relationship'
                )

# Add properties to properties level
print(f"\n🏷️ Adding properties and their relationships...")

property_count = 0
for core_type in core_types:
    properties = parser.get_class_properties(core_type)
    for prop_info in properties[:5]:  # Limit to first 5 properties per class for demo
        prop_id = f"prop_{prop_info['id']}"
        hkg.add_entity_to_level(
            prop_id,
            "property_definition",
            {
                'property_name': prop_info['id'],
                'label': prop_info['label'],
                'comment': prop_info['comment'],
                'domain_classes': prop_info['domain_includes'],
                'range_classes': prop_info['range_includes'],
                'ontology_type': 'property_definition'
            },
            "properties"
        )
        
        # Link property to its domain class
        hkg.add_cross_level_relationship(
            f"class_{core_type.lower()}",
            prop_id,
            "has_property",
            {'direction': 'downward', 'source_level': 'classes', 'target_level': 'properties'}
        )
        
        # Also add as semantic edge to the knowledge graph
        hkg.knowledge_graph.add_edge(
            [f"class_{core_type.lower()}", prop_id],
            data={'relationship_type': 'has_property', 'direction': 'downward'},
            edge_type='property_relationship'
        )
        property_count += 1

print(f"  Added {property_count} properties to hierarchy")

# Add sample instances using our CSV data
print(f"\n📝 Adding sample instances from CSV data...")

# Load sample data from our test CSV files
csv_data_path = Path(__file__).parent.parent / "anant_test" / "data_schemaorg"

if csv_data_path.exists():
    instance_count = 0
    
    # Load persons
    persons_file = csv_data_path / "persons.csv"
    if persons_file.exists():
        persons_df = pl.read_csv(str(persons_file))
        for row in persons_df.head(3).iter_rows(named=True):  # Limit to 3 for demo
            person_id = f"instance_{row['entity_id']}"
            hkg.add_entity_to_level(
                person_id,
                "Person",
                {
                    'schema_type': 'Person',
                    'name': row['name'],
                    'email': row['email'],
                    'telephone': row['telephone'],
                    'description': row['description'],
                    'ontology_type': 'instance'
                },
                "instances"
            )
            
            # Link to Person class
            hkg.add_cross_level_relationship(
                "class_person",
                person_id,
                "instantiated_by",
                {'direction': 'downward', 'source_level': 'classes', 'target_level': 'instances'}
            )
            
            # Also add as semantic edge to the knowledge graph
            hkg.knowledge_graph.add_edge(
                ["class_person", person_id],
                data={'relationship_type': 'instantiated_by', 'direction': 'downward'},
                edge_type='instantiation_relationship'
            )
            instance_count += 1
    
    # Load organizations  
    orgs_file = csv_data_path / "organizations.csv"
    if orgs_file.exists():
        orgs_df = pl.read_csv(str(orgs_file))
        for row in orgs_df.head(2).iter_rows(named=True):  # Limit to 2 for demo
            org_id = f"instance_{row['entity_id']}"
            hkg.add_entity_to_level(
                org_id,
                "Organization",
                {
                    'schema_type': 'Organization',
                    'name': row['name'],
                    'description': row['description'],
                    'email': row['email'],
                    'telephone': row['telephone'],
                    'ontology_type': 'instance'
                },
                "instances"
            )
            
            # Link to Organization class
            hkg.add_cross_level_relationship(
                "class_organization",
                org_id,
                "instantiated_by",
                {'direction': 'downward', 'source_level': 'classes', 'target_level': 'instances'}
            )
            
            # Also add as semantic edge to the knowledge graph
            hkg.knowledge_graph.add_edge(
                ["class_organization", org_id],
                data={'relationship_type': 'instantiated_by', 'direction': 'downward'},
                edge_type='instantiation_relationship'
            )
            instance_count += 1
            
    print(f"  Added {instance_count} instances to hierarchy")
    
    # Add semantic relationships between instances based on CSV data
    print(f"\n🔗 Creating semantic relationships between instances...")
    semantic_relationships = 0
    
    if csv_data_path.exists():
        # Create worksFor relationships between persons and organizations
        persons_file = csv_data_path / "persons.csv"
        if persons_file.exists():
            persons_df = pl.read_csv(str(persons_file))
            for row in persons_df.head(3).iter_rows(named=True):
                person_id = f"instance_{row['entity_id']}"
                
                # Parse properties JSON to extract relationships
                try:
                    import json
                    properties = json.loads(row['properties'])
                    works_for = properties.get('worksFor', '')
                    address = properties.get('address', '')
                    
                    # Create worksFor relationship
                    if works_for:
                        org_entity_id = f"instance_{works_for}"
                        hkg.knowledge_graph.add_edge(
                            [person_id, org_entity_id],
                            data={'relationship_type': 'worksFor', 'source': 'csv_properties'},
                            edge_type='employment_relationship'
                        )
                        semantic_relationships += 1
                        print(f"    Added worksFor: {person_id} → {org_entity_id}")
                    
                    # Create address relationship  
                    if address:
                        place_entity_id = f"instance_{address}"
                        hkg.knowledge_graph.add_edge(
                            [person_id, place_entity_id],
                            data={'relationship_type': 'hasAddress', 'source': 'csv_properties'},
                            edge_type='location_relationship'
                        )
                        semantic_relationships += 1
                        print(f"    Added hasAddress: {person_id} → {place_entity_id}")
                        
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"    Warning: Could not parse properties for {person_id}: {e}")
    
    print(f"  Added {semantic_relationships} semantic relationships between instances")
else:
    print("  CSV data not found, skipping instance loading")

# Display hierarchy statistics
print(f"\n📊 Hierarchical Knowledge Graph Statistics:")
print(f"  Total levels: {len(hkg.levels)}")
for level_id, level_info in hkg.levels.items():
    entity_count = len(hkg.get_entities_at_level(level_id))
    print(f"  Level '{level_id}': {entity_count} entities")

print(f"  Total nodes: {hkg.num_nodes()}")
print(f"  Total edges: {hkg.num_edges()}")

# Demonstrate ontology-aware querying
print(f"\n🔍 Demonstrating ontology-aware queries...")

# Find all properties of Person class
person_props = parser.get_class_properties("Person")
print(f"  Person class has {len(person_props)} properties:")
for prop in person_props[:3]:  # Show first 3
    print(f"    - {prop['label']}: {prop['comment'][:50]}...")

# Show class hierarchy
print(f"\n🌳 Class hierarchy structure:")
for parent, children in parser.class_hierarchy.items():
    if parent in ['Thing', 'Person', 'Organization', 'Product', 'Place']:
        print(f"  {parent} -> {children[:3]}")  # Show first 3 children

# Export ontology-enhanced data
print(f"\n💾 Exporting ontology-enhanced data...")

# Save enhanced structure 
export_data = {
    'timestamp': '2025-10-20T15:59:16',
    'ontology_source': 'schema.org complete.jsonld',
    'ontology_classes': len(parser.classes),
    'ontology_properties': len(parser.properties),
    'hierarchy_levels': len(hkg.levels),
    'total_entities': hkg.num_nodes(),
    'total_relationships': hkg.num_edges(),
    'levels_breakdown': {
        level_id: len(hkg.get_entities_at_level(level_id)) 
        for level_id in hkg.levels.keys()
    },
    'class_hierarchy_sample': dict(list(parser.class_hierarchy.items())[:5]),  # First 5 for demo
    'core_classes': parser.get_core_types(),
    'sample_properties': [
        prop['label'] for prop in person_props[:5]
    ] if person_props else []
}

with open('ontology_analysis_results.json', 'w') as f:
    json.dump(export_data, f, indent=2)

print(f"✅ Exported ontology analysis to ontology_analysis_results.json")
print(f"\n🎉 Schema.org ontology-driven hierarchical knowledge graph complete!")
print(f"🎯 Successfully created ontology-aware {len(hkg.levels)}-level hierarchy:")
print(f"   • Loaded {len(parser.classes)} Schema.org classes and {len(parser.properties)} properties")
print(f"   • Built {len(hkg.levels)} hierarchical levels with proper ontological relationships")
print(f"   • Connected {hkg.num_nodes()} entities with {hkg.num_edges()} semantic relationships")
print(f"   • Demonstrates true ontology-driven knowledge graph construction!")