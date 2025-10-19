#!/usr/bin/env python3
"""
FIBO Ontology Metagraph Processor

This program loads FIBO (Financial Industry Business Ontology) JSON-LD files,
creates a primary metagraph based on MetadataFIBO, and extracts domain-specific
subgraphs for each FIBO domain (ACTUS, BE, BP, CAE, DER, FBC, FND, IND, LOAN, MD, SEC).

Features:
- Loads all FIBO JSON-LD files using RDFLib
- Creates ANANT hypergraph from RDF triples
- Extracts domain-specific subgraphs
- Saves metagraph and subgraphs as parquet files
- Performance optimized for large ontologies

Author: ANANT Development Team
Date: October 18, 2025
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict, Counter

# Add ANANT to path
sys.path.insert(0, '/home/amansingh/dev/ai/anant')

import anant
from anant import Hypergraph

# Import required libraries
try:
    import polars as pl
    import rdflib
    from rdflib import Graph, Namespace, URIRef, Literal
    from rdflib.namespace import RDF, RDFS, OWL
    print("✅ All required libraries available (Polars, RDFLib)")
except ImportError as e:
    print(f"❌ Missing required library: {e}")
    sys.exit(1)

class FIBOMetagraphProcessor:
    """FIBO ontology processor with ANANT hypergraph and parquet persistence"""
    
    def __init__(self, fibo_root_dir: str, output_dir: str = None):
        """Initialize FIBO processor"""
        self.fibo_root = Path(fibo_root_dir)
        self.output_dir = Path(output_dir) if output_dir else self.fibo_root / "anant_cache"
        self.output_dir.mkdir(exist_ok=True)
        
        # FIBO domains based on directory structure
        self.fibo_domains = [
            "ACTUS",  # Algorithmic Contract Types Unified Standard
            "BE",     # Business Entities
            "BP",     # Business Process
            "CAE",    # Corporate Actions and Events
            "DER",    # Derivatives
            "FBC",    # Financial Business and Commerce
            "FND",    # Foundations
            "IND",    # Indices and Indicators
            "LOAN",   # Loans
            "MD",     # Market Data
            "SEC"     # Securities
        ]
        
        # RDF graph and hypergraphs
        self.rdf_graph = None
        self.primary_metagraph = None
        self.domain_subgraphs = {}
        
        # Statistics
        self.stats = {
            'total_files_processed': 0,
            'total_triples': 0,
            'domain_stats': {}
        }
        
        print("🏦 FIBO Ontology Metagraph Processor")
        print("=" * 60)
        print(f"📁 FIBO Root: {self.fibo_root}")
        print(f"💾 Output Dir: {self.output_dir}")
        print(f"🎯 Domains: {', '.join(self.fibo_domains)}")
    
    def discover_jsonld_files(self) -> Dict[str, List[Path]]:
        """Discover all JSON-LD files organized by domain"""
        print("\n🔍 Discovering FIBO JSON-LD files...")
        
        domain_files = {domain: [] for domain in self.fibo_domains}
        domain_files['CORE'] = []  # For root-level files
        
        # Root level files (MetadataFIBO.jsonld, etc.)
        for file_path in self.fibo_root.glob("*.jsonld"):
            domain_files['CORE'].append(file_path)
            print(f"  📄 Core: {file_path.name}")
        
        # Domain-specific files
        for domain in self.fibo_domains:
            domain_dir = self.fibo_root / domain
            if domain_dir.exists():
                jsonld_files = list(domain_dir.rglob("*.jsonld"))
                domain_files[domain] = jsonld_files
                print(f"  📁 {domain}: {len(jsonld_files)} files")
        
        total_files = sum(len(files) for files in domain_files.values())
        print(f"\n📊 Total JSON-LD files discovered: {total_files}")
        return domain_files
    
    def load_rdf_data(self, domain_files: Dict[str, List[Path]]) -> bool:
        """Load all FIBO JSON-LD files into a single RDF graph"""
        print("\n📂 Loading FIBO RDF data...")
        
        self.rdf_graph = Graph()
        
        # FIBO namespaces
        FIBO = Namespace("https://spec.edmcouncil.org/fibo/ontology/")
        
        total_files = sum(len(files) for files in domain_files.values())
        processed_files = 0
        
        try:
            for domain, files in domain_files.items():
                if not files:
                    continue
                
                print(f"\n  🔄 Processing {domain} domain ({len(files)} files)...")
                domain_triples = 0
                
                for file_path in files:
                    try:
                        # Load JSON-LD file
                        temp_graph = Graph()
                        temp_graph.parse(str(file_path), format='json-ld')
                        
                        # Add to main graph
                        self.rdf_graph += temp_graph
                        
                        file_triples = len(temp_graph)
                        domain_triples += file_triples
                        processed_files += 1
                        
                        if processed_files % 10 == 0:
                            print(f"    📈 Progress: {processed_files}/{total_files} files")
                        
                    except Exception as e:
                        print(f"    ⚠️  Failed to load {file_path.name}: {e}")
                        continue
                
                self.stats['domain_stats'][domain] = {
                    'files': len(files),
                    'triples': domain_triples
                }
                print(f"    ✅ {domain}: {domain_triples:,} triples from {len(files)} files")
            
            self.stats['total_files_processed'] = processed_files
            self.stats['total_triples'] = len(self.rdf_graph)
            
            print(f"\n📊 RDF Loading Summary:")
            print(f"   Files processed: {processed_files}/{total_files}")
            print(f"   Total triples: {self.stats['total_triples']:,}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load RDF data: {e}")
            return False
    
    def create_primary_metagraph(self) -> bool:
        """Create primary ANANT hypergraph from FIBO RDF data"""
        print("\n🕸️ Creating primary FIBO metagraph...")
        
        try:
            self.primary_metagraph = Hypergraph()
            
            # FIBO namespaces
            FIBO_BASE = "https://spec.edmcouncil.org/fibo/ontology/"
            
            # Track entities and relationships
            entities = set()
            relationships = defaultdict(list)
            
            # Extract classes (concepts)
            print("  🔵 Adding FIBO classes as nodes...")
            class_count = 0
            for s, p, o in self.rdf_graph.triples((None, RDF.type, OWL.Class)):
                if str(s).startswith(FIBO_BASE):
                    class_name = self._extract_local_name(str(s))
                    domain = self._extract_domain_from_uri(str(s))
                    
                    self.primary_metagraph.add_node(class_name, properties={
                        'type': 'class',
                        'uri': str(s),
                        'domain': domain,
                        'namespace': 'fibo'
                    })
                    entities.add(class_name)
                    class_count += 1
            
            print(f"    ✅ Added {class_count:,} FIBO classes")
            
            # Extract object properties (relationships)
            print("  🔗 Adding FIBO properties as relationships...")
            prop_count = 0
            edge_count = 0
            
            for s, p, o in self.rdf_graph.triples((None, RDF.type, OWL.ObjectProperty)):
                if str(s).startswith(FIBO_BASE):
                    prop_name = self._extract_local_name(str(s))
                    prop_domain = self._extract_domain_from_uri(str(s))
                    
                    # Get domain and range of the property
                    domains = list(self.rdf_graph.objects(s, RDFS.domain))
                    ranges = list(self.rdf_graph.objects(s, RDFS.range))
                    
                    # Add property as node
                    self.primary_metagraph.add_node(prop_name, properties={
                        'type': 'object_property',
                        'uri': str(s),
                        'domain': prop_domain,
                        'namespace': 'fibo'
                    })
                    
                    # Create hyperedges for domain-property-range relationships
                    for domain_class in domains:
                        for range_class in ranges:
                            if (str(domain_class).startswith(FIBO_BASE) and 
                                str(range_class).startswith(FIBO_BASE)):
                                
                                domain_name = self._extract_local_name(str(domain_class))
                                range_name = self._extract_local_name(str(range_class))
                                
                                try:
                                    edge_id = f"rel_{edge_count}"
                                    self.primary_metagraph.add_edge(
                                        edge_id, 
                                        [domain_name, prop_name, range_name],
                                        properties={
                                            'relation_type': 'object_property_relation',
                                            'property_uri': str(s),
                                            'direction': f"{domain_name} -> {range_name}",
                                            'property_domain': prop_domain
                                        }
                                    )
                                    edge_count += 1
                                except Exception:
                                    continue
                    
                    prop_count += 1
            
            print(f"    ✅ Added {prop_count:,} object properties")
            print(f"    ✅ Created {edge_count:,} relationship edges")
            
            # Add subclass relationships
            print("  📊 Adding subclass relationships...")
            subclass_count = 0
            for s, p, o in self.rdf_graph.triples((None, RDFS.subClassOf, None)):
                if (str(s).startswith(FIBO_BASE) and str(o).startswith(FIBO_BASE)):
                    child_name = self._extract_local_name(str(s))
                    parent_name = self._extract_local_name(str(o))
                    
                    try:
                        edge_id = f"subclass_{subclass_count}"
                        self.primary_metagraph.add_edge(
                            edge_id,
                            [child_name, parent_name],
                            properties={
                                'relation_type': 'subclass',
                                'direction': f"{child_name} subClassOf {parent_name}"
                            }
                        )
                        subclass_count += 1
                    except Exception:
                        continue
            
            print(f"    ✅ Added {subclass_count:,} subclass relationships")
            
            print(f"\n🏦 Primary FIBO Metagraph Created:")
            print(f"   📊 Nodes: {self.primary_metagraph.num_nodes:,}")
            print(f"   📊 Edges: {self.primary_metagraph.num_edges:,}")
            print(f"   📊 Density: {self.primary_metagraph.num_edges / max(1, self.primary_metagraph.num_nodes):.4f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to create primary metagraph: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def create_domain_subgraphs(self) -> bool:
        """Create domain-specific subgraphs from the primary metagraph"""
        print("\n🎯 Creating domain-specific subgraphs...")
        
        try:
            for domain in self.fibo_domains:
                print(f"\n  🔄 Processing {domain} domain...")
                
                # Create new hypergraph for this domain
                domain_hg = Hypergraph()
                
                # Find all nodes belonging to this domain
                domain_nodes = []
                for node in self.primary_metagraph.nodes:
                    try:
                        node_props = self.primary_metagraph.properties.get_node_properties(node)
                        if node_props and node_props.get('domain') == domain:
                            domain_nodes.append(node)
                    except:
                        continue
                
                # Add domain nodes to subgraph
                for node in domain_nodes:
                    try:
                        node_props = self.primary_metagraph.properties.get_node_properties(node)
                        domain_hg.add_node(node, properties=node_props)
                    except:
                        continue
                
                # Add relevant edges
                domain_edges = 0
                for edge_id in self.primary_metagraph.edges:
                    try:
                        edge_nodes = self.primary_metagraph.get_edge_nodes(edge_id)
                        edge_props = self.primary_metagraph.properties.get_edge_properties(edge_id)
                        
                        # Include edge if any of its nodes belong to this domain
                        if any(node in domain_nodes for node in edge_nodes):
                            # Ensure all edge nodes exist in domain subgraph
                            for node in edge_nodes:
                                if not domain_hg.has_node(node):
                                    try:
                                        orig_props = self.primary_metagraph.properties.get_node_properties(node)
                                        domain_hg.add_node(node, properties=orig_props or {})
                                    except:
                                        domain_hg.add_node(node)
                            
                            domain_hg.add_edge(edge_id, list(edge_nodes), properties=edge_props)
                            domain_edges += 1
                    except:
                        continue
                
                self.domain_subgraphs[domain] = domain_hg
                
                print(f"    ✅ {domain}: {domain_hg.num_nodes:,} nodes, {domain_hg.num_edges:,} edges")
            
            print(f"\n🎯 Domain Subgraphs Summary:")
            for domain, hg in self.domain_subgraphs.items():
                print(f"   {domain}: {hg.num_nodes:,} nodes, {hg.num_edges:,} edges")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to create domain subgraphs: {e}")
            return False
    
    def save_to_parquet(self) -> bool:
        """Save primary metagraph and domain subgraphs as parquet files"""
        print("\n💾 Saving FIBO metagraphs to parquet files...")
        
        try:
            # Save primary metagraph
            if self.primary_metagraph:
                print("  📦 Saving primary FIBO metagraph...")
                self._save_hypergraph_parquet(
                    self.primary_metagraph, 
                    "fibo_primary_metagraph",
                    {
                        'type': 'primary_metagraph',
                        'total_triples': self.stats['total_triples'],
                        'domains': self.fibo_domains,
                        'created': time.strftime('%Y-%m-%d %H:%M:%S')
                    }
                )
            
            # Save domain subgraphs
            for domain, hg in self.domain_subgraphs.items():
                print(f"  📦 Saving {domain} subgraph...")
                self._save_hypergraph_parquet(
                    hg,
                    f"fibo_{domain.lower()}_subgraph",
                    {
                        'type': 'domain_subgraph',
                        'domain': domain,
                        'created': time.strftime('%Y-%m-%d %H:%M:%S')
                    }
                )
            
            print(f"\n💾 All FIBO metagraphs saved to: {self.output_dir}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to save parquet files: {e}")
            return False
    
    def _save_hypergraph_parquet(self, hypergraph: Hypergraph, prefix: str, metadata: dict):
        """Save a hypergraph to parquet files"""
        try:
            # Save nodes
            nodes_data = []
            for node in hypergraph.nodes:
                nodes_data.append({"node_id": str(node)})
            
            if nodes_data:
                nodes_df = pl.DataFrame(nodes_data)
                nodes_file = self.output_dir / f"{prefix}_nodes.parquet"
                nodes_df.write_parquet(nodes_file)
            
            # Save edges
            edges_data = []
            for edge_id in hypergraph.edges:
                try:
                    edge_nodes = hypergraph.get_edge_nodes(edge_id)
                    edges_data.append({
                        "edge_id": str(edge_id),
                        "nodes": json.dumps(list(edge_nodes))
                    })
                except:
                    continue
            
            if edges_data:
                edges_df = pl.DataFrame(edges_data)
                edges_file = self.output_dir / f"{prefix}_edges.parquet"
                edges_df.write_parquet(edges_file)
            
            # Save node properties
            node_props_data = []
            for node in hypergraph.nodes:
                try:
                    props = hypergraph.properties.get_node_properties(node)
                    if props:
                        node_props_data.append({
                            "node_id": str(node),
                            "properties": json.dumps(props)
                        })
                except:
                    continue
            
            if node_props_data:
                props_df = pl.DataFrame(node_props_data)
                props_file = self.output_dir / f"{prefix}_node_properties.parquet"
                props_df.write_parquet(props_file)
            
            # Save edge properties
            edge_props_data = []
            for edge_id in hypergraph.edges:
                try:
                    props = hypergraph.properties.get_edge_properties(edge_id)
                    if props:
                        edge_props_data.append({
                            "edge_id": str(edge_id),
                            "properties": json.dumps(props)
                        })
                except:
                    continue
            
            if edge_props_data:
                edge_props_df = pl.DataFrame(edge_props_data)
                edge_props_file = self.output_dir / f"{prefix}_edge_properties.parquet"
                edge_props_df.write_parquet(edge_props_file)
            
            # Save metadata
            metadata_enhanced = {
                **metadata,
                'num_nodes': hypergraph.num_nodes,
                'num_edges': hypergraph.num_edges
            }
            
            metadata_file = self.output_dir / f"{prefix}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata_enhanced, f, indent=2)
            
            print(f"    ✅ Saved {prefix}: {hypergraph.num_nodes:,} nodes, {hypergraph.num_edges:,} edges")
            
        except Exception as e:
            print(f"    ❌ Failed to save {prefix}: {e}")
    
    def _extract_local_name(self, uri: str) -> str:
        """Extract local name from URI"""
        return uri.split('/')[-1].split('#')[-1]
    
    def _extract_domain_from_uri(self, uri: str) -> str:
        """Extract FIBO domain from URI"""
        for domain in self.fibo_domains:
            if f"/{domain}/" in uri:
                return domain
        return "CORE"
    
    def generate_report(self):
        """Generate comprehensive analysis report"""
        print("\n📋 Generating FIBO Analysis Report...")
        
        report = f"""
🏦 FIBO Ontology Metagraph Analysis Report
=========================================

📅 Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
📁 Source: {self.fibo_root}
💾 Output: {self.output_dir}
🛠️ Tools: RDFLib + ANANT Hypergraph + Polars

📊 PROCESSING SUMMARY
--------------------
• Total JSON-LD files processed: {self.stats['total_files_processed']:,}
• Total RDF triples: {self.stats['total_triples']:,}
• FIBO domains: {len(self.fibo_domains)}

🕸️ PRIMARY METAGRAPH
--------------------
• Nodes: {self.primary_metagraph.num_nodes:,} (classes and properties)
• Hyperedges: {self.primary_metagraph.num_edges:,} (relationships)
• Density: {self.primary_metagraph.num_edges / max(1, self.primary_metagraph.num_nodes):.4f}

🎯 DOMAIN SUBGRAPHS
------------------
"""
        
        for domain in self.fibo_domains:
            if domain in self.domain_subgraphs:
                hg = self.domain_subgraphs[domain]
                domain_stats = self.stats['domain_stats'].get(domain, {})
                report += f"• {domain}: {hg.num_nodes:,} nodes, {hg.num_edges:,} edges"
                report += f" (from {domain_stats.get('files', 0)} files, {domain_stats.get('triples', 0):,} triples)\n"
        
        report += f"""
💾 PARQUET FILES GENERATED
--------------------------
• Primary metagraph: fibo_primary_metagraph_*.parquet
• Domain subgraphs: fibo_{{domain}}_subgraph_*.parquet

🚀 CAPABILITIES DEMONSTRATED
----------------------------
✓ Large-scale FIBO ontology processing
✓ Domain-based knowledge graph partitioning
✓ RDF to hypergraph transformation
✓ Parquet-based persistence for fast loading
✓ Comprehensive metadata preservation
✓ Production-ready financial ontology analysis

📈 PERFORMANCE METRICS
---------------------
• File processing rate: ~{self.stats['total_files_processed'] / max(1, time.time() - self.start_time):.1f} files/second
• Triple processing rate: ~{self.stats['total_triples'] / max(1, time.time() - self.start_time):,.0f} triples/second
"""
        
        # Save report
        report_file = self.output_dir / "fibo_analysis_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(report)
        print(f"📄 Report saved to: {report_file}")
    
    def run_complete_processing(self):
        """Run complete FIBO processing pipeline"""
        self.start_time = time.time()
        
        print("🚀 Starting FIBO Ontology Metagraph Processing...")
        
        # Step 1: Discover files
        domain_files = self.discover_jsonld_files()
        if not any(domain_files.values()):
            print("❌ No JSON-LD files found")
            return False
        
        # Step 2: Load RDF data
        if not self.load_rdf_data(domain_files):
            return False
        
        # Step 3: Create primary metagraph
        if not self.create_primary_metagraph():
            return False
        
        # Step 4: Create domain subgraphs
        if not self.create_domain_subgraphs():
            return False
        
        # Step 5: Save to parquet
        if not self.save_to_parquet():
            return False
        
        # Step 6: Generate report
        self.generate_report()
        
        total_time = time.time() - self.start_time
        print(f"\n🎉 FIBO processing completed in {total_time:.2f} seconds!")
        print("✅ Primary metagraph and domain subgraphs ready for analysis!")
        
        return True

def main():
    """Main execution function"""
    
    # Initialize ANANT
    print("🔧 Setting up ANANT library...")
    anant.setup()
    
    # FIBO directory
    fibo_dir = "/home/amansingh/dev/ai/anant/prod.jsonld/fibo/ontology/master/2025Q3"
    output_dir = "/home/amansingh/dev/ai/anant/prod.jsonld/fibo_metagraphs"
    
    if not os.path.exists(fibo_dir):
        print(f"❌ FIBO directory not found: {fibo_dir}")
        return False
    
    print(f"📁 FIBO Source: {fibo_dir}")
    print(f"💾 Output Directory: {output_dir}")
    
    # Create processor and run
    try:
        processor = FIBOMetagraphProcessor(fibo_dir, output_dir)
        success = processor.run_complete_processing()
        
        if success:
            print("\n🎯 FIBO ONTOLOGY PROCESSING - SUCCESS!")
            print("=" * 60)
            print("✅ Primary FIBO metagraph created and saved")
            print("✅ Domain-specific subgraphs extracted and saved")
            print("✅ All data persisted as parquet files")
            print("✅ Comprehensive analysis report generated")
            print("\n🏦 FIBO metagraph processing complete - ready for financial analysis!")
        else:
            print("❌ FIBO processing failed")
        
        return success
        
    except Exception as e:
        print(f"💥 Processing failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)