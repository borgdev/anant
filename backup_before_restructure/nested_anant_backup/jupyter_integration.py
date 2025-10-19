"""
Jupyter Integration Module for Anant Library

Provides enhanced Jupyter notebook support including:
- Interactive widgets for hypergraph exploration
- Rich display representations 
- Progress bars and status indicators
- Interactive plotting integration
- Notebook-specific utilities and helpers

This module significantly improves the developer experience in Jupyter environments.
"""

import sys
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import polars as pl

# Check for Jupyter environment and import dependencies conditionally
try:
    from IPython.display import display, HTML, JSON, clear_output
    from IPython.core.magic import Magics, line_magic, cell_magic, magics_class
    from IPython import get_ipython
    JUPYTER_AVAILABLE = True
except ImportError:
    JUPYTER_AVAILABLE = False
    display = None
    HTML = None

try:
    import ipywidgets as widgets
    from ipywidgets import interact, interactive, fixed, interact_manual
    WIDGETS_AVAILABLE = True
except ImportError:
    WIDGETS_AVAILABLE = False
    widgets = None

try:
    from tqdm.notebook import tqdm
    TQDM_NOTEBOOK_AVAILABLE = True
except ImportError:
    try:
        from tqdm import tqdm
        TQDM_NOTEBOOK_AVAILABLE = False
    except ImportError:
        tqdm = None


class JupyterIntegration:
    """
    Main Jupyter integration class providing enhanced notebook support
    """
    
    def __init__(self):
        self.jupyter_available = JUPYTER_AVAILABLE
        self.widgets_available = WIDGETS_AVAILABLE
        self.tqdm_available = TQDM_NOTEBOOK_AVAILABLE
        
        if self.jupyter_available:
            self._setup_display_formatters()
    
    def _setup_display_formatters(self):
        """Setup custom display formatters for Anant objects"""
        try:
            from anant.classes.hypergraph import Hypergraph
            
            # Register HTML formatter for Hypergraph
            def hypergraph_html_formatter(hg):
                return self.create_hypergraph_summary_html(hg)
            
            # Register in IPython if available
            ip = get_ipython()
            if ip and hasattr(ip.display_formatter, 'formatters'):
                ip.display_formatter.formatters['text/html'].for_type(
                    Hypergraph, hypergraph_html_formatter
                )
        except ImportError:
            pass
    
    def create_hypergraph_summary_html(self, hg) -> str:
        """Create rich HTML summary for Hypergraph objects"""
        
        try:
            # Basic statistics
            num_nodes = hg.num_nodes
            num_edges = hg.num_edges
            
            # Node and edge samples
            node_sample = list(hg.nodes[:5]) if len(hg.nodes) > 0 else []
            edge_sample = list(hg.edges[:5]) if len(hg.edges) > 0 else []
            
            # Data summary
            data_shape = hg.incidences.data.shape if hasattr(hg.incidences, 'data') else (0, 0)
            
            html = f"""
            <div style="border: 1px solid #ddd; border-radius: 8px; padding: 15px; margin: 10px 0; background-color: #f9f9f9;">
                <h3 style="margin-top: 0; color: #333;">🔗 Hypergraph Summary</h3>
                
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 15px;">
                    <div>
                        <h4 style="color: #666; margin-bottom: 8px;">📊 Structure</h4>
                        <p><strong>Nodes:</strong> {num_nodes:,}</p>
                        <p><strong>Edges:</strong> {num_edges:,}</p>
                        <p><strong>Data Shape:</strong> {data_shape[0]:,} × {data_shape[1]}</p>
                    </div>
                    
                    <div>
                        <h4 style="color: #666; margin-bottom: 8px;">🎯 Density</h4>
                        <p><strong>Avg Nodes/Edge:</strong> {(num_nodes/max(1, num_edges)):.2f}</p>
                        <p><strong>Avg Edges/Node:</strong> {(num_edges/max(1, num_nodes)):.2f}</p>
                    </div>
                </div>
                
                {self._create_sample_section("Nodes", node_sample, num_nodes)}
                {self._create_sample_section("Edges", edge_sample, num_edges)}
                
                <div style="margin-top: 15px; padding: 10px; background-color: #e7f3ff; border-radius: 4px; font-size: 0.9em;">
                    💡 <strong>Tip:</strong> Use <code>.explore()</code> for interactive exploration or <code>.visualize()</code> for plotting
                </div>
            </div>
            """
            
            return html
            
        except Exception as e:
            return f"<div><strong>Hypergraph Object</strong> (display error: {e})</div>"
    
    def _create_sample_section(self, label: str, sample: List[str], total: int) -> str:
        """Create HTML section showing sample items"""
        if not sample:
            return f"<p><strong>{label}:</strong> None</p>"
        
        sample_str = ", ".join(f"<code>{item}</code>" for item in sample)
        more_text = f" ... and {total - len(sample):,} more" if total > len(sample) else ""
        
        return f"""
        <div style="margin: 10px 0;">
            <p><strong>{label} Sample:</strong> {sample_str}{more_text}</p>
        </div>
        """


class HypergraphExplorer:
    """
    Interactive widget for exploring hypergraphs in Jupyter notebooks
    """
    
    def __init__(self, hypergraph):
        self.hypergraph = hypergraph
        self.widgets_available = WIDGETS_AVAILABLE
        
        if not self.widgets_available:
            print("⚠️ ipywidgets not available. Install with: pip install ipywidgets")
            return
        
        self._create_explorer_interface()
    
    def _create_explorer_interface(self):
        """Create the interactive explorer interface"""
        
        # Node selection widget
        node_options = ["All"] + list(self.hypergraph.nodes[:100])  # Limit for performance
        self.node_selector = widgets.SelectMultiple(
            options=node_options,
            value=["All"],
            description="Nodes:",
            disabled=False,
            layout=widgets.Layout(height='150px', width='300px')
        )
        
        # Edge selection widget  
        edge_options = ["All"] + list(self.hypergraph.edges[:100])  # Limit for performance
        self.edge_selector = widgets.SelectMultiple(
            options=edge_options,
            value=["All"], 
            description="Edges:",
            disabled=False,
            layout=widgets.Layout(height='150px', width='300px')
        )
        
        # Analysis options
        self.analysis_selector = widgets.Dropdown(
            options=[
                ('Basic Statistics', 'basic_stats'),
                ('Degree Distribution', 'degree_dist'),
                ('Node Properties', 'node_props'),
                ('Edge Properties', 'edge_props'),
                ('Connectivity', 'connectivity')
            ],
            value='basic_stats',
            description="Analysis:"
        )
        
        # Update button
        self.update_button = widgets.Button(
            description="Update Analysis",
            button_style='primary',
            tooltip='Click to update the analysis',
            icon='refresh'
        )
        
        # Output area
        self.output_area = widgets.Output()
        
        # Setup interactions
        self.update_button.on_click(self._update_analysis)
        
        # Layout
        controls = widgets.VBox([
            widgets.HTML("<h3>🔍 Hypergraph Explorer</h3>"),
            widgets.HBox([self.node_selector, self.edge_selector]),
            self.analysis_selector,
            self.update_button
        ])
        
        self.explorer_widget = widgets.VBox([
            controls,
            self.output_area
        ])
        
        # Initial update
        self._update_analysis(None)
    
    def _update_analysis(self, button):
        """Update analysis based on current selections"""
        
        with self.output_area:
            clear_output(wait=True)
            
            try:
                # Get selected items
                selected_nodes = self.node_selector.value
                selected_edges = self.edge_selector.value  
                analysis_type = self.analysis_selector.value
                
                # Filter hypergraph if needed
                filtered_hg = self._filter_hypergraph(selected_nodes, selected_edges)
                
                # Perform analysis
                if analysis_type == 'basic_stats':
                    self._show_basic_stats(filtered_hg)
                elif analysis_type == 'degree_dist':
                    self._show_degree_distribution(filtered_hg)
                elif analysis_type == 'node_props':
                    self._show_node_properties(filtered_hg)
                elif analysis_type == 'edge_props':
                    self._show_edge_properties(filtered_hg)
                elif analysis_type == 'connectivity':
                    self._show_connectivity(filtered_hg)
                    
            except Exception as e:
                print(f"❌ Analysis error: {e}")
    
    def _filter_hypergraph(self, selected_nodes, selected_edges):
        """Filter hypergraph based on selections"""
        # For now, return original hypergraph
        # In a full implementation, this would create filtered views
        return self.hypergraph
    
    def _show_basic_stats(self, hg):
        """Display basic statistics"""
        print("📊 Basic Hypergraph Statistics")
        print("=" * 40)
        print(f"Nodes: {hg.num_nodes:,}")
        print(f"Edges: {hg.num_edges:,}")
        
        if hasattr(hg.incidences, 'data'):
            data = hg.incidences.data
            print(f"Data rows: {len(data):,}")
            print(f"Data columns: {len(data.columns)}")
            
            # Show sample data
            if len(data) > 0:
                print("\n📋 Sample Data:")
                display(data.head())
    
    def _show_degree_distribution(self, hg):
        """Display degree distribution"""
        print("📈 Degree Distribution")
        print("=" * 30)
        
        try:
            # Calculate node degrees
            node_degrees = {}
            for edge in hg.edges:
                edge_nodes = hg.edges[edge]
                for node in edge_nodes:
                    node_degrees[node] = node_degrees.get(node, 0) + 1
            
            if node_degrees:
                degrees = list(node_degrees.values())
                print(f"Min degree: {min(degrees)}")
                print(f"Max degree: {max(degrees)}")
                print(f"Avg degree: {sum(degrees)/len(degrees):.2f}")
                
                # Show distribution
                from collections import Counter
                degree_dist = Counter(degrees)
                
                print("\nDegree distribution (degree: count):")
                for degree in sorted(degree_dist.keys())[:10]:  # Show top 10
                    count = degree_dist[degree]
                    print(f"  {degree}: {count}")
                    
        except Exception as e:
            print(f"Could not calculate degrees: {e}")
    
    def _show_node_properties(self, hg):
        """Display node properties"""
        print("🔵 Node Properties")
        print("=" * 25)
        
        try:
            # Sample some nodes
            sample_nodes = list(hg.nodes[:10])
            
            for node in sample_nodes:
                try:
                    props = hg.get_node_properties(node)
                    print(f"\n{node}: {props}")
                except:
                    print(f"\n{node}: No properties")
                    
        except Exception as e:
            print(f"Could not show node properties: {e}")
    
    def _show_edge_properties(self, hg):
        """Display edge properties"""
        print("🔗 Edge Properties")
        print("=" * 25)
        
        try:
            # Sample some edges
            sample_edges = list(hg.edges[:10])
            
            for edge in sample_edges:
                try:
                    props = hg.get_edge_properties(edge)
                    nodes = hg.edges[edge]
                    print(f"\n{edge} → {nodes}: {props}")
                except:
                    print(f"\n{edge}: No properties")
                    
        except Exception as e:
            print(f"Could not show edge properties: {e}")
    
    def _show_connectivity(self, hg):
        """Display connectivity information"""
        print("🌐 Connectivity Analysis")
        print("=" * 30)
        
        try:
            # Basic connectivity metrics
            print(f"Total edges: {hg.num_edges}")
            print(f"Total nodes: {hg.num_nodes}")
            
            # Edge size distribution
            edge_sizes = []
            for edge in list(hg.edges)[:100]:  # Sample for performance
                try:
                    size = len(hg.edges[edge])
                    edge_sizes.append(size)
                except:
                    pass
            
            if edge_sizes:
                print(f"\nEdge size statistics:")
                print(f"  Min edge size: {min(edge_sizes)}")
                print(f"  Max edge size: {max(edge_sizes)}")
                print(f"  Avg edge size: {sum(edge_sizes)/len(edge_sizes):.2f}")
                
        except Exception as e:
            print(f"Could not analyze connectivity: {e}")
    
    def display(self):
        """Display the explorer widget"""
        if self.widgets_available:
            display(self.explorer_widget)
        else:
            print("❌ Widgets not available - install ipywidgets")


class ProgressReporter:
    """
    Enhanced progress reporting for Jupyter notebooks
    """
    
    def __init__(self, description: str = "Processing", total: Optional[int] = None):
        self.description = description
        self.total = total
        self.jupyter_available = JUPYTER_AVAILABLE
        self.tqdm_available = tqdm is not None
        
        if self.tqdm_available and TQDM_NOTEBOOK_AVAILABLE:
            self.progress_bar = tqdm(total=total, desc=description)
        elif self.tqdm_available:
            self.progress_bar = tqdm(total=total, desc=description)
        else:
            self.progress_bar = None
            self.current = 0
    
    def update(self, n: int = 1, description: Optional[str] = None):
        """Update progress"""
        if self.progress_bar:
            self.progress_bar.update(n)
            if description:
                self.progress_bar.set_description(description)
        else:
            self.current += n
            if self.jupyter_available and description:
                clear_output(wait=True)
                if self.total:
                    percent = (self.current / self.total) * 100
                    print(f"{description}: {self.current}/{self.total} ({percent:.1f}%)")
                else:
                    print(f"{description}: {self.current}")
    
    def close(self):
        """Close progress reporter"""
        if self.progress_bar:
            self.progress_bar.close()


@magics_class
class AnantMagics(Magics):
    """
    Custom IPython magic commands for Anant library
    """
    
    @line_magic
    def anant_info(self, line):
        """Display Anant library information"""
        info = {
            "version": "0.1.0",  
            "jupyter_available": JUPYTER_AVAILABLE,
            "widgets_available": WIDGETS_AVAILABLE,
            "tqdm_available": TQDM_NOTEBOOK_AVAILABLE
        }
        
        if JUPYTER_AVAILABLE:
            display(JSON(info))
        else:
            print(f"Anant Library Info: {info}")
    
    @cell_magic
    def anant_benchmark(self, line, cell):
        """Benchmark Anant operations"""
        import time
        
        print("⏱️ Benchmarking Anant operations...")
        start_time = time.time()
        
        # Execute the cell code
        exec(cell)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"✅ Execution completed in {duration:.3f} seconds")
        
        return duration


# Convenience functions

def setup_jupyter_integration():
    """Setup Jupyter integration for Anant library"""
    
    if not JUPYTER_AVAILABLE:
        print("❌ Jupyter environment not detected")
        return None
    
    # Create integration instance
    integration = JupyterIntegration()
    
    # Register magic commands
    try:
        ip = get_ipython()
        if ip:
            ip.register_magic_function(AnantMagics(ip).anant_info, 'line')
            ip.register_magic_function(AnantMagics(ip).anant_benchmark, 'cell')
            print("✅ Anant magic commands registered:")
            print("   %anant_info - Display library information")
            print("   %%anant_benchmark - Benchmark cell execution")
    except:
        print("⚠️ Could not register magic commands")
    
    print("✅ Jupyter integration setup complete")
    return integration


def explore_hypergraph(hypergraph):
    """Launch interactive hypergraph explorer"""
    explorer = HypergraphExplorer(hypergraph)
    explorer.display()
    return explorer


def create_progress_reporter(description: str = "Processing", total: Optional[int] = None):
    """Create a progress reporter for long operations"""
    return ProgressReporter(description, total)


# Auto-setup if imported in Jupyter
if JUPYTER_AVAILABLE and 'ipykernel' in sys.modules:
    try:
        _integration = setup_jupyter_integration()
    except:
        pass  # Silent fail for auto-setup