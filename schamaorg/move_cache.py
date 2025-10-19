#!/usr/bin/env python3
"""
Simple test to move Schema.org cache files from governance to schamaorg folder
"""

import os
import shutil
from pathlib import Path

def move_cache_files():
    """Move existing cache files from governance to schamaorg folder"""
    
    governance_dir = Path("/home/amansingh/dev/ai/anant/governance")
    schamaorg_dir = Path("/home/amansingh/dev/ai/anant/schamaorg")
    
    cache_files = [
        "schema_org_nodes.parquet",
        "schema_org_edges.parquet", 
        "schema_org_node_properties.parquet",
        "schema_org_edge_properties.parquet",
        "schema_org_metadata.json"
    ]
    
    print("🚚 Moving Schema.org cache files from governance to schamaorg folder...")
    
    moved_count = 0
    for filename in cache_files:
        source_path = governance_dir / filename
        dest_path = schamaorg_dir / filename
        
        if source_path.exists():
            try:
                shutil.move(str(source_path), str(dest_path))
                print(f"  ✅ Moved {filename}")
                moved_count += 1
            except Exception as e:
                print(f"  ❌ Failed to move {filename}: {e}")
        else:
            print(f"  ⚠️  {filename} not found in governance folder")
    
    print(f"\n📊 Successfully moved {moved_count} cache files to schamaorg folder")
    
    # Verify files are now in schamaorg
    print(f"\n📋 Files now in {schamaorg_dir}:")
    for filename in cache_files:
        dest_path = schamaorg_dir / filename
        if dest_path.exists():
            size = dest_path.stat().st_size
            print(f"  ✅ {filename} ({size:,} bytes)")
        else:
            print(f"  ❌ {filename} - not found")

if __name__ == "__main__":
    move_cache_files()
    print("\n🎯 Cache files successfully relocated to schamaorg folder!")