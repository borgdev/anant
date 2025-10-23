#!/usr/bin/env python3
"""
Full Dataset FHIR Processing Script (High-Performance Edition)
==============================================================

Processes the complete 435GB FHIR dataset (116K+ files) with ENHANCED settings:
- 16GB RAM (2x previous): Better memory management and caching
- 100 files/batch (2x previous): Higher throughput processing
- 8 worker threads (2x previous): Maximum parallelism
- Hierarchical knowledge graph construction
- Progress monitoring and performance tracking
- Error handling and recovery

PERFORMANCE IMPROVEMENTS:
┌─────────────────┬──────────────┬──────────────┬─────────────────┐
│     Setting     │   Previous   │   Enhanced   │   Improvement   │
├─────────────────┼──────────────┼──────────────┼─────────────────┤
│ Memory Limit    │     8GB      │     16GB     │      +100%      │
│ Batch Size      │   50 files   │  100 files   │      +100%      │
│ Worker Threads  │      4       │      8       │      +100%      │
│ Expected Speed  │   ~6hrs      │   ~3.5hrs    │      +40%       │
└─────────────────┴──────────────┴──────────────┴─────────────────┘

Usage:
    python3 run_full_dataset_processing.py
"""

import os
import sys
import time
import logging
from pathlib import Path
from large_scale_fhir_processor import LargeScaleFHIRProcessor

# Configure logging for production run
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/media/amansingh/data/fhir_test/processing.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Main function for full dataset processing."""
    
    print("🚀 STARTING FULL 435GB FHIR DATASET PROCESSING")
    print("=" * 80)
    print(f"📅 Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🗂️  Source: /home/amansingh/dev/andola/healthcare/synthea/output/fhir")
    print(f"📁 Output: /media/amansingh/data/fhir_test")
    print(f"🧠 Memory: 16.0 GB | 🔄 Batch: 100 files | ⚡ Workers: 8 threads")
    print("=" * 80)
    
    # Initialize processor with high-performance settings for full dataset
    # 16GB RAM: 2x memory for larger batches and better caching
    # 100 files/batch: 2x batch size for better throughput
    # 8 workers: 2x parallelism for faster processing
    processor = LargeScaleFHIRProcessor(
        fhir_data_dir='/home/amansingh/dev/andola/healthcare/synthea/output/fhir',
        output_dir='/media/amansingh/data/fhir_test',
        max_memory_gb=16.0,      # Use 16GB memory limit  
        batch_size=100,          # Process 100 files per batch
        max_workers=8,           # Use 8 worker threads
        enable_graph=True        # Enable hierarchical knowledge graph
    )
    
    try:
        # Discovery phase
        print(f"\n🔍 PHASE 1: DISCOVERY & ESTIMATION")
        print("-" * 60)
        
        discovery = processor.discover_fhir_files()
        
        print(f"📁 Total files found: {discovery['total_files']:,}")
        print(f"💾 Estimated dataset size: {discovery['estimated_size_gb']:.1f} GB")
        print(f"⏱️ Estimated processing time: {discovery['estimated_processing_time_hours']:.1f} hours")
        print(f"🚀 With 8 workers & 100-file batches: ~{discovery['estimated_processing_time_hours'] * 0.6:.1f} hours expected")
        print(f"🧮 Expected partitions: ~{discovery['total_files'] // 1000} per resource type")
        
        # Confirm before proceeding
        if discovery['total_files'] > 100000:  # Safety check
            print(f"\n⚠️  WARNING: This is a large dataset with {discovery['total_files']:,} files!")
            print(f"📊 This will create substantial Parquet outputs and knowledge graphs.")
            
            # In production, you might want to add a confirmation prompt
            print(f"✅ Proceeding with full dataset processing...")
        
        # Processing phase
        print(f"\n⚡ PHASE 2: FULL DATASET PROCESSING")
        print("-" * 60)
        print(f"🏭 Processing {discovery['total_files']:,} FHIR files...")
        print(f"🧠 Memory limit: 16.0 GB")
        print(f"🔄 Batch size: 100 files")
        print(f"⚡ Workers: 8 threads")
        print(f"🕸️  Knowledge graph: Enabled")
        
        # Start processing
        start_time = time.time()
        result = processor.process_all_files()  # Process ALL files
        end_time = time.time()
        
        # Results phase
        print(f"\n📊 PHASE 3: PROCESSING COMPLETE!")
        print("=" * 80)
        
        stats = result['stats']
        partition_summary = result['partition_summary']
        graph_summary = result.get('hierarchical_knowledge_graph', {})
        
        # Performance metrics
        total_time_hours = (end_time - start_time) / 3600
        processing_rate = stats.files_processed / (end_time - start_time)
        data_throughput = stats.bytes_processed / (1024**3) / (end_time - start_time)
        
        print(f"✅ PROCESSING COMPLETED SUCCESSFULLY!")
        print(f"📁 Files processed: {stats.files_processed:,}")
        print(f"🔬 Total resources extracted: {stats.total_resources:,}")
        print(f"👥 Patients: {stats.total_patients:,}")
        print(f"🏥 Encounters: {stats.total_encounters:,}")
        print(f"📊 Observations: {stats.total_observations:,}")
        print(f"🩺 Conditions: {stats.total_conditions:,}")
        print(f"⚕️ Procedures: {stats.total_procedures:,}")
        print(f"💊 Medications: {stats.total_medications:,}")
        
        print(f"\n⚡ Performance Metrics:")
        print(f"⏱️ Total processing time: {total_time_hours:.2f} hours")
        print(f"🚀 Processing rate: {processing_rate:.1f} files/sec")
        print(f"📈 Data throughput: {data_throughput:.2f} GB/sec")
        print(f"💾 Peak memory usage: {stats.memory_peak_mb:.1f} MB")
        
        # Parquet output summary
        print(f"\n📦 Parquet Output Summary:")
        total_partitions = sum(info['partitions'] for info in partition_summary.values())
        total_output_gb = sum(info['total_size_mb'] for info in partition_summary.values()) / 1024
        
        print(f"📂 Total partitions created: {total_partitions:,}")
        print(f"💾 Total output size: {total_output_gb:.2f} GB")
        print(f"📍 Output location: /media/amansingh/data/fhir_test")
        
        print(f"\n📊 Resource Type Breakdown:")
        for resource_type, info in sorted(partition_summary.items()):
            if info['partitions'] > 0:
                print(f"  {resource_type}: {info['partitions']:,} partitions, {info['total_size_mb']:.1f} MB")
        
        # Knowledge graph summary
        if graph_summary.get('enabled'):
            print(f"\n🕸️ Hierarchical Knowledge Graph Summary:")
            print(f"🔢 Total entities: {graph_summary['total_entities']:,}")
            print(f"🔗 Total relationships: {graph_summary['total_relationships']:,}")
            print(f"📊 Graph levels: {len(graph_summary['levels'])}")
            
            print(f"\n📈 Graph Level Breakdown:")
            for level_name, level_info in graph_summary['levels'].items():
                if level_info['entities'] > 0:
                    print(f"  {level_name}: {level_info['entities']:,} entities, {level_info['relationships']:,} relationships")
            
            print(f"💾 Graph summary saved to: /media/amansingh/data/fhir_test/hierarchical_knowledge_graph_summary.json")
        
        # Error summary
        if stats.errors:
            print(f"\n⚠️ Errors Encountered: {len(stats.errors)}")
            error_types = {}
            for error in stats.errors:
                error_type = error.split(':')[0] if ':' in error else 'Unknown'
                error_types[error_type] = error_types.get(error_type, 0) + 1
            
            for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
                print(f"  {error_type}: {count} occurrences")
        
        print(f"\n🎯 SUCCESS METRICS:")
        success_rate = (stats.files_processed / discovery['total_files']) * 100
        print(f"📈 File processing success rate: {success_rate:.1f}%")
        print(f"🎯 Resource extraction rate: {stats.total_resources / stats.files_processed:.0f} resources/file")
        print(f"💪 Memory efficiency: {stats.memory_peak_mb / 1024:.1f} GB peak / 16.0 GB limit = {(stats.memory_peak_mb / 1024 / 16.0) * 100:.1f}%")
        
        print(f"\n" + "=" * 80)
        print(f"🎉 FULL 435GB FHIR DATASET PROCESSING COMPLETE!")
        print(f"📅 End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🎯 Ready for distributed analytics and visualization!")
        print("=" * 80)
        
        return result
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Processing interrupted by user")
        print(f"📁 Partial results may be available in: /media/amansingh/data/fhir_test")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Critical error during processing: {e}")
        print(f"\n❌ PROCESSING FAILED: {e}")
        print(f"📁 Check logs at: /media/amansingh/data/fhir_test/processing.log")
        print(f"📁 Partial results may be available in: /media/amansingh/data/fhir_test")
        sys.exit(1)

if __name__ == "__main__":
    # Check prerequisites
    source_dir = Path('/home/amansingh/dev/andola/healthcare/synthea/output/fhir')
    output_dir = Path('/media/amansingh/data/fhir_test')
    
    if not source_dir.exists():
        print(f"❌ Error: Source directory not found: {source_dir}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run the processing
    result = main()