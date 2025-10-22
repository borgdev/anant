#!/usr/bin/env python3
"""
Quick test of the large-scale FHIR processor
"""

from large_scale_fhir_processor import LargeScaleFHIRProcessor

def test_processor():
    """Test the processor with a limited dataset."""
    
    print("🧪 TESTING LARGE-SCALE FHIR PROCESSOR")
    print("=" * 60)
    
    # Configuration for testing
    fhir_data_dir = "/home/amansingh/dev/andola/healthcare/synthea/output/fhir"
    output_dir = "/media/amansingh/data/fhir_test"
    
    try:
        # Initialize processor
        processor = LargeScaleFHIRProcessor(
            fhir_data_dir=fhir_data_dir,
            output_dir=output_dir,
            max_memory_gb=4.0,  # Conservative for testing
            batch_size=10,      # Small batches for testing
            max_workers=2       # Limited workers for testing
        )
        
        print("✅ Processor initialized successfully")
        
        # Discovery phase
        print("\n🔍 Running discovery...")
        discovery = processor.discover_fhir_files()
        print(f"📁 Found {discovery['total_files']:,} files")
        print(f"💾 Estimated size: {discovery['estimated_size_gb']:.1f} GB")
        
        # Process small sample
        print("\n⚡ Processing sample (100 files)...")
        result = processor.process_all_files(max_files=100)
        
        print(f"\n📊 TEST RESULTS:")
        stats = result['stats']
        print(f"Files processed: {stats.files_processed:,}")
        print(f"Total resources: {stats.total_resources:,}")
        print(f"Patients: {stats.total_patients:,}")
        print(f"Encounters: {stats.total_encounters:,}")
        print(f"Observations: {stats.total_observations:,}")
        print(f"Other resources: {stats.total_other_resources:,}")
        print(f"Processing time: {stats.processing_time:.2f}s")
        print(f"Rate: {stats.files_processed/stats.processing_time:.1f} files/sec")
        
        if stats.resource_type_counts:
            print(f"\n🔬 Resource Types Found:")
            for resource_type, count in sorted(stats.resource_type_counts.items()):
                print(f"  {resource_type}: {count:,}")
        
        partition_summary = result['partition_summary']
        print(f"\n📦 Parquet Output:")
        for resource_type, info in partition_summary.items():
            if info['partitions'] > 0:
                print(f"  {resource_type}: {info['partitions']} partitions, {info['total_size_mb']:.1f} MB")
        
        print("\n✅ TEST COMPLETED SUCCESSFULLY!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_processor()