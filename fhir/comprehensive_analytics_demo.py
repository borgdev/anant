"""
Final Comprehensive FHIR Analytics Demo
======================================

This script demonstrates the complete capabilities of our enhanced FHIR system
by building a comprehensive knowledge graph and performing real healthcare analytics.
Includes comprehensive timing measurements for performance analysis.
"""

import sys
import json
import statistics
import time
from pathlib import Path
from datetime import datetime
import tempfile

# Add the parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fhir.enhanced_comprehensive_fhir_system import EnhancedFHIRSystem


def analyze_fhir_data_directly(data_dir: Path) -> tuple[dict, dict]:
    """Analyze FHIR data directly from JSON files for comprehensive statistics."""
    
    print("📊 COMPREHENSIVE FHIR DATA ANALYTICS")
    print("=" * 60)
    
    # Initialize timing tracking
    timing_stats = {
        'total_analysis_time': 0.0,
        'file_processing_time': 0.0,
        'patient_analysis_time': 0.0,
        'medication_analysis_time': 0.0,
        'observation_analysis_time': 0.0,
        'condition_analysis_time': 0.0,
        'provider_analysis_time': 0.0,
        'files_processed': 0
    }
    
    analytics = {
        'patients': {'total': 0, 'demographics': {}},
        'medications': {'total': 0, 'popular': {}},
        'observations': {'total': 0, 'types': {}, 'abnormal_values': []},
        'conditions': {'total': 0, 'common': {}},
        'practitioners': {'total': 0, 'specialties': {}},
        'organizations': {'total': 0, 'types': {}},
        'encounters': {'total': 0, 'types': {}},
        'procedures': {'total': 0, 'types': {}}
    }
    
    analysis_start_time = time.time()
    
    # Process each data file
    data_files = list(data_dir.glob("*.json"))
    print(f"🔍 Processing {len(data_files)} data files...")
    
    file_processing_start = time.time()
    
    for file_idx, file_path in enumerate(data_files, 1):
        try:
            file_start_time = time.time()
            
            print(f"   Processing file {file_idx}/{len(data_files)}: {file_path.name}")
            
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            if 'entry' in data:
                for entry in data['entry']:
                    resource = entry['resource']
                    resource_type = resource['resourceType']
                    
                    # Time individual resource type processing
                    resource_start = time.time()
                    
                    if resource_type == 'Patient':
                        _analyze_patient(resource, analytics)
                        timing_stats['patient_analysis_time'] += time.time() - resource_start
                    elif resource_type == 'MedicationRequest':
                        _analyze_medication(resource, analytics)
                        timing_stats['medication_analysis_time'] += time.time() - resource_start
                    elif resource_type == 'Observation':
                        _analyze_observation(resource, analytics)
                        timing_stats['observation_analysis_time'] += time.time() - resource_start
                    elif resource_type == 'Condition':
                        _analyze_condition(resource, analytics)
                        timing_stats['condition_analysis_time'] += time.time() - resource_start
                    elif resource_type == 'Practitioner':
                        _analyze_practitioner(resource, analytics)
                        timing_stats['provider_analysis_time'] += time.time() - resource_start
                    elif resource_type == 'Organization':
                        _analyze_organization(resource, analytics)
                        timing_stats['provider_analysis_time'] += time.time() - resource_start
                    elif resource_type == 'Encounter':
                        _analyze_encounter(resource, analytics)
                    elif resource_type == 'Procedure':
                        _analyze_procedure(resource, analytics)
            
            file_end_time = time.time()
            file_duration = file_end_time - file_start_time
            print(f"      ✓ Completed in {file_duration:.3f} seconds")
            timing_stats['files_processed'] += 1
                        
        except Exception as e:
            print(f"⚠️ Error processing {file_path.name}: {str(e)}")
    
    timing_stats['file_processing_time'] = time.time() - file_processing_start
    timing_stats['total_analysis_time'] = time.time() - analysis_start_time
    
    return analytics, timing_stats


def _analyze_patient(patient: dict, analytics: dict):
    """Analyze individual patient data."""
    analytics['patients']['total'] += 1
    
    # Gender analysis
    gender = patient.get('gender', 'unknown')
    if 'gender' not in analytics['patients']['demographics']:
        analytics['patients']['demographics']['gender'] = {}
    analytics['patients']['demographics']['gender'][gender] = analytics['patients']['demographics']['gender'].get(gender, 0) + 1
    
    # Age analysis
    if 'birthDate' in patient:
        birth_year = int(patient['birthDate'][:4])
        current_year = datetime.now().year
        age = current_year - birth_year
        
        if 'ages' not in analytics['patients']['demographics']:
            analytics['patients']['demographics']['ages'] = []
        analytics['patients']['demographics']['ages'].append(age)
    
    # Marital status
    if 'maritalStatus' in patient:
        status = patient['maritalStatus']['coding'][0]['display']
        if 'marital_status' not in analytics['patients']['demographics']:
            analytics['patients']['demographics']['marital_status'] = {}
        analytics['patients']['demographics']['marital_status'][status] = analytics['patients']['demographics']['marital_status'].get(status, 0) + 1


def _analyze_medication(medication: dict, analytics: dict):
    """Analyze medication data."""
    analytics['medications']['total'] += 1
    
    if 'medicationCodeableConcept' in medication:
        med_name = medication['medicationCodeableConcept']['coding'][0]['display']
        analytics['medications']['popular'][med_name] = analytics['medications']['popular'].get(med_name, 0) + 1


def _analyze_observation(observation: dict, analytics: dict):
    """Analyze observation data."""
    analytics['observations']['total'] += 1
    
    if 'code' in observation:
        obs_type = observation['code']['coding'][0]['display']
        analytics['observations']['types'][obs_type] = analytics['observations']['types'].get(obs_type, 0) + 1
        
        # Check for abnormal values
        if 'valueQuantity' in observation:
            value = observation['valueQuantity']['value']
            unit = observation['valueQuantity'].get('unit', '')
            
            # Simple abnormal value detection
            if obs_type == "Systolic blood pressure" and (value > 140 or value < 90):
                analytics['observations']['abnormal_values'].append(f"Abnormal {obs_type}: {value} {unit}")
            elif obs_type == "Diastolic blood pressure" and (value > 90 or value < 60):
                analytics['observations']['abnormal_values'].append(f"Abnormal {obs_type}: {value} {unit}")
            elif obs_type == "Blood glucose" and (value > 125 or value < 70):
                analytics['observations']['abnormal_values'].append(f"Abnormal {obs_type}: {value} {unit}")


def _analyze_condition(condition: dict, analytics: dict):
    """Analyze condition data."""
    analytics['conditions']['total'] += 1
    
    if 'code' in condition:
        condition_name = condition['code']['coding'][0]['display']
        analytics['conditions']['common'][condition_name] = analytics['conditions']['common'].get(condition_name, 0) + 1


def _analyze_practitioner(practitioner: dict, analytics: dict):
    """Analyze practitioner data."""
    analytics['practitioners']['total'] += 1
    
    if 'specialty' in practitioner and practitioner['specialty']:
        specialty = practitioner['specialty'][0]['coding'][0]['display']
        analytics['practitioners']['specialties'][specialty] = analytics['practitioners']['specialties'].get(specialty, 0) + 1


def _analyze_organization(organization: dict, analytics: dict):
    """Analyze organization data."""
    analytics['organizations']['total'] += 1
    
    if 'type' in organization and organization['type']:
        org_type = organization['type'][0]['coding'][0]['display']
        analytics['organizations']['types'][org_type] = analytics['organizations']['types'].get(org_type, 0) + 1


def _analyze_encounter(encounter: dict, analytics: dict):
    """Analyze encounter data."""
    analytics['encounters']['total'] += 1
    
    if 'class' in encounter:
        encounter_type = encounter['class']['display']
        analytics['encounters']['types'][encounter_type] = analytics['encounters']['types'].get(encounter_type, 0) + 1


def _analyze_procedure(procedure: dict, analytics: dict):
    """Analyze procedure data."""
    analytics['procedures']['total'] += 1
    
    if 'code' in procedure:
        procedure_type = procedure['code']['coding'][0]['display']
        analytics['procedures']['types'][procedure_type] = analytics['procedures']['types'].get(procedure_type, 0) + 1


def display_comprehensive_results(analytics: dict, timing_stats: dict):
    """Display comprehensive analytics results with timing information."""
    
    print("\n" + "=" * 80)
    print("🏥 COMPREHENSIVE HEALTHCARE ANALYTICS RESULTS")
    print("=" * 80)
    
    # Patient Analytics
    patients = analytics['patients']
    print(f"\n👥 PATIENT DEMOGRAPHICS (Total: {patients['total']})")
    print("-" * 50)
    
    if 'gender' in patients['demographics']:
        print("🚻 Gender Distribution:")
        for gender, count in patients['demographics']['gender'].items():
            percentage = (count / patients['total']) * 100
            print(f"   {gender.title()}: {count} ({percentage:.1f}%)")
    
    if 'ages' in patients['demographics']:
        ages = patients['demographics']['ages']
        avg_age = statistics.mean(ages)
        median_age = statistics.median(ages)
        print(f"\n📅 Age Statistics:")
        print(f"   Average Age: {avg_age:.1f} years")
        print(f"   Median Age: {median_age:.1f} years")
        print(f"   Age Range: {min(ages)} - {max(ages)} years")
        
        # Age groups
        age_groups = {'0-18': 0, '19-35': 0, '36-50': 0, '51-65': 0, '65+': 0}
        for age in ages:
            if age <= 18:
                age_groups['0-18'] += 1
            elif age <= 35:
                age_groups['19-35'] += 1
            elif age <= 50:
                age_groups['36-50'] += 1
            elif age <= 65:
                age_groups['51-65'] += 1
            else:
                age_groups['65+'] += 1
        
        print("\n👨‍👩‍👧‍👦 Age Groups:")
        for group, count in age_groups.items():
            percentage = (count / len(ages)) * 100
            print(f"   {group}: {count} ({percentage:.1f}%)")
    
    if 'marital_status' in patients['demographics']:
        print("\n💑 Marital Status:")
        for status, count in patients['demographics']['marital_status'].items():
            percentage = (count / patients['total']) * 100
            print(f"   {status}: {count} ({percentage:.1f}%)")
    
    # Medication Analytics
    medications = analytics['medications']
    print(f"\n💊 MEDICATION ANALYSIS (Total Prescriptions: {medications['total']})")
    print("-" * 50)
    
    if medications['popular']:
        sorted_meds = sorted(medications['popular'].items(), key=lambda x: x[1], reverse=True)
        print("🏆 Most Popular Medications:")
        for i, (med_name, count) in enumerate(sorted_meds[:10], 1):
            percentage = (count / medications['total']) * 100
            print(f"   {i}. {med_name}: {count} prescriptions ({percentage:.1f}%)")
    
    # Clinical Observations
    observations = analytics['observations']
    print(f"\n🩺 CLINICAL OBSERVATIONS (Total: {observations['total']})")
    print("-" * 50)
    
    if observations['types']:
        print("🔬 Observation Types:")
        for obs_type, count in observations['types'].items():
            percentage = (count / observations['total']) * 100
            print(f"   {obs_type}: {count} ({percentage:.1f}%)")
    
    if observations['abnormal_values']:
        print(f"\n⚠️ Abnormal Values Detected ({len(observations['abnormal_values'])}):")
        for abnormal in observations['abnormal_values'][:10]:  # Show first 10
            print(f"   🚨 {abnormal}")
        if len(observations['abnormal_values']) > 10:
            print(f"   ... and {len(observations['abnormal_values']) - 10} more")
    
    # Medical Conditions
    conditions = analytics['conditions']
    print(f"\n🏥 MEDICAL CONDITIONS (Total: {conditions['total']})")
    print("-" * 50)
    
    if conditions['common']:
        sorted_conditions = sorted(conditions['common'].items(), key=lambda x: x[1], reverse=True)
        print("📋 Most Common Conditions:")
        for i, (condition, count) in enumerate(sorted_conditions, 1):
            percentage = (count / conditions['total']) * 100
            print(f"   {i}. {condition}: {count} cases ({percentage:.1f}%)")
    
    # Healthcare Providers
    practitioners = analytics['practitioners']
    organizations = analytics['organizations']
    print(f"\n👨‍⚕️ HEALTHCARE PROVIDERS")
    print("-" * 50)
    print(f"Practitioners: {practitioners['total']}")
    print(f"Organizations: {organizations['total']}")
    
    if practitioners['specialties']:
        print("\n🩺 Medical Specialties:")
        for specialty, count in practitioners['specialties'].items():
            percentage = (count / practitioners['total']) * 100 if practitioners['total'] > 0 else 0
            print(f"   {specialty}: {count} ({percentage:.1f}%)")
    
    # Encounters and Procedures
    encounters = analytics['encounters']
    procedures = analytics['procedures']
    print(f"\n🏥 HEALTHCARE UTILIZATION")
    print("-" * 50)
    print(f"Total Encounters: {encounters['total']}")
    print(f"Total Procedures: {procedures['total']}")
    
    # Summary Insights
    print(f"\n🔍 KEY INSIGHTS")
    print("-" * 50)
    if patients['total'] > 0:
        print(f"• Average medications per patient: {medications['total'] / patients['total']:.1f}")
        print(f"• Average observations per patient: {observations['total'] / patients['total']:.1f}")
        print(f"• Average conditions per patient: {conditions['total'] / patients['total']:.1f}")
        print(f"• Average encounters per patient: {encounters['total'] / patients['total']:.1f}")
    
    if practitioners['total'] > 0:
        print(f"• Patient-to-provider ratio: {patients['total'] / practitioners['total']:.1f}:1")
    
    abnormal_percentage = (len(observations['abnormal_values']) / observations['total']) * 100 if observations['total'] > 0 else 0
    print(f"• Abnormal vital signs rate: {abnormal_percentage:.1f}%")
    
    # Display comprehensive timing information
    print(f"\n⏱️ PERFORMANCE ANALYTICS")
    print("-" * 50)
    print(f"• Total analysis time: {timing_stats['total_analysis_time']:.3f} seconds")
    print(f"• File processing time: {timing_stats['file_processing_time']:.3f} seconds")
    print(f"• Files processed: {timing_stats['files_processed']}")
    print(f"• Average time per file: {timing_stats['file_processing_time']/timing_stats['files_processed']:.3f} seconds" if timing_stats['files_processed'] > 0 else "• Average time per file: N/A")
    
    print(f"\n🔍 Resource Analysis Timing:")
    print(f"   Patient analysis: {timing_stats['patient_analysis_time']:.3f} seconds")
    print(f"   Medication analysis: {timing_stats['medication_analysis_time']:.3f} seconds") 
    print(f"   Observation analysis: {timing_stats['observation_analysis_time']:.3f} seconds")
    print(f"   Condition analysis: {timing_stats['condition_analysis_time']:.3f} seconds")
    print(f"   Provider analysis: {timing_stats['provider_analysis_time']:.3f} seconds")
    
    # Calculate throughput metrics
    total_resources = (patients['total'] + analytics['medications']['total'] + 
                      analytics['observations']['total'] + analytics['conditions']['total'] +
                      analytics['practitioners']['total'] + analytics['organizations']['total'] +
                      analytics['encounters']['total'] + analytics['procedures']['total'])
    
    if timing_stats['total_analysis_time'] > 0:
        resources_per_second = total_resources / timing_stats['total_analysis_time']
        print(f"\n📊 Throughput Metrics:")
        print(f"   Total resources processed: {total_resources:,}")
        print(f"   Processing rate: {resources_per_second:.0f} resources/second")


def main():
    """Main function to run comprehensive FHIR analytics."""
    
    print("🚀 COMPREHENSIVE FHIR HEALTHCARE ANALYTICS SYSTEM")
    print("=" * 70)
    
    # Track overall timing
    main_start_time = time.time()
    
    # Create temporary test environment
    temp_dir = Path(tempfile.mkdtemp())
    schema_dir = temp_dir / 'schema'
    data_dir = temp_dir / 'data' / 'output' / 'fhir'
    
    # Initialize and build the enhanced system
    print("🏗️ Building Enhanced FHIR Knowledge Graph...")
    build_start_time = time.time()
    
    enhanced_system = EnhancedFHIRSystem(
        schema_dir=str(schema_dir),
        data_dir=str(data_dir),
        graph_name="ComprehensiveAnalyticsFHIRKG"
    )
    
    # Build the knowledge graph
    build_results = enhanced_system.build_enhanced_graph(force_load_all=True)
    build_end_time = time.time()
    build_duration = build_end_time - build_start_time
    
    if build_results['status'] in ['success', 'completed_with_errors']:
        print(f"✅ Knowledge graph built successfully in {build_duration:.2f} seconds!")
        
        # Analyze data directly from files for detailed statistics
        print("\n🔍 Performing comprehensive data analysis...")
        analytics_start_time = time.time()
        
        analytics_results, timing_stats = analyze_fhir_data_directly(data_dir)
        
        analytics_end_time = time.time()
        analytics_duration = analytics_end_time - analytics_start_time
        
        print(f"✅ Analytics completed in {analytics_duration:.2f} seconds!")
        
        # Display comprehensive results
        display_comprehensive_results(analytics_results, timing_stats)
        
        # Display overall system timing
        main_end_time = time.time()
        total_duration = main_end_time - main_start_time
        
        print(f"\n⏱️ OVERALL SYSTEM PERFORMANCE")
        print("-" * 50)
        print(f"• Knowledge graph build time: {build_duration:.2f} seconds")
        print(f"• Data analytics time: {analytics_duration:.2f} seconds")
        print(f"• Total execution time: {total_duration:.2f} seconds")
        print(f"• System efficiency: {(timing_stats['total_analysis_time']/total_duration)*100:.1f}% active analysis time")
        
        print("\n" + "=" * 70)
        print("✅ COMPREHENSIVE FHIR ANALYTICS COMPLETE!")
        print("=" * 70)
        
    else:
        print(f"❌ Failed to build knowledge graph: {build_results.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()