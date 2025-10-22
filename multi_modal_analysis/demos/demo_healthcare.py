"""
Healthcare Multi-Modal Analysis Demo
====================================

Demonstrates multi-modal relationship analysis for healthcare patient journeys.

Modalities:
- Treatments: Patient-treatment relationships
- Diagnoses: Patient-diagnosis relationships
- Providers: Patient-provider relationships
- Medications: Patient-medication relationships

Cross-Modal Insights:
- Treatment effectiveness across patient populations
- Provider coordination patterns
- Medication interaction networks
- Care pathway optimization
"""

import sys
from pathlib import Path
import random
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import polars as pl
    import numpy as np
except ImportError:
    print("⚠️  Required dependencies not installed. Please install: polars numpy")
    sys.exit(1)

from core.multi_modal_hypergraph import MultiModalHypergraph


class MockHypergraph:
    """Mock hypergraph for demo"""
    def __init__(self, data):
        self.data = data
        self.incidences = type('obj', (object,), {'data': data})()
        if 'nodes' in data.columns:
            self._nodes = set(data['nodes'].unique())
    
    def nodes(self):
        return self._nodes


def generate_healthcare_data(
    num_patients: int = 300,
    num_treatments: int = 50,
    num_diagnoses: int = 30,
    num_providers: int = 20,
    num_medications: int = 40
):
    """Generate synthetic healthcare data"""
    
    print("📊 Generating Healthcare Data...")
    print(f"   Patients: {num_patients}")
    print(f"   Treatments: {num_treatments}")
    print(f"   Diagnoses: {num_diagnoses}")
    
    patients = [f"patient_{i:04d}" for i in range(num_patients)]
    treatments = [f"treatment_{i:03d}" for i in range(num_treatments)]
    diagnoses = [f"diagnosis_{i:03d}" for i in range(num_diagnoses)]
    providers = [f"provider_{i:03d}" for i in range(num_providers)]
    medications = [f"medication_{i:03d}" for i in range(num_medications)]
    
    # Generate treatment records
    treatment_records = []
    for i in range(num_patients * 3):  # ~3 treatments per patient
        treatment_id = f"treatment_event_{i:05d}"
        patient = random.choice(patients)
        treatment = random.choice(treatments)
        
        treatment_records.extend([
            {'edges': treatment_id, 'nodes': patient, 'weight': 1.0, 'role': 'patient'},
            {'edges': treatment_id, 'nodes': treatment, 'weight': 1.0, 'role': 'treatment'}
        ])
    
    # Generate diagnosis records
    diagnosis_records = []
    for i in range(num_patients * 2):  # ~2 diagnoses per patient
        diagnosis_id = f"diagnosis_event_{i:05d}"
        patient = random.choice(patients)
        diagnosis = random.choice(diagnoses)
        
        diagnosis_records.extend([
            {'edges': diagnosis_id, 'nodes': patient, 'weight': 1.0, 'role': 'patient'},
            {'edges': diagnosis_id, 'nodes': diagnosis, 'weight': 1.0, 'role': 'diagnosis'}
        ])
    
    # Generate provider encounters
    provider_records = []
    for i in range(num_patients * 4):  # ~4 encounters per patient
        encounter_id = f"encounter_{i:05d}"
        patient = random.choice(patients)
        provider = random.choice(providers)
        
        provider_records.extend([
            {'edges': encounter_id, 'nodes': patient, 'weight': 1.0, 'role': 'patient'},
            {'edges': encounter_id, 'nodes': provider, 'weight': 1.0, 'role': 'provider'}
        ])
    
    # Generate medication prescriptions
    medication_records = []
    for i in range(num_patients * 3):  # ~3 medications per patient
        prescription_id = f"prescription_{i:05d}"
        patient = random.choice(patients)
        medication = random.choice(medications)
        
        medication_records.extend([
            {'edges': prescription_id, 'nodes': patient, 'weight': 1.0, 'role': 'patient'},
            {'edges': prescription_id, 'nodes': medication, 'weight': 1.0, 'role': 'medication'}
        ])
    
    return {
        'treatments': pl.DataFrame(treatment_records),
        'diagnoses': pl.DataFrame(diagnosis_records),
        'providers': pl.DataFrame(provider_records),
        'medications': pl.DataFrame(medication_records)
    }


def demo_healthcare_multimodal():
    """Demo: Healthcare multi-modal analysis"""
    
    print("\n" + "="*70)
    print("🏥 Healthcare Multi-Modal Analysis Demo")
    print("="*70)
    
    # Generate data
    data = generate_healthcare_data()
    
    # Create hypergraphs
    print("\n📦 Creating Healthcare Hypergraphs...")
    treatment_hg = MockHypergraph(data['treatments'])
    diagnosis_hg = MockHypergraph(data['diagnoses'])
    provider_hg = MockHypergraph(data['providers'])
    medication_hg = MockHypergraph(data['medications'])
    
    # Build multi-modal hypergraph
    print("\n🔗 Constructing Multi-Modal Patient Journey Graph...")
    mmhg = MultiModalHypergraph(name="patient_journey")
    
    mmhg.add_modality("treatments", treatment_hg, weight=2.0, 
                     description="Treatment interventions")
    mmhg.add_modality("diagnoses", diagnosis_hg, weight=2.5,
                     description="Clinical diagnoses")
    mmhg.add_modality("providers", provider_hg, weight=1.5,
                     description="Healthcare provider encounters")
    mmhg.add_modality("medications", medication_hg, weight=2.0,
                     description="Medication prescriptions")
    
    # Summary
    summary = mmhg.generate_summary()
    print(f"\n📊 Patient Journey Summary:")
    print(f"   Total Patients: {summary['total_unique_entities']}")
    print(f"   Modalities: {summary['num_modalities']}")
    print(f"   Avg Modalities per Patient: {summary['avg_modalities_per_entity']:.2f}")
    
    # Find complex care patients (in multiple modalities)
    print("\n🔍 Finding Complex Care Patients...")
    complex_patients = mmhg.find_modal_bridges(min_modalities=3)
    print(f"   Patients in 3+ modalities: {len(complex_patients)}")
    
    # Care coordination analysis
    print("\n🔗 Care Coordination Analysis...")
    coordination = mmhg.discover_inter_modal_relationships(
        "treatments", "providers"
    )
    print(f"   Treatment-Provider connections: {len(coordination)}")
    
    # Treatment-diagnosis correlation
    print("\n📊 Clinical Correlations...")
    treat_diag_corr = mmhg.compute_modal_correlation("treatments", "diagnoses")
    treat_med_corr = mmhg.compute_modal_correlation("treatments", "medications")
    
    print(f"   Treatment-Diagnosis correlation: {treat_diag_corr:.3f}")
    print(f"   Treatment-Medication correlation: {treat_med_corr:.3f}")
    
    # Cross-modal patterns
    print("\n🔎 Detecting Care Patterns...")
    patterns = mmhg.detect_cross_modal_patterns(min_support=5)
    print(f"   Care patterns detected: {len(patterns)}")
    
    for i, pattern in enumerate(patterns[:3], 1):
        print(f"\n   Pattern {i}:")
        print(f"      Type: {pattern['type']}")
        print(f"      Description: {pattern['description']}")
        print(f"      Support: {pattern['support']}")
    
    # Business insights
    print("\n" + "="*70)
    print("💡 Clinical Insights")
    print("="*70)
    
    print(f"\n1. Patient Complexity:")
    print(f"   • {len(complex_patients)} patients require coordinated care")
    print(f"   • Focus on care team coordination for these patients")
    
    print(f"\n2. Care Coordination:")
    if treat_diag_corr > 0.5:
        print(f"   • Strong treatment-diagnosis alignment ({treat_diag_corr:.1%})")
    else:
        print(f"   • Review diagnostic consistency ({treat_diag_corr:.1%})")
    
    print(f"\n3. Medication Management:")
    if treat_med_corr > 0.6:
        print(f"   • Good treatment-medication correlation")
    else:
        print(f"   • Review prescribing patterns")
    
    print("\n" + "="*70)
    print("✅ Healthcare Demo Complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    demo_healthcare_multimodal()
