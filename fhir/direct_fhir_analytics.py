"""
Direct FHIR Data Analytics - Patient Demographics & Statistics
============================================================

This script analyzes FHIR data directly from JSON files to provide
comprehensive healthcare analytics including:
- Patient count and demographics
- Average age calculation
- Popular medications
- Clinical observations
- Medical conditions
"""

import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import tempfile


class DirectFHIRAnalyzer:
    """Direct FHIR data analytics from JSON files."""
    
    def __init__(self, data_dir: str):
        """Initialize the analyzer."""
        self.data_dir = Path(data_dir)
        self.patients = []
        self.medications = []
        self.observations = []
        self.conditions = []
        self.practitioners = []
        self.organizations = []
        
    def create_comprehensive_test_data(self):
        """Create comprehensive FHIR test data."""
        print("📊 Creating comprehensive FHIR test data...")
        
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create patients data
        patients_data = {
            "resourceType": "Bundle",
            "id": "patients-bundle",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Patient",
                        "id": "patient-001",
                        "name": [{"family": "Smith", "given": ["John"]}],
                        "gender": "male",
                        "birthDate": "1985-03-15",
                        "maritalStatus": {"coding": [{"code": "M", "display": "Married"}]},
                        "address": [{"city": "Boston", "state": "MA"}]
                    }
                },
                {
                    "resource": {
                        "resourceType": "Patient",
                        "id": "patient-002",
                        "name": [{"family": "Johnson", "given": ["Mary"]}],
                        "gender": "female",
                        "birthDate": "1992-07-22",
                        "maritalStatus": {"coding": [{"code": "S", "display": "Single"}]},
                        "address": [{"city": "Cambridge", "state": "MA"}]
                    }
                },
                {
                    "resource": {
                        "resourceType": "Patient",
                        "id": "patient-003",
                        "name": [{"family": "Williams", "given": ["Robert"]}],
                        "gender": "male",
                        "birthDate": "1978-11-08",
                        "maritalStatus": {"coding": [{"code": "D", "display": "Divorced"}]},
                        "address": [{"city": "Somerville", "state": "MA"}]
                    }
                },
                {
                    "resource": {
                        "resourceType": "Patient",
                        "id": "patient-004",
                        "name": [{"family": "Brown", "given": ["Sarah"]}],
                        "gender": "female",
                        "birthDate": "1965-04-30",
                        "maritalStatus": {"coding": [{"code": "M", "display": "Married"}]},
                        "address": [{"city": "Newton", "state": "MA"}]
                    }
                },
                {
                    "resource": {
                        "resourceType": "Patient",
                        "id": "patient-005",
                        "name": [{"family": "Davis", "given": ["Michael"]}],
                        "gender": "male",
                        "birthDate": "2001-12-03",
                        "maritalStatus": {"coding": [{"code": "S", "display": "Single"}]},
                        "address": [{"city": "Brookline", "state": "MA"}]
                    }
                },
                {
                    "resource": {
                        "resourceType": "Patient",
                        "id": "patient-006",
                        "name": [{"family": "Wilson", "given": ["Lisa"]}],
                        "gender": "female",
                        "birthDate": "1990-08-14",
                        "maritalStatus": {"coding": [{"code": "S", "display": "Single"}]},
                        "address": [{"city": "Arlington", "state": "MA"}]
                    }
                }
            ]
        }
        
        # Create medications data
        medications_data = {
            "resourceType": "Bundle",
            "id": "medications-bundle",
            "entry": [
                {
                    "resource": {
                        "resourceType": "MedicationRequest",
                        "id": "med-001",
                        "status": "active",
                        "medicationCodeableConcept": {"coding": [{"display": "Lisinopril 10mg"}]},
                        "subject": {"reference": "Patient/patient-001"},
                        "dosageInstruction": [{"text": "10mg once daily"}]
                    }
                },
                {
                    "resource": {
                        "resourceType": "MedicationRequest",
                        "id": "med-002",
                        "status": "active",
                        "medicationCodeableConcept": {"coding": [{"display": "Metformin 500mg"}]},
                        "subject": {"reference": "Patient/patient-003"},
                        "dosageInstruction": [{"text": "500mg twice daily"}]
                    }
                },
                {
                    "resource": {
                        "resourceType": "MedicationRequest",
                        "id": "med-003",
                        "status": "active",
                        "medicationCodeableConcept": {"coding": [{"display": "Lisinopril 5mg"}]},
                        "subject": {"reference": "Patient/patient-004"},
                        "dosageInstruction": [{"text": "5mg once daily"}]
                    }
                },
                {
                    "resource": {
                        "resourceType": "MedicationRequest",
                        "id": "med-004",
                        "status": "active",
                        "medicationCodeableConcept": {"coding": [{"display": "Aspirin 81mg"}]},
                        "subject": {"reference": "Patient/patient-002"},
                        "dosageInstruction": [{"text": "81mg once daily"}]
                    }
                },
                {
                    "resource": {
                        "resourceType": "MedicationRequest",
                        "id": "med-005",
                        "status": "active",
                        "medicationCodeableConcept": {"coding": [{"display": "Metformin 500mg"}]},
                        "subject": {"reference": "Patient/patient-005"},
                        "dosageInstruction": [{"text": "500mg twice daily"}]
                    }
                }
            ]
        }
        
        # Create observations data
        observations_data = {
            "resourceType": "Bundle",
            "id": "observations-bundle",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Observation",
                        "id": "obs-001",
                        "status": "final",
                        "code": {"coding": [{"display": "Systolic blood pressure"}]},
                        "subject": {"reference": "Patient/patient-001"},
                        "valueQuantity": {"value": 140, "unit": "mmHg"}
                    }
                },
                {
                    "resource": {
                        "resourceType": "Observation",
                        "id": "obs-002",
                        "status": "final",
                        "code": {"coding": [{"display": "Diastolic blood pressure"}]},
                        "subject": {"reference": "Patient/patient-001"},
                        "valueQuantity": {"value": 90, "unit": "mmHg"}
                    }
                },
                {
                    "resource": {
                        "resourceType": "Observation",
                        "id": "obs-003",
                        "status": "final",
                        "code": {"coding": [{"display": "Blood glucose"}]},
                        "subject": {"reference": "Patient/patient-003"},
                        "valueQuantity": {"value": 180, "unit": "mg/dL"}
                    }
                },
                {
                    "resource": {
                        "resourceType": "Observation",
                        "id": "obs-004",
                        "status": "final",
                        "code": {"coding": [{"display": "Body weight"}]},
                        "subject": {"reference": "Patient/patient-002"},
                        "valueQuantity": {"value": 65, "unit": "kg"}
                    }
                },
                {
                    "resource": {
                        "resourceType": "Observation",
                        "id": "obs-005",
                        "status": "final",
                        "code": {"coding": [{"display": "Heart rate"}]},
                        "subject": {"reference": "Patient/patient-004"},
                        "valueQuantity": {"value": 72, "unit": "bpm"}
                    }
                }
            ]
        }
        
        # Create conditions data
        conditions_data = {
            "resourceType": "Bundle",
            "id": "conditions-bundle",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Condition",
                        "id": "cond-001",
                        "clinicalStatus": {"coding": [{"code": "active", "display": "Active"}]},
                        "code": {"coding": [{"display": "Hypertension"}]},
                        "subject": {"reference": "Patient/patient-001"}
                    }
                },
                {
                    "resource": {
                        "resourceType": "Condition",
                        "id": "cond-002",
                        "clinicalStatus": {"coding": [{"code": "active", "display": "Active"}]},
                        "code": {"coding": [{"display": "Type 2 Diabetes"}]},
                        "subject": {"reference": "Patient/patient-003"}
                    }
                },
                {
                    "resource": {
                        "resourceType": "Condition",
                        "id": "cond-003",
                        "clinicalStatus": {"coding": [{"code": "active", "display": "Active"}]},
                        "code": {"coding": [{"display": "Hypertension"}]},
                        "subject": {"reference": "Patient/patient-004"}
                    }
                }
            ]
        }
        
        # Save all data files
        with open(self.data_dir / "patients.json", 'w') as f:
            json.dump(patients_data, f, indent=2)
        
        with open(self.data_dir / "medications.json", 'w') as f:
            json.dump(medications_data, f, indent=2)
            
        with open(self.data_dir / "observations.json", 'w') as f:
            json.dump(observations_data, f, indent=2)
            
        with open(self.data_dir / "conditions.json", 'w') as f:
            json.dump(conditions_data, f, indent=2)
        
        print("✅ Test data created successfully!")
    
    def load_data(self):
        """Load all FHIR data from JSON files."""
        print("📂 Loading FHIR data from files...")
        
        # Load patients
        patients_file = self.data_dir / "patients.json"
        if patients_file.exists():
            with open(patients_file, 'r') as f:
                data = json.load(f)
                for entry in data.get('entry', []):
                    if entry['resource']['resourceType'] == 'Patient':
                        self.patients.append(entry['resource'])
        
        # Load medications
        medications_file = self.data_dir / "medications.json"
        if medications_file.exists():
            with open(medications_file, 'r') as f:
                data = json.load(f)
                for entry in data.get('entry', []):
                    if entry['resource']['resourceType'] == 'MedicationRequest':
                        self.medications.append(entry['resource'])
        
        # Load observations
        observations_file = self.data_dir / "observations.json"
        if observations_file.exists():
            with open(observations_file, 'r') as f:
                data = json.load(f)
                for entry in data.get('entry', []):
                    if entry['resource']['resourceType'] == 'Observation':
                        self.observations.append(entry['resource'])
        
        # Load conditions
        conditions_file = self.data_dir / "conditions.json"
        if conditions_file.exists():
            with open(conditions_file, 'r') as f:
                data = json.load(f)
                for entry in data.get('entry', []):
                    if entry['resource']['resourceType'] == 'Condition':
                        self.conditions.append(entry['resource'])
        
        print(f"✅ Loaded: {len(self.patients)} patients, {len(self.medications)} medications, {len(self.observations)} observations, {len(self.conditions)} conditions")
    
    def analyze_patient_demographics(self):
        """Analyze patient demographics and calculate statistics."""
        print("\\n👥 PATIENT DEMOGRAPHICS ANALYSIS")
        print("=" * 50)
        
        total_patients = len(self.patients)
        print(f"📊 Total Patients: {total_patients}")
        
        if total_patients == 0:
            print("⚠️ No patients found")
            return {}
        
        # Gender distribution
        gender_counts = {}
        for patient in self.patients:
            gender = patient.get('gender', 'unknown')
            gender_counts[gender] = gender_counts.get(gender, 0) + 1
        
        print("\\n🚻 Gender Distribution:")
        for gender, count in gender_counts.items():
            percentage = (count / total_patients) * 100
            print(f"   {gender.title()}: {count} ({percentage:.1f}%)")
        
        # Age analysis
        current_year = datetime.now().year
        ages = []
        
        for patient in self.patients:
            birth_date = patient.get('birthDate')
            if birth_date:
                birth_year = int(birth_date.split('-')[0])
                age = current_year - birth_year
                ages.append(age)
        
        if ages:
            avg_age = statistics.mean(ages)
            median_age = statistics.median(ages)
            min_age = min(ages)
            max_age = max(ages)
            
            print(f"\\n📅 Age Statistics:")
            print(f"   Average Age: {avg_age:.1f} years")
            print(f"   Median Age: {median_age:.1f} years")
            print(f"   Age Range: {min_age} - {max_age} years")
            
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
            
            print(f"\\n👨‍👩‍👧‍👦 Age Group Distribution:")
            for group, count in age_groups.items():
                percentage = (count / len(ages)) * 100
                print(f"   {group}: {count} ({percentage:.1f}%)")
        
        # Marital status
        marital_counts = {}
        for patient in self.patients:
            marital_status = patient.get('maritalStatus', {}).get('coding', [{}])[0].get('display', 'Unknown')
            marital_counts[marital_status] = marital_counts.get(marital_status, 0) + 1
        
        print(f"\\n💑 Marital Status Distribution:")
        for status, count in marital_counts.items():
            percentage = (count / total_patients) * 100
            print(f"   {status}: {count} ({percentage:.1f}%)")
        
        # Geographic distribution
        city_counts = {}
        for patient in self.patients:
            address = patient.get('address', [{}])[0]
            city = address.get('city', 'Unknown')
            city_counts[city] = city_counts.get(city, 0) + 1
        
        print(f"\\n🏙️ Geographic Distribution:")
        for city, count in city_counts.items():
            percentage = (count / total_patients) * 100
            print(f"   {city}: {count} ({percentage:.1f}%)")
        
        return {
            'total_patients': total_patients,
            'average_age': avg_age if ages else 0,
            'gender_distribution': gender_counts,
            'age_groups': age_groups if ages else {},
            'marital_status': marital_counts,
            'geographic': city_counts
        }
    
    def analyze_popular_medications(self):
        """Analyze medication prescriptions and popularity."""
        print("\\n💊 POPULAR MEDICATIONS ANALYSIS")
        print("=" * 50)
        
        total_prescriptions = len(self.medications)
        print(f"📊 Total Prescriptions: {total_prescriptions}")
        
        if total_prescriptions == 0:
            print("⚠️ No medications found")
            return {}
        
        # Count medication frequency
        med_counts = {}
        dosage_patterns = {}
        
        for med in self.medications:
            med_name = med.get('medicationCodeableConcept', {}).get('coding', [{}])[0].get('display', 'Unknown')
            med_counts[med_name] = med_counts.get(med_name, 0) + 1
            
            # Dosage analysis
            dosage = med.get('dosageInstruction', [{}])[0].get('text', 'Unknown')
            dosage_patterns[dosage] = dosage_patterns.get(dosage, 0) + 1
        
        # Sort by popularity
        sorted_meds = sorted(med_counts.items(), key=lambda x: x[1], reverse=True)
        
        print("\\n🏆 Most Popular Medications:")
        for i, (med_name, count) in enumerate(sorted_meds, 1):
            percentage = (count / total_prescriptions) * 100
            print(f"   {i}. {med_name}: {count} prescriptions ({percentage:.1f}%)")
        
        print("\\n💉 Dosage Patterns:")
        for dosage, count in dosage_patterns.items():
            percentage = (count / total_prescriptions) * 100
            print(f"   {dosage}: {count} ({percentage:.1f}%)")
        
        # Patients on multiple medications
        patient_med_counts = {}
        for med in self.medications:
            patient_ref = med.get('subject', {}).get('reference', '')
            patient_med_counts[patient_ref] = patient_med_counts.get(patient_ref, 0) + 1
        
        if patient_med_counts:
            avg_meds_per_patient = statistics.mean(patient_med_counts.values())
            max_meds = max(patient_med_counts.values())
            print(f"\\n👤 Medication Usage Per Patient:")
            print(f"   Average medications per patient: {avg_meds_per_patient:.1f}")
            print(f"   Maximum medications for one patient: {max_meds}")
        
        return {
            'total_prescriptions': total_prescriptions,
            'popular_medications': dict(sorted_meds),
            'dosage_patterns': dosage_patterns,
            'avg_meds_per_patient': avg_meds_per_patient if patient_med_counts else 0
        }
    
    def analyze_clinical_observations(self):
        """Analyze clinical observations and vital signs."""
        print("\\n🩺 CLINICAL OBSERVATIONS ANALYSIS")
        print("=" * 50)
        
        total_observations = len(self.observations)
        print(f"📊 Total Observations: {total_observations}")
        
        if total_observations == 0:
            print("⚠️ No observations found")
            return {}
        
        # Group observations by type
        obs_types = {}
        vital_signs = {}
        
        for obs in self.observations:
            obs_type = obs.get('code', {}).get('coding', [{}])[0].get('display', 'Unknown')
            obs_types[obs_type] = obs_types.get(obs_type, 0) + 1
            
            # Collect values for vital signs
            if 'valueQuantity' in obs:
                value = obs['valueQuantity']['value']
                unit = obs['valueQuantity'].get('unit', '')
                
                if obs_type not in vital_signs:
                    vital_signs[obs_type] = []
                vital_signs[obs_type].append({'value': value, 'unit': unit})
        
        print("\\n🔬 Observation Types:")
        for obs_type, count in obs_types.items():
            percentage = (count / total_observations) * 100
            print(f"   {obs_type}: {count} ({percentage:.1f}%)")
        
        print("\\n📈 Vital Signs Averages:")
        abnormal_alerts = []
        
        for obs_type, values in vital_signs.items():
            if values:
                avg_value = statistics.mean([v['value'] for v in values])
                unit = values[0]['unit']
                print(f"   {obs_type}: {avg_value:.1f} {unit} (from {len(values)} readings)")
                
                # Check for abnormal values
                for v in values:
                    value = v['value']
                    if obs_type == "Systolic blood pressure" and (value > 140 or value < 90):
                        abnormal_alerts.append(f"High/Low systolic BP: {value} {unit}")
                    elif obs_type == "Diastolic blood pressure" and (value > 90 or value < 60):
                        abnormal_alerts.append(f"High/Low diastolic BP: {value} {unit}")
                    elif obs_type == "Blood glucose" and (value > 125 or value < 70):
                        abnormal_alerts.append(f"High/Low glucose: {value} {unit}")
        
        if abnormal_alerts:
            print("\\n⚠️ Abnormal Value Alerts:")
            for alert in abnormal_alerts:
                print(f"   🚨 {alert}")
        else:
            print("\\n✅ No abnormal values detected")
        
        return {
            'total_observations': total_observations,
            'observation_types': obs_types,
            'vital_signs_averages': {k: statistics.mean([v['value'] for v in vs]) for k, vs in vital_signs.items()},
            'abnormal_alerts': abnormal_alerts
        }
    
    def analyze_medical_conditions(self):
        """Analyze medical conditions and diagnoses."""
        print("\\n🏥 MEDICAL CONDITIONS ANALYSIS")
        print("=" * 50)
        
        total_conditions = len(self.conditions)
        print(f"📊 Total Conditions: {total_conditions}")
        
        if total_conditions == 0:
            print("⚠️ No conditions found")
            return {}
        
        # Count condition frequency
        condition_counts = {}
        status_counts = {}
        
        for condition in self.conditions:
            condition_name = condition.get('code', {}).get('coding', [{}])[0].get('display', 'Unknown')
            condition_counts[condition_name] = condition_counts.get(condition_name, 0) + 1
            
            status = condition.get('clinicalStatus', {}).get('coding', [{}])[0].get('display', 'Unknown')
            status_counts[status] = status_counts.get(status, 0) + 1
        
        print("\\n🔍 Most Common Conditions:")
        sorted_conditions = sorted(condition_counts.items(), key=lambda x: x[1], reverse=True)
        for i, (condition, count) in enumerate(sorted_conditions, 1):
            percentage = (count / total_conditions) * 100
            print(f"   {i}. {condition}: {count} cases ({percentage:.1f}%)")
        
        print("\\n📋 Condition Status:")
        for status, count in status_counts.items():
            percentage = (count / total_conditions) * 100
            print(f"   {status}: {count} ({percentage:.1f}%)")
        
        # Patient impact
        patient_condition_counts = {}
        for condition in self.conditions:
            patient_ref = condition.get('subject', {}).get('reference', '')
            patient_condition_counts[patient_ref] = patient_condition_counts.get(patient_ref, 0) + 1
        
        if patient_condition_counts:
            patients_with_conditions = len(patient_condition_counts)
            avg_conditions_per_patient = statistics.mean(patient_condition_counts.values())
            
            print(f"\\n👥 Patient Impact:")
            print(f"   Patients with conditions: {patients_with_conditions}")
            print(f"   Average conditions per patient: {avg_conditions_per_patient:.1f}")
        
        return {
            'total_conditions': total_conditions,
            'common_conditions': dict(sorted_conditions),
            'condition_status': status_counts,
            'patients_with_conditions': len(patient_condition_counts) if patient_condition_counts else 0
        }
    
    def run_comprehensive_analysis(self):
        """Run complete healthcare analytics."""
        print("\\n" + "=" * 80)
        print("🏥 COMPREHENSIVE HEALTHCARE DATA ANALYTICS")
        print("=" * 80)
        
        # Create and load data
        self.create_comprehensive_test_data()
        self.load_data()
        
        # Run all analyses
        patient_analysis = self.analyze_patient_demographics()
        medication_analysis = self.analyze_popular_medications()
        observation_analysis = self.analyze_clinical_observations()
        condition_analysis = self.analyze_medical_conditions()
        
        # Summary
        print("\\n" + "=" * 80)
        print("📋 HEALTHCARE ANALYTICS SUMMARY")
        print("=" * 80)
        
        print(f"👥 Total Patients: {patient_analysis.get('total_patients', 0)}")
        print(f"📅 Average Patient Age: {patient_analysis.get('average_age', 0):.1f} years")
        
        if patient_analysis.get('gender_distribution'):
            most_common_gender = max(patient_analysis['gender_distribution'].items(), key=lambda x: x[1])
            print(f"🚻 Most Common Gender: {most_common_gender[0].title()} ({most_common_gender[1]} patients)")
        
        print(f"💊 Total Prescriptions: {medication_analysis.get('total_prescriptions', 0)}")
        
        if medication_analysis.get('popular_medications'):
            most_prescribed = list(medication_analysis['popular_medications'].items())[0]
            print(f"🏆 Most Prescribed Medication: {most_prescribed[0]} ({most_prescribed[1]} prescriptions)")
        
        print(f"🩺 Total Clinical Observations: {observation_analysis.get('total_observations', 0)}")
        print(f"🏥 Total Medical Conditions: {condition_analysis.get('total_conditions', 0)}")
        
        if condition_analysis.get('common_conditions'):
            most_common_condition = list(condition_analysis['common_conditions'].items())[0]
            print(f"📋 Most Common Condition: {most_common_condition[0]} ({most_common_condition[1]} cases)")
        
        # Key insights
        print("\\n🔍 KEY INSIGHTS:")
        print(f"   • Average medications per patient: {medication_analysis.get('avg_meds_per_patient', 0):.1f}")
        print(f"   • Patients with medical conditions: {condition_analysis.get('patients_with_conditions', 0)}")
        print(f"   • Abnormal vital signs detected: {len(observation_analysis.get('abnormal_alerts', []))}")
        
        return {
            'patients': patient_analysis,
            'medications': medication_analysis,
            'observations': observation_analysis,
            'conditions': condition_analysis
        }


def main():
    """Run direct FHIR analytics test."""
    print("🚀 STARTING DIRECT FHIR HEALTHCARE ANALYTICS")
    print("=" * 50)
    
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    data_dir = temp_dir / "fhir_data"
    
    # Initialize analyzer
    analyzer = DirectFHIRAnalyzer(str(data_dir))
    
    # Run comprehensive analysis
    results = analyzer.run_comprehensive_analysis()
    
    print("\\n✅ HEALTHCARE ANALYTICS COMPLETE!")
    return results


if __name__ == "__main__":
    results = main()