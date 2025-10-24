"""
Data Processing Sub-Application
==============================

Data processing, ETL operations, and data pipelines using Ray for
distributed data processing.
"""

import ray
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union
import asyncio
import numpy as np
import json
from datetime import datetime
import uuid
import io


data_app = FastAPI(
    title="Data Processing Service",
    description="Distributed data processing and ETL with Ray"
)


class DataProcessingRequest(BaseModel):
    operation: str
    data: Union[List[Dict[str, Any]], List[List[float]]]
    parameters: Optional[Dict[str, Any]] = {}
    chunk_size: Optional[int] = 1000


class BatchProcessingRequest(BaseModel):
    batch_id: str
    operations: List[str]
    data_source: str
    output_format: str = "json"


class DataProcessingResponse(BaseModel):
    job_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    processing_time: Optional[float] = None
    records_processed: Optional[int] = None


class PipelineRequest(BaseModel):
    pipeline_name: str
    stages: List[Dict[str, Any]]
    input_data: List[Dict[str, Any]]
    parallel: bool = True


# In-memory storage for demo
processing_jobs = {}
pipelines = {}


@ray.remote
def process_chunk(chunk: List[Dict[str, Any]], operation: str, 
                 parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Process a chunk of data with the specified operation."""
    import time
    start_time = time.time()
    
    if operation == "filter":
        field = parameters.get("field", "value")
        min_val = parameters.get("min", 0)
        max_val = parameters.get("max", float('inf'))
        
        filtered = [
            record for record in chunk 
            if min_val <= record.get(field, 0) <= max_val
        ]
        result = {"filtered_records": filtered, "count": len(filtered)}
        
    elif operation == "aggregate":
        field = parameters.get("field", "value")
        agg_type = parameters.get("type", "sum")
        
        values = [record.get(field, 0) for record in chunk if field in record]
        
        if agg_type == "sum":
            result_value = sum(values)
        elif agg_type == "mean":
            result_value = sum(values) / len(values) if values else 0
        elif agg_type == "max":
            result_value = max(values) if values else 0
        elif agg_type == "min":
            result_value = min(values) if values else 0
        elif agg_type == "count":
            result_value = len(values)
        else:
            result_value = 0
            
        result = {
            "aggregation": {
                "field": field,
                "type": agg_type,
                "value": result_value,
                "count": len(values)
            }
        }
        
    elif operation == "transform":
        transform_type = parameters.get("type", "multiply")
        field = parameters.get("field", "value")
        factor = parameters.get("factor", 1)
        
        transformed = []
        for record in chunk:
            new_record = record.copy()
            if field in new_record:
                if transform_type == "multiply":
                    new_record[field] = new_record[field] * factor
                elif transform_type == "add":
                    new_record[field] = new_record[field] + factor
                elif transform_type == "normalize":
                    # Simple min-max normalization within chunk
                    values = [r.get(field, 0) for r in chunk if field in r]
                    if values:
                        min_val, max_val = min(values), max(values)
                        if max_val > min_val:
                            new_record[field] = (new_record[field] - min_val) / (max_val - min_val)
            transformed.append(new_record)
            
        result = {"transformed_records": transformed, "count": len(transformed)}
        
    elif operation == "validate":
        required_fields = parameters.get("required_fields", [])
        valid_records = []
        invalid_records = []
        
        for record in chunk:
            if all(field in record for field in required_fields):
                valid_records.append(record)
            else:
                invalid_records.append(record)
                
        result = {
            "valid_records": valid_records,
            "invalid_records": invalid_records,
            "valid_count": len(valid_records),
            "invalid_count": len(invalid_records)
        }
        
    elif operation == "deduplicate":
        key_field = parameters.get("key_field", "id")
        seen = set()
        unique_records = []
        
        for record in chunk:
            key = record.get(key_field)
            if key not in seen:
                seen.add(key)
                unique_records.append(record)
                
        result = {
            "unique_records": unique_records,
            "original_count": len(chunk),
            "unique_count": len(unique_records),
            "duplicates_removed": len(chunk) - len(unique_records)
        }
        
    else:
        result = {"error": f"Unknown operation: {operation}"}
    
    processing_time = time.time() - start_time
    result["processing_time"] = processing_time
    result["chunk_size"] = len(chunk)
    
    return result


@ray.remote
def generate_sample_data(size: int, data_type: str) -> List[Dict[str, Any]]:
    """Generate sample data for testing."""
    import random
    
    data = []
    for i in range(size):
        if data_type == "sales":
            record = {
                "id": i,
                "product_id": random.randint(1, 100),
                "quantity": random.randint(1, 10),
                "price": round(random.uniform(10, 1000), 2),
                "customer_id": random.randint(1, 1000),
                "timestamp": datetime.now().isoformat(),
                "region": random.choice(["North", "South", "East", "West"])
            }
        elif data_type == "users":
            record = {
                "id": i,
                "name": f"User_{i}",
                "age": random.randint(18, 80),
                "email": f"user_{i}@example.com",
                "score": random.randint(0, 100),
                "active": random.choice([True, False]),
                "join_date": datetime.now().isoformat()
            }
        elif data_type == "events":
            record = {
                "id": i,
                "event_type": random.choice(["click", "view", "purchase", "signup"]),
                "user_id": random.randint(1, 1000),
                "value": random.uniform(0, 100),
                "timestamp": datetime.now().isoformat(),
                "metadata": {"source": random.choice(["web", "mobile", "api"])}
            }
        else:
            record = {
                "id": i,
                "value": random.uniform(0, 100),
                "category": random.choice(["A", "B", "C"]),
                "timestamp": datetime.now().isoformat()
            }
        data.append(record)
    
    return data


@ray.remote
class DataPipelineProcessor:
    """Distributed data pipeline processor."""
    
    def __init__(self):
        self.stage_functions = {
            "filter": self._filter_stage,
            "transform": self._transform_stage,
            "aggregate": self._aggregate_stage,
            "validate": self._validate_stage,
            "enrich": self._enrich_stage
        }
    
    def process_pipeline(self, data: List[Dict[str, Any]], 
                        stages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process data through multiple pipeline stages."""
        import time
        start_time = time.time()
        
        current_data = data
        stage_results = []
        
        for i, stage in enumerate(stages):
            stage_type = stage.get("type")
            stage_params = stage.get("parameters", {})
            
            if stage_type in self.stage_functions:
                stage_result = self.stage_functions[stage_type](current_data, stage_params)
                current_data = stage_result.get("output_data", current_data)
                stage_results.append({
                    "stage": i + 1,
                    "type": stage_type,
                    "result": stage_result
                })
            else:
                stage_results.append({
                    "stage": i + 1,
                    "type": stage_type,
                    "error": f"Unknown stage type: {stage_type}"
                })
        
        processing_time = time.time() - start_time
        
        return {
            "final_data": current_data,
            "stage_results": stage_results,
            "processing_time": processing_time,
            "input_count": len(data),
            "output_count": len(current_data)
        }
    
    def _filter_stage(self, data: List[Dict[str, Any]], 
                     params: Dict[str, Any]) -> Dict[str, Any]:
        """Filter stage implementation."""
        field = params.get("field", "value")
        condition = params.get("condition", "greater_than")
        value = params.get("value", 0)
        
        filtered_data = []
        for record in data:
            record_value = record.get(field, 0)
            
            if condition == "greater_than" and record_value > value:
                filtered_data.append(record)
            elif condition == "less_than" and record_value < value:
                filtered_data.append(record)
            elif condition == "equals" and record_value == value:
                filtered_data.append(record)
            elif condition == "not_equals" and record_value != value:
                filtered_data.append(record)
        
        return {
            "output_data": filtered_data,
            "filtered_count": len(filtered_data),
            "removed_count": len(data) - len(filtered_data)
        }
    
    def _transform_stage(self, data: List[Dict[str, Any]], 
                        params: Dict[str, Any]) -> Dict[str, Any]:
        """Transform stage implementation."""
        field = params.get("field", "value")
        operation = params.get("operation", "multiply")
        factor = params.get("factor", 1)
        
        transformed_data = []
        for record in data:
            new_record = record.copy()
            if field in new_record:
                if operation == "multiply":
                    new_record[field] = new_record[field] * factor
                elif operation == "add":
                    new_record[field] = new_record[field] + factor
                elif operation == "square":
                    new_record[field] = new_record[field] ** 2
            transformed_data.append(new_record)
        
        return {
            "output_data": transformed_data,
            "transformed_count": len(transformed_data)
        }
    
    def _aggregate_stage(self, data: List[Dict[str, Any]], 
                        params: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate stage implementation."""
        group_by = params.get("group_by", "category")
        agg_field = params.get("field", "value")
        agg_type = params.get("type", "sum")
        
        groups = {}
        for record in data:
            group_key = record.get(group_by, "unknown")
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(record.get(agg_field, 0))
        
        aggregated_data = []
        for group_key, values in groups.items():
            if agg_type == "sum":
                agg_value = sum(values)
            elif agg_type == "mean":
                agg_value = sum(values) / len(values) if values else 0
            elif agg_type == "count":
                agg_value = len(values)
            else:
                agg_value = 0
            
            aggregated_data.append({
                group_by: group_key,
                f"{agg_type}_{agg_field}": agg_value,
                "count": len(values)
            })
        
        return {
            "output_data": aggregated_data,
            "groups_created": len(groups)
        }
    
    def _validate_stage(self, data: List[Dict[str, Any]], 
                       params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate stage implementation."""
        required_fields = params.get("required_fields", [])
        valid_data = []
        invalid_data = []
        
        for record in data:
            if all(field in record for field in required_fields):
                valid_data.append(record)
            else:
                invalid_data.append(record)
        
        return {
            "output_data": valid_data,
            "valid_count": len(valid_data),
            "invalid_count": len(invalid_data),
            "invalid_data": invalid_data
        }
    
    def _enrich_stage(self, data: List[Dict[str, Any]], 
                     params: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich stage implementation."""
        enrichment_type = params.get("type", "timestamp")
        
        enriched_data = []
        for record in data:
            new_record = record.copy()
            
            if enrichment_type == "timestamp":
                new_record["processed_at"] = datetime.now().isoformat()
            elif enrichment_type == "id":
                new_record["processing_id"] = str(uuid.uuid4())
            elif enrichment_type == "hash":
                # Simple hash of record content
                content = str(sorted(record.items()))
                new_record["content_hash"] = hash(content)
            
            enriched_data.append(new_record)
        
        return {
            "output_data": enriched_data,
            "enriched_count": len(enriched_data)
        }


@data_app.get("/")
async def data_root():
    """Data processing service root."""
    return {
        "service": "Data Processing",
        "description": "Distributed data processing and ETL with Ray",
        "endpoints": [
            "/process - Process data with operations",
            "/batch - Batch processing jobs",
            "/pipeline - Data pipeline processing",
            "/generate - Generate sample data"
        ]
    }


@data_app.post("/process", response_model=DataProcessingResponse)
async def process_data(request: DataProcessingRequest):
    """Process data using distributed Ray workers."""
    if not ray.is_initialized():
        raise HTTPException(status_code=503, detail="Ray cluster not available")
    
    job_id = str(uuid.uuid4())
    start_time = datetime.now()
    
    try:
        # Split data into chunks for parallel processing
        data = request.data
        chunk_size = request.chunk_size or 1000
        chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        
        # Process chunks in parallel
        chunk_tasks = []
        for chunk in chunks:
            task = process_chunk.remote(chunk, request.operation, request.parameters)
            chunk_tasks.append(task)
        
        # Wait for all chunks to complete
        chunk_results = await asyncio.create_task(
            asyncio.to_thread(ray.get, chunk_tasks)
        )
        
        # Combine results
        if request.operation == "aggregate":
            # Combine aggregation results
            total_value = sum(result["aggregation"]["value"] for result in chunk_results)
            total_count = sum(result["aggregation"]["count"] for result in chunk_results)
            
            final_result = {
                "aggregation": {
                    "field": request.parameters.get("field", "value"),
                    "type": request.parameters.get("type", "sum"),
                    "value": total_value,
                    "total_count": total_count
                }
            }
        elif request.operation in ["filter", "transform", "validate", "deduplicate"]:
            # Combine list results
            all_records = []
            for result in chunk_results:
                key = "filtered_records" if request.operation == "filter" else \
                      "transformed_records" if request.operation == "transform" else \
                      "valid_records" if request.operation == "validate" else \
                      "unique_records"
                
                if key in result:
                    all_records.extend(result[key])
            
            final_result = {
                "records": all_records,
                "total_count": len(all_records),
                "chunks_processed": len(chunk_results)
            }
        else:
            final_result = {"chunk_results": chunk_results}
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        return DataProcessingResponse(
            job_id=job_id,
            status="completed",
            result=final_result,
            processing_time=processing_time,
            records_processed=len(data)
        )
        
    except Exception as e:
        return DataProcessingResponse(
            job_id=job_id,
            status="failed",
            result={"error": str(e)}
        )


@data_app.post("/pipeline")
async def process_pipeline(request: PipelineRequest):
    """Process data through a multi-stage pipeline."""
    if not ray.is_initialized():
        raise HTTPException(status_code=503, detail="Ray cluster not available")
    
    pipeline_id = str(uuid.uuid4())
    
    try:
        if request.parallel:
            # Process pipeline stages in parallel where possible
            processor = DataPipelineProcessor.remote()
            result_ref = processor.process_pipeline.remote(
                request.input_data, request.stages
            )
            result = await asyncio.create_task(
                asyncio.to_thread(ray.get, result_ref)
            )
        else:
            # Sequential processing
            current_data = request.input_data
            stage_results = []
            
            for i, stage in enumerate(request.stages):
                stage_type = stage.get("type")
                stage_params = stage.get("parameters", {})
                
                # Process stage sequentially
                chunk_result_ref = process_chunk.remote(
                    current_data, stage_type, stage_params
                )
                chunk_result = await asyncio.create_task(
                    asyncio.to_thread(ray.get, chunk_result_ref)
                )
                
                # Update data for next stage
                if "transformed_records" in chunk_result:
                    current_data = chunk_result["transformed_records"]
                elif "filtered_records" in chunk_result:
                    current_data = chunk_result["filtered_records"]
                elif "valid_records" in chunk_result:
                    current_data = chunk_result["valid_records"]
                
                stage_results.append({
                    "stage": i + 1,
                    "type": stage_type,
                    "result": chunk_result
                })
            
            result = {
                "final_data": current_data,
                "stage_results": stage_results,
                "input_count": len(request.input_data),
                "output_count": len(current_data)
            }
        
        # Store pipeline for future reference
        pipelines[pipeline_id] = {
            "name": request.pipeline_name,
            "stages": request.stages,
            "result": result,
            "created_at": datetime.now().isoformat()
        }
        
        return {
            "pipeline_id": pipeline_id,
            "status": "completed",
            "result": result
        }
        
    except Exception as e:
        return {
            "pipeline_id": pipeline_id,
            "status": "failed",
            "error": str(e)
        }


@data_app.get("/generate/{data_type}")
async def generate_sample_data_endpoint(data_type: str, size: int = 1000):
    """Generate sample data for testing."""
    if not ray.is_initialized():
        raise HTTPException(status_code=503, detail="Ray cluster not available")
    
    try:
        data_ref = generate_sample_data.remote(size, data_type)
        data = await asyncio.create_task(
            asyncio.to_thread(ray.get, data_ref)
        )
        
        return {
            "data_type": data_type,
            "size": size,
            "sample": data[:5],  # Return first 5 records as sample
            "full_data": data
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data generation failed: {str(e)}")


@data_app.get("/pipelines")
async def list_pipelines():
    """List all stored pipelines."""
    return {"pipelines": pipelines}


@data_app.get("/pipelines/{pipeline_id}")
async def get_pipeline(pipeline_id: str):
    """Get pipeline details."""
    if pipeline_id not in pipelines:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    
    return pipelines[pipeline_id]