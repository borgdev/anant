"""
ML Service Sub-Application
=========================

Machine Learning services with Ray for distributed training and inference.
"""

import ray
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import asyncio
import numpy as np
import json
from datetime import datetime
import uuid


ml_app = FastAPI(
    title="ML Service",
    description="Machine Learning services with Ray scaling"
)


class TrainingRequest(BaseModel):
    model_type: str
    dataset_size: int = 1000
    features: int = 10
    hyperparameters: Dict[str, Any] = {}
    validation_split: float = 0.2


class PredictionRequest(BaseModel):
    model_id: str
    features: List[List[float]]
    batch_size: Optional[int] = 32


class TrainingResponse(BaseModel):
    job_id: str
    status: str
    model_id: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None
    training_time: Optional[float] = None


class PredictionResponse(BaseModel):
    predictions: List[float]
    model_id: str
    inference_time: float
    batch_size: int


# In-memory storage for demo (use Redis/database in production)
training_jobs = {}
models = {}


@ray.remote
class DistributedTrainer:
    """Distributed model trainer using Ray."""
    
    def __init__(self):
        self.model_params = {}
    
    def train_model(self, model_type: str, X: np.ndarray, y: np.ndarray, 
                   hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        """Train a model with given data."""
        import time
        start_time = time.time()
        
        if model_type == "linear_regression":
            # Simple linear regression
            # Add bias term
            X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
            
            # Normal equation: theta = (X^T X)^-1 X^T y
            try:
                theta = np.linalg.inv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
                mse = np.mean((X_with_bias @ theta - y) ** 2)
                r2 = 1 - (np.sum((y - X_with_bias @ theta) ** 2) / 
                         np.sum((y - np.mean(y)) ** 2))
                
                model_params = {
                    "type": "linear_regression",
                    "weights": theta.tolist(),
                    "mse": float(mse),
                    "r2": float(r2)
                }
            except np.linalg.LinAlgError:
                # Fallback to gradient descent if normal equation fails
                model_params = self._gradient_descent(X_with_bias, y, hyperparameters)
                
        elif model_type == "logistic_regression":
            model_params = self._train_logistic_regression(X, y, hyperparameters)
            
        elif model_type == "neural_network":
            model_params = self._train_neural_network(X, y, hyperparameters)
            
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        training_time = time.time() - start_time
        model_params["training_time"] = training_time
        model_params["hyperparameters"] = hyperparameters
        
        return model_params
    
    def _gradient_descent(self, X: np.ndarray, y: np.ndarray, 
                         hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        """Simple gradient descent for linear regression."""
        learning_rate = hyperparameters.get("learning_rate", 0.01)
        epochs = hyperparameters.get("epochs", 100)
        
        # Initialize weights
        theta = np.random.normal(0, 0.1, X.shape[1])
        
        for epoch in range(epochs):
            predictions = X @ theta
            errors = predictions - y
            gradient = X.T @ errors / len(y)
            theta -= learning_rate * gradient
        
        # Calculate final metrics
        final_predictions = X @ theta
        mse = np.mean((final_predictions - y) ** 2)
        r2 = 1 - (np.sum((y - final_predictions) ** 2) / 
                 np.sum((y - np.mean(y)) ** 2))
        
        return {
            "type": "linear_regression",
            "weights": theta.tolist(),
            "mse": float(mse),
            "r2": float(r2),
            "method": "gradient_descent"
        }
    
    def _train_logistic_regression(self, X: np.ndarray, y: np.ndarray,
                                 hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        """Train logistic regression model."""
        learning_rate = hyperparameters.get("learning_rate", 0.01)
        epochs = hyperparameters.get("epochs", 100)
        
        # Add bias term
        X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
        theta = np.random.normal(0, 0.1, X_with_bias.shape[1])
        
        for epoch in range(epochs):
            z = X_with_bias @ theta
            predictions = 1 / (1 + np.exp(-np.clip(z, -250, 250)))  # Sigmoid with clipping
            gradient = X_with_bias.T @ (predictions - y) / len(y)
            theta -= learning_rate * gradient
        
        # Calculate accuracy
        final_predictions = 1 / (1 + np.exp(-np.clip(X_with_bias @ theta, -250, 250)))
        accuracy = np.mean((final_predictions > 0.5) == y)
        
        return {
            "type": "logistic_regression",
            "weights": theta.tolist(),
            "accuracy": float(accuracy)
        }
    
    def _train_neural_network(self, X: np.ndarray, y: np.ndarray,
                            hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        """Train a simple neural network."""
        hidden_size = hyperparameters.get("hidden_size", 64)
        learning_rate = hyperparameters.get("learning_rate", 0.001)
        epochs = hyperparameters.get("epochs", 50)
        
        # Simple 2-layer neural network
        input_size = X.shape[1]
        output_size = 1
        
        # Initialize weights
        W1 = np.random.normal(0, 0.1, (input_size, hidden_size))
        b1 = np.zeros((1, hidden_size))
        W2 = np.random.normal(0, 0.1, (hidden_size, output_size))
        b2 = np.zeros((1, output_size))
        
        for epoch in range(epochs):
            # Forward pass
            z1 = X @ W1 + b1
            a1 = np.maximum(0, z1)  # ReLU activation
            z2 = a1 @ W2 + b2
            predictions = z2.flatten()
            
            # Loss (MSE)
            loss = np.mean((predictions - y) ** 2)
            
            # Backward pass
            d_loss = 2 * (predictions - y) / len(y)
            
            d_W2 = a1.T @ d_loss.reshape(-1, 1)
            d_b2 = np.sum(d_loss)
            
            d_a1 = d_loss.reshape(-1, 1) @ W2.T
            d_z1 = d_a1 * (z1 > 0)  # ReLU derivative
            
            d_W1 = X.T @ d_z1
            d_b1 = np.sum(d_z1, axis=0)
            
            # Update weights
            W1 -= learning_rate * d_W1
            b1 -= learning_rate * d_b1.reshape(1, -1)
            W2 -= learning_rate * d_W2
            b2 -= learning_rate * d_b2
        
        # Final evaluation
        z1 = X @ W1 + b1
        a1 = np.maximum(0, z1)
        z2 = a1 @ W2 + b2
        final_predictions = z2.flatten()
        mse = np.mean((final_predictions - y) ** 2)
        
        return {
            "type": "neural_network",
            "weights": {
                "W1": W1.tolist(),
                "b1": b1.tolist(),
                "W2": W2.tolist(),
                "b2": b2.tolist()
            },
            "mse": float(mse),
            "hidden_size": hidden_size
        }


@ray.remote
def generate_dataset(size: int, features: int, model_type: str) -> tuple:
    """Generate synthetic dataset for training."""
    np.random.seed(42)  # For reproducibility
    
    X = np.random.randn(size, features)
    
    if model_type == "linear_regression" or model_type == "neural_network":
        # Generate linear relationship with noise
        true_weights = np.random.randn(features)
        y = X @ true_weights + np.random.normal(0, 0.1, size)
    elif model_type == "logistic_regression":
        # Generate binary classification data
        true_weights = np.random.randn(features)
        logits = X @ true_weights
        probabilities = 1 / (1 + np.exp(-logits))
        y = np.random.binomial(1, probabilities)
    else:
        y = np.random.randn(size)
    
    return X, y


@ray.remote
class ModelPredictor:
    """Distributed model predictor."""
    
    def __init__(self):
        pass
    
    def predict(self, model_params: Dict[str, Any], X: np.ndarray) -> np.ndarray:
        """Make predictions using trained model."""
        model_type = model_params["type"]
        
        if model_type == "linear_regression":
            weights = np.array(model_params["weights"])
            X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
            return X_with_bias @ weights
            
        elif model_type == "logistic_regression":
            weights = np.array(model_params["weights"])
            X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
            logits = X_with_bias @ weights
            return 1 / (1 + np.exp(-np.clip(logits, -250, 250)))
            
        elif model_type == "neural_network":
            weights = model_params["weights"]
            W1 = np.array(weights["W1"])
            b1 = np.array(weights["b1"])
            W2 = np.array(weights["W2"])
            b2 = np.array(weights["b2"])
            
            z1 = X @ W1 + b1
            a1 = np.maximum(0, z1)
            z2 = a1 @ W2 + b2
            return z2.flatten()
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")


@ml_app.get("/")
async def ml_root():
    """ML service root."""
    return {
        "service": "ML Service",
        "description": "Distributed machine learning with Ray",
        "endpoints": [
            "/train - Train models",
            "/predict - Make predictions",
            "/models - List trained models",
            "/jobs - List training jobs"
        ]
    }


@ml_app.post("/train", response_model=TrainingResponse)
async def train_model(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Train a machine learning model using Ray distributed computing."""
    if not ray.is_initialized():
        raise HTTPException(status_code=503, detail="Ray cluster not available")
    
    job_id = str(uuid.uuid4())
    
    # Initialize job status
    training_jobs[job_id] = {
        "status": "started",
        "model_type": request.model_type,
        "created_at": datetime.now().isoformat()
    }
    
    async def train_async():
        """Async training function."""
        try:
            # Generate dataset
            X_ref, y_ref = generate_dataset.remote(
                request.dataset_size, 
                request.features, 
                request.model_type
            )
            
            X, y = await asyncio.create_task(
                asyncio.to_thread(ray.get, [X_ref, y_ref])
            )
            
            # Split data
            split_idx = int(len(X) * (1 - request.validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Train model
            trainer = DistributedTrainer.remote()
            model_params_ref = trainer.train_model.remote(
                request.model_type, X_train, y_train, request.hyperparameters
            )
            
            model_params = await asyncio.create_task(
                asyncio.to_thread(ray.get, model_params_ref)
            )
            
            # Validate model
            predictor = ModelPredictor.remote()
            val_predictions_ref = predictor.predict.remote(model_params, X_val)
            val_predictions = await asyncio.create_task(
                asyncio.to_thread(ray.get, val_predictions_ref)
            )
            
            # Calculate validation metrics
            if request.model_type in ["linear_regression", "neural_network"]:
                val_mse = np.mean((val_predictions - y_val) ** 2)
                val_r2 = 1 - (np.sum((y_val - val_predictions) ** 2) / 
                             np.sum((y_val - np.mean(y_val)) ** 2))
                validation_metrics = {"mse": float(val_mse), "r2": float(val_r2)}
            else:  # logistic_regression
                val_accuracy = np.mean((val_predictions > 0.5) == y_val)
                validation_metrics = {"accuracy": float(val_accuracy)}
            
            # Save model
            model_id = f"model_{job_id[:8]}"
            models[model_id] = {
                "params": model_params,
                "validation_metrics": validation_metrics,
                "created_at": datetime.now().isoformat(),
                "model_type": request.model_type
            }
            
            # Update job status
            training_jobs[job_id].update({
                "status": "completed",
                "model_id": model_id,
                "metrics": validation_metrics,
                "training_time": model_params.get("training_time"),
                "completed_at": datetime.now().isoformat()
            })
            
        except Exception as e:
            training_jobs[job_id].update({
                "status": "failed",
                "error": str(e),
                "failed_at": datetime.now().isoformat()
            })
    
    # Start training in background
    background_tasks.add_task(train_async)
    
    return TrainingResponse(
        job_id=job_id,
        status="started"
    )


@ml_app.get("/jobs/{job_id}")
async def get_training_job(job_id: str):
    """Get training job status."""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return training_jobs[job_id]


@ml_app.get("/jobs")
async def list_training_jobs():
    """List all training jobs."""
    return {"jobs": training_jobs}


@ml_app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make predictions using a trained model."""
    if not ray.is_initialized():
        raise HTTPException(status_code=503, detail="Ray cluster not available")
    
    if request.model_id not in models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model_info = models[request.model_id]
    model_params = model_info["params"]
    
    import time
    start_time = time.time()
    
    # Convert input to numpy array
    X = np.array(request.features)
    
    # Make predictions
    predictor = ModelPredictor.remote()
    predictions_ref = predictor.predict.remote(model_params, X)
    predictions = await asyncio.create_task(
        asyncio.to_thread(ray.get, predictions_ref)
    )
    
    inference_time = time.time() - start_time
    
    return PredictionResponse(
        predictions=predictions.tolist(),
        model_id=request.model_id,
        inference_time=inference_time,
        batch_size=len(request.features)
    )


@ml_app.get("/models")
async def list_models():
    """List all trained models."""
    return {"models": models}


@ml_app.get("/models/{model_id}")
async def get_model(model_id: str):
    """Get model details."""
    if model_id not in models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return models[model_id]