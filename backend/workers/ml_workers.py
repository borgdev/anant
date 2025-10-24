"""
Machine Learning Workers
=======================

Specialized ML workers for distributed training, inference, and model management.
"""

import ray
import numpy as np
import asyncio
from typing import List, Dict, Any, Optional, Union, Tuple
import time
import json
from datetime import datetime
import logging
import pickle
import tempfile
import os

logger = logging.getLogger(__name__)


@ray.remote
class MLTrainer:
    """Distributed machine learning trainer."""
    
    def __init__(self, model_type: str = "linear_regression"):
        self.model_type = model_type
        self.model = None
        self.training_history = []
        self.is_trained = False
        self.worker_id = f"ml_trainer_{ray.get_runtime_context().get_worker_id()}"
        
    def prepare_data(self, X: List[List[float]], y: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data."""
        X_array = np.array(X)
        y_array = np.array(y)
        
        # Add bias term for linear models
        if self.model_type in ["linear_regression", "logistic_regression"]:
            X_array = np.column_stack([np.ones(X_array.shape[0]), X_array])
        
        return X_array, y_array
    
    def train_linear_regression(self, X: np.ndarray, y: np.ndarray, learning_rate: float = 0.01, epochs: int = 100) -> Dict[str, Any]:
        """Train linear regression model."""
        # Initialize weights
        weights = np.random.normal(0, 0.01, X.shape[1])
        losses = []
        
        start_time = time.time()
        
        for epoch in range(epochs):
            # Forward pass
            predictions = X.dot(weights)
            
            # Calculate loss (MSE)
            loss = np.mean((predictions - y) ** 2)
            losses.append(loss)
            
            # Backward pass (gradient descent)
            gradient = (2 / len(y)) * X.T.dot(predictions - y)
            weights -= learning_rate * gradient
            
            # Early stopping if loss is very small
            if loss < 1e-10:
                break
        
        training_time = time.time() - start_time
        
        self.model = {
            "type": "linear_regression",
            "weights": weights.tolist(),
            "final_loss": loss,
            "epochs_trained": epoch + 1
        }
        self.is_trained = True
        
        # Store training history
        training_record = {
            "timestamp": datetime.now().isoformat(),
            "model_type": self.model_type,
            "epochs": epoch + 1,
            "final_loss": loss,
            "training_time": training_time,
            "data_size": len(y)
        }
        self.training_history.append(training_record)
        
        return {
            "status": "success",
            "model": self.model,
            "losses": losses,
            "training_time": training_time,
            "worker_id": self.worker_id
        }
    
    def train_logistic_regression(self, X: np.ndarray, y: np.ndarray, learning_rate: float = 0.01, epochs: int = 100) -> Dict[str, Any]:
        """Train logistic regression model."""
        # Initialize weights
        weights = np.random.normal(0, 0.01, X.shape[1])
        losses = []
        
        start_time = time.time()
        
        def sigmoid(z):
            # Clip z to prevent overflow
            z = np.clip(z, -250, 250)
            return 1 / (1 + np.exp(-z))
        
        for epoch in range(epochs):
            # Forward pass
            z = X.dot(weights)
            predictions = sigmoid(z)
            
            # Calculate loss (binary cross-entropy)
            # Add small epsilon to prevent log(0)
            epsilon = 1e-15
            predictions = np.clip(predictions, epsilon, 1 - epsilon)
            loss = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
            losses.append(loss)
            
            # Backward pass
            gradient = (1 / len(y)) * X.T.dot(predictions - y)
            weights -= learning_rate * gradient
            
            # Early stopping
            if loss < 1e-10:
                break
        
        training_time = time.time() - start_time
        
        self.model = {
            "type": "logistic_regression",
            "weights": weights.tolist(),
            "final_loss": loss,
            "epochs_trained": epoch + 1
        }
        self.is_trained = True
        
        training_record = {
            "timestamp": datetime.now().isoformat(),
            "model_type": self.model_type,
            "epochs": epoch + 1,
            "final_loss": loss,
            "training_time": training_time,
            "data_size": len(y)
        }
        self.training_history.append(training_record)
        
        return {
            "status": "success",
            "model": self.model,
            "losses": losses,
            "training_time": training_time,
            "worker_id": self.worker_id
        }
    
    def train_neural_network(self, X: np.ndarray, y: np.ndarray, hidden_size: int = 10, learning_rate: float = 0.01, epochs: int = 100) -> Dict[str, Any]:
        """Train simple neural network."""
        # Initialize weights for 2-layer network
        np.random.seed(42)  # For reproducibility
        W1 = np.random.normal(0, 0.1, (X.shape[1], hidden_size))
        b1 = np.zeros(hidden_size)
        W2 = np.random.normal(0, 0.1, (hidden_size, 1))
        b2 = np.zeros(1)
        
        losses = []
        start_time = time.time()
        
        def relu(x):
            return np.maximum(0, x)
        
        def relu_derivative(x):
            return (x > 0).astype(float)
        
        for epoch in range(epochs):
            # Forward pass
            z1 = X.dot(W1) + b1
            a1 = relu(z1)
            z2 = a1.dot(W2) + b2
            predictions = z2.flatten()
            
            # Calculate loss (MSE)
            loss = np.mean((predictions - y) ** 2)
            losses.append(loss)
            
            # Backward pass
            dz2 = 2 * (predictions - y) / len(y)
            dW2 = a1.T.dot(dz2.reshape(-1, 1))
            db2 = np.mean(dz2)
            
            da1 = dz2.reshape(-1, 1).dot(W2.T)
            dz1 = da1 * relu_derivative(z1)
            dW1 = X.T.dot(dz1)
            db1 = np.mean(dz1, axis=0)
            
            # Update weights
            W1 -= learning_rate * dW1
            b1 -= learning_rate * db1
            W2 -= learning_rate * dW2
            b2 -= learning_rate * db2
            
            if loss < 1e-10:
                break
        
        training_time = time.time() - start_time
        
        self.model = {
            "type": "neural_network",
            "W1": W1.tolist(),
            "b1": b1.tolist(),
            "W2": W2.tolist(),
            "b2": b2.tolist(),
            "hidden_size": hidden_size,
            "final_loss": loss,
            "epochs_trained": epoch + 1
        }
        self.is_trained = True
        
        training_record = {
            "timestamp": datetime.now().isoformat(),
            "model_type": self.model_type,
            "epochs": epoch + 1,
            "final_loss": loss,
            "training_time": training_time,
            "data_size": len(y),
            "hidden_size": hidden_size
        }
        self.training_history.append(training_record)
        
        return {
            "status": "success",
            "model": self.model,
            "losses": losses,
            "training_time": training_time,
            "worker_id": self.worker_id
        }
    
    def train(self, X: List[List[float]], y: List[float], **kwargs) -> Dict[str, Any]:
        """Train model based on model type."""
        try:
            X_prepared, y_prepared = self.prepare_data(X, y)
            
            if self.model_type == "linear_regression":
                return self.train_linear_regression(X_prepared, y_prepared, **kwargs)
            elif self.model_type == "logistic_regression":
                return self.train_logistic_regression(X_prepared, y_prepared, **kwargs)
            elif self.model_type == "neural_network":
                return self.train_neural_network(X_prepared, y_prepared, **kwargs)
            else:
                return {"status": "error", "message": f"Unknown model type: {self.model_type}"}
                
        except Exception as e:
            return {"status": "error", "message": str(e), "worker_id": self.worker_id}
    
    def predict(self, X: List[List[float]]) -> Dict[str, Any]:
        """Make predictions with trained model."""
        if not self.is_trained:
            return {"status": "error", "message": "Model not trained"}
        
        try:
            X_array = np.array(X)
            
            if self.model["type"] == "linear_regression":
                # Add bias term
                X_array = np.column_stack([np.ones(X_array.shape[0]), X_array])
                weights = np.array(self.model["weights"])
                predictions = X_array.dot(weights)
                
            elif self.model["type"] == "logistic_regression":
                # Add bias term
                X_array = np.column_stack([np.ones(X_array.shape[0]), X_array])
                weights = np.array(self.model["weights"])
                z = X_array.dot(weights)
                predictions = 1 / (1 + np.exp(-np.clip(z, -250, 250)))
                
            elif self.model["type"] == "neural_network":
                W1 = np.array(self.model["W1"])
                b1 = np.array(self.model["b1"])
                W2 = np.array(self.model["W2"])
                b2 = np.array(self.model["b2"])
                
                z1 = X_array.dot(W1) + b1
                a1 = np.maximum(0, z1)  # ReLU
                z2 = a1.dot(W2) + b2
                predictions = z2.flatten()
            else:
                return {"status": "error", "message": "Unknown model type"}
            
            return {
                "status": "success",
                "predictions": predictions.tolist(),
                "model_type": self.model["type"],
                "worker_id": self.worker_id
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e), "worker_id": self.worker_id}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "worker_id": self.worker_id,
            "model_type": self.model_type,
            "is_trained": self.is_trained,
            "model": self.model,
            "training_history": self.training_history
        }


@ray.remote
class MLInferenceServer:
    """Distributed ML inference server."""
    
    def __init__(self):
        self.models = {}
        self.inference_count = 0
        self.server_id = f"inference_server_{ray.get_runtime_context().get_worker_id()}"
    
    def load_model(self, model_id: str, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Load a trained model."""
        try:
            self.models[model_id] = model_data
            return {
                "status": "success",
                "model_id": model_id,
                "model_type": model_data.get("type", "unknown"),
                "server_id": self.server_id
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def predict_batch(self, model_id: str, X_batch: List[List[float]]) -> Dict[str, Any]:
        """Perform batch inference."""
        if model_id not in self.models:
            return {"status": "error", "message": f"Model {model_id} not found"}
        
        try:
            model = self.models[model_id]
            X_array = np.array(X_batch)
            
            start_time = time.time()
            
            if model["type"] == "linear_regression":
                X_array = np.column_stack([np.ones(X_array.shape[0]), X_array])
                weights = np.array(model["weights"])
                predictions = X_array.dot(weights)
                
            elif model["type"] == "logistic_regression":
                X_array = np.column_stack([np.ones(X_array.shape[0]), X_array])
                weights = np.array(model["weights"])
                z = X_array.dot(weights)
                predictions = 1 / (1 + np.exp(-np.clip(z, -250, 250)))
                
            elif model["type"] == "neural_network":
                W1 = np.array(model["W1"])
                b1 = np.array(model["b1"])
                W2 = np.array(model["W2"])
                b2 = np.array(model["b2"])
                
                z1 = X_array.dot(W1) + b1
                a1 = np.maximum(0, z1)
                z2 = a1.dot(W2) + b2
                predictions = z2.flatten()
            else:
                return {"status": "error", "message": "Unknown model type"}
            
            inference_time = time.time() - start_time
            self.inference_count += len(X_batch)
            
            return {
                "status": "success",
                "predictions": predictions.tolist(),
                "inference_time": round(inference_time, 4),
                "batch_size": len(X_batch),
                "model_id": model_id,
                "server_id": self.server_id
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def get_server_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        return {
            "server_id": self.server_id,
            "models_loaded": len(self.models),
            "total_inferences": self.inference_count,
            "available_models": list(self.models.keys())
        }


@ray.remote
class DistributedMLCoordinator:
    """Coordinates distributed ML training across multiple workers."""
    
    def __init__(self, num_trainers: int = 3):
        self.trainers = []
        self.inference_servers = []
        self.num_trainers = num_trainers
        self.trained_models = {}
        
    def initialize_trainers(self, model_type: str = "linear_regression") -> List[str]:
        """Initialize ML trainers."""
        trainer_ids = []
        for i in range(self.num_trainers):
            trainer = MLTrainer.remote(model_type)
            self.trainers.append(trainer)
            trainer_ids.append(f"trainer_{i}")
        
        return trainer_ids
    
    def distributed_training(self, X: List[List[float]], y: List[float], model_type: str = "linear_regression", **kwargs) -> Dict[str, Any]:
        """Perform distributed training with data parallelism."""
        if not self.trainers:
            self.initialize_trainers(model_type)
        
        # Split data across trainers
        data_size = len(X)
        chunk_size = data_size // len(self.trainers)
        
        training_tasks = []
        for i, trainer in enumerate(self.trainers):
            start_idx = i * chunk_size
            if i == len(self.trainers) - 1:  # Last trainer gets remaining data
                end_idx = data_size
            else:
                end_idx = start_idx + chunk_size
            
            X_chunk = X[start_idx:end_idx]
            y_chunk = y[start_idx:end_idx]
            
            if X_chunk:  # Only train if chunk is not empty
                task = trainer.train.remote(X_chunk, y_chunk, **kwargs)
                training_tasks.append((task, i, len(X_chunk)))
        
        # Collect training results
        training_results = []
        for task, trainer_idx, chunk_size in training_tasks:
            result = ray.get(task)
            result["trainer_index"] = trainer_idx
            result["chunk_size"] = chunk_size
            training_results.append(result)
        
        # Aggregate models (simple averaging for demonstration)
        if model_type in ["linear_regression", "logistic_regression"]:
            all_weights = []
            for result in training_results:
                if result["status"] == "success":
                    all_weights.append(result["model"]["weights"])
            
            if all_weights:
                avg_weights = np.mean(all_weights, axis=0)
                aggregated_model = {
                    "type": model_type,
                    "weights": avg_weights.tolist(),
                    "ensemble_size": len(all_weights)
                }
            else:
                aggregated_model = None
        else:
            # For complex models, use the best performing one
            successful_results = [r for r in training_results if r["status"] == "success"]
            if successful_results:
                best_result = min(successful_results, key=lambda x: x["model"]["final_loss"])
                aggregated_model = best_result["model"]
            else:
                aggregated_model = None
        
        return {
            "aggregated_model": aggregated_model,
            "individual_results": training_results,
            "total_data_size": data_size,
            "trainers_used": len(training_tasks),
            "model_type": model_type
        }
    
    def deploy_model(self, model_id: str, model_data: Dict[str, Any], num_servers: int = 2) -> Dict[str, Any]:
        """Deploy model to inference servers."""
        # Create inference servers if needed
        while len(self.inference_servers) < num_servers:
            server = MLInferenceServer.remote()
            self.inference_servers.append(server)
        
        # Deploy model to all servers
        deployment_tasks = []
        for server in self.inference_servers[:num_servers]:
            task = server.load_model.remote(model_id, model_data)
            deployment_tasks.append(task)
        
        deployment_results = ray.get(deployment_tasks)
        
        # Store model reference
        self.trained_models[model_id] = model_data
        
        return {
            "model_id": model_id,
            "servers_deployed": len(deployment_results),
            "deployment_results": deployment_results
        }
    
    def distributed_inference(self, model_id: str, X_batch: List[List[float]]) -> Dict[str, Any]:
        """Perform distributed inference across multiple servers."""
        if not self.inference_servers:
            return {"status": "error", "message": "No inference servers available"}
        
        # Split batch across available servers
        batch_size = len(X_batch)
        chunk_size = max(1, batch_size // len(self.inference_servers))
        
        inference_tasks = []
        for i, server in enumerate(self.inference_servers):
            start_idx = i * chunk_size
            if i == len(self.inference_servers) - 1:
                end_idx = batch_size
            else:
                end_idx = start_idx + chunk_size
            
            X_chunk = X_batch[start_idx:end_idx]
            if X_chunk:
                task = server.predict_batch.remote(model_id, X_chunk)
                inference_tasks.append((task, start_idx, end_idx))
        
        # Collect results and reconstruct order
        all_predictions = [None] * batch_size
        inference_results = []
        
        for task, start_idx, end_idx in inference_tasks:
            result = ray.get(task)
            inference_results.append(result)
            
            if result["status"] == "success":
                predictions = result["predictions"]
                for j, pred in enumerate(predictions):
                    all_predictions[start_idx + j] = pred
        
        return {
            "predictions": all_predictions,
            "batch_size": batch_size,
            "servers_used": len(inference_tasks),
            "individual_results": inference_results,
            "model_id": model_id
        }


# Utility functions
def create_ml_cluster(num_trainers: int = 3, num_inference_servers: int = 2) -> "DistributedMLCoordinator":
    """Create a complete ML cluster."""
    coordinator = DistributedMLCoordinator.remote(num_trainers)
    return coordinator


def generate_sample_data(n_samples: int = 1000, n_features: int = 5, task_type: str = "regression") -> Tuple[List[List[float]], List[float]]:
    """Generate sample data for ML tasks."""
    np.random.seed(42)
    
    X = np.random.normal(0, 1, (n_samples, n_features))
    
    if task_type == "regression":
        # Generate linear relationship with noise
        true_weights = np.random.normal(0, 1, n_features)
        y = X.dot(true_weights) + np.random.normal(0, 0.1, n_samples)
    else:  # classification
        # Generate binary classification data
        true_weights = np.random.normal(0, 1, n_features)
        linear_combination = X.dot(true_weights)
        probabilities = 1 / (1 + np.exp(-linear_combination))
        y = (probabilities > 0.5).astype(float)
    
    return X.tolist(), y.tolist()