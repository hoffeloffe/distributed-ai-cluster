#!/usr/bin/env python3
"""
AI Model Distribution System for Raspberry Pi Cluster
Handles model sharding, inference distribution, and synchronization
"""

import asyncio
import json
import pickle
import time
import uuid
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
import logging

logger = logging.getLogger(__name__)

class ModelShard:
    """Represents a shard of a distributed neural network"""

    def __init__(self, shard_id: str, layer_indices: List[int], model_path: str):
        self.shard_id = shard_id
        self.layer_indices = layer_indices
        self.model_path = model_path
        self.model = None
        self.is_loaded = False

    def load_model(self):
        """Load the model shard"""
        try:
            self.model = load_model(self.model_path)
            # Extract only the specified layers
            self.model = Model(
                inputs=self.model.inputs,
                outputs=self.model.layers[max(self.layer_indices)].output
            )
            self.is_loaded = True
            logger.info(f"Loaded model shard {self.shard_id}")
        except Exception as e:
            logger.error(f"Failed to load model shard {self.shard_id}: {e}")

    def inference(self, input_data: np.ndarray) -> np.ndarray:
        """Perform inference on this shard"""
        if not self.is_loaded or self.model is None:
            raise RuntimeError(f"Model shard {self.shard_id} not loaded")

        return self.model.predict(input_data, verbose=0)

class ModelDistributor:
    """Handles model distribution across cluster nodes"""

    def __init__(self, config: Dict):
        self.config = config
        self.model_shards: Dict[str, ModelShard] = {}
        self.shard_assignments: Dict[str, str] = {}  # shard_id -> node_id

    def create_model_shards(self, model_path: str, num_shards: int = 3):
        """Split a model into shards for distribution"""
        try:
            # Load the full model to analyze its structure
            full_model = load_model(model_path)
            total_layers = len(full_model.layers)

            # Simple sharding strategy: divide layers evenly
            layers_per_shard = total_layers // num_shards
            remaining_layers = total_layers % num_shards

            current_layer = 0
            for i in range(num_shards):
                # Calculate layers for this shard
                shard_layers = layers_per_shard
                if i < remaining_layers:
                    shard_layers += 1

                layer_indices = list(range(current_layer, current_layer + shard_layers))

                shard = ModelShard(
                    shard_id=f"shard_{i}",
                    layer_indices=layer_indices,
                    model_path=model_path
                )

                self.model_shards[shard.shard_id] = shard
                current_layer += shard_layers

                logger.info(f"Created shard {shard.shard_id} with layers {layer_indices}")

        except Exception as e:
            logger.error(f"Failed to create model shards: {e}")

class DistributedInference:
    """Handles distributed AI inference across cluster nodes"""

    def __init__(self, node_id: str, cluster_manager, model_distributor: ModelDistributor):
        self.node_id = node_id
        self.cluster_manager = cluster_manager
        self.model_distributor = model_distributor
        self.inference_cache: Dict[str, Tuple[np.ndarray, float]] = {}
        self.cache_ttl = 300  # 5 minutes

    async def distributed_inference(self, input_data: np.ndarray) -> np.ndarray:
        """Perform inference across distributed model shards"""
        start_time = time.time()

        try:
            # Check cache first
            cache_key = self._generate_cache_key(input_data)
            if self._is_cache_valid(cache_key):
                cached_result, _ = self.inference_cache[cache_key]
                logger.info(f"Cache hit for inference request")
                return cached_result

            # Route to appropriate shards based on model distribution
            results = await self._route_to_shards(input_data)

            # Combine results from all shards
            final_result = self._combine_shard_results(results)

            # Cache the result
            self.inference_cache[cache_key] = (final_result, start_time)

            # Clean up old cache entries
            self._cleanup_cache()

            latency = (time.time() - start_time) * 1000
            logger.info(f"Distributed inference completed in {latency:.2f}ms")

            return final_result

        except Exception as e:
            logger.error(f"Distributed inference failed: {e}")
            raise

    def _generate_cache_key(self, input_data: np.ndarray) -> str:
        """Generate cache key for input data"""
        # Use hash of input data shape and first few values
        data_hash = hash(str(input_data.shape) + str(input_data.flatten()[:10]))
        return f"cache_{abs(data_hash)}"

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is still valid"""
        if cache_key not in self.inference_cache:
            return False

        _, cache_time = self.inference_cache[cache_key]
        return (time.time() - cache_time) < self.cache_ttl

    async def _route_to_shards(self, input_data: np.ndarray) -> Dict[str, np.ndarray]:
        """Route inference request to appropriate shards"""
        results = {}

        # For now, simple strategy: send to all shards
        # In production, this would be more intelligent based on model distribution
        for shard_id, shard in self.model_distributor.model_shards.items():
            if shard.is_loaded:
                try:
                    result = await self._send_to_shard(shard_id, input_data)
                    results[shard_id] = result
                except Exception as e:
                    logger.error(f"Shard {shard_id} inference failed: {e}")

        return results

    async def _send_to_shard(self, shard_id: str, input_data: np.ndarray) -> np.ndarray:
        """Send inference request to a specific shard"""
        # This would send data to the appropriate worker node
        # For now, return local inference
        shard = self.model_distributor.model_shards[shard_id]
        return shard.inference(input_data)

    def _combine_shard_results(self, shard_results: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine results from multiple shards"""
        if not shard_results:
            raise RuntimeError("No shard results to combine")

        # Simple combination strategy: average all results
        # In practice, this would depend on the model architecture
        result_arrays = list(shard_results.values())

        if len(result_arrays) == 1:
            return result_arrays[0]
        else:
            # Stack and average
            stacked = np.stack(result_arrays, axis=0)
            return np.mean(stacked, axis=0)

    def _cleanup_cache(self):
        """Remove expired cache entries"""
        current_time = time.time()
        expired_keys = [
            key for key, (_, cache_time) in self.inference_cache.items()
            if (current_time - cache_time) >= self.cache_ttl
        ]

        for key in expired_keys:
            del self.inference_cache[key]

        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

class PerformanceMonitor:
    """Monitor cluster performance and health"""

    def __init__(self):
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_latency": 0.0,
            "node_count": 0,
            "active_shards": 0,
            "cache_hit_rate": 0.0
        }
        self.request_times: List[float] = []

    def record_request(self, latency_ms: float, success: bool = True):
        """Record a request for metrics calculation"""
        self.metrics["total_requests"] += 1

        if success:
            self.metrics["successful_requests"] += 1
        else:
            self.metrics["failed_requests"] += 1

        # Update average latency
        self.request_times.append(latency_ms)
        if len(self.request_times) > 1000:  # Keep only last 1000 requests
            self.request_times = self.request_times[-1000:]

        self.metrics["average_latency"] = sum(self.request_times) / len(self.request_times)

    def record_cache_hit(self):
        """Record cache hit for rate calculation"""
        # This would be tracked separately for cache hit rate calculation
        pass

    def get_cluster_status(self) -> Dict:
        """Get current cluster status"""
        return {
            "timestamp": time.time(),
            "metrics": self.metrics.copy(),
            "cache_size": len(self.inference_cache) if hasattr(self, 'inference_cache') else 0,
            "uptime_seconds": getattr(self, 'start_time', 0)
        }

class RaspberryPiCluster:
    """Main cluster interface for distributed AI"""

    def __init__(self, node_id: str, config_path: str = "config/cluster_config.json"):
        self.node_id = node_id
        self.config = self._load_config(config_path)
        self.cluster_manager = None
        self.model_distributor = None
        self.distributed_inference = None
        self.performance_monitor = PerformanceMonitor()
        self.start_time = time.time()

    def _load_config(self, config_path: str) -> Dict:
        """Load cluster configuration"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Config file {config_path} not found")
            return {}

    def initialize_cluster(self):
        """Initialize the cluster with model distribution"""
        logger.info(f"Initializing cluster node: {self.node_id}")

        # Initialize model distributor
        self.model_distributor = ModelDistributor(self.config)

        # Create model shards based on configuration
        model_path = self.config.get("ai_model", {}).get("model_path", "models/mobilenet_v2.h5")
        num_nodes = len(self.config.get("cluster", {}).get("worker_nodes", [])) + 1  # +1 for master

        self.model_distributor.create_model_shards(model_path, num_nodes)

        # Load model shards for this node
        for shard in self.model_distributor.model_shards.values():
            shard.load_model()

        logger.info(f"Cluster initialized with {len(self.model_distributor.model_shards)} model shards")

    async def process_image(self, image_path: str) -> Dict:
        """Process an image through the distributed AI system"""
        try:
            # Load and preprocess image
            image = tf.keras.preprocessing.image.load_img(
                image_path,
                target_size=self.config["ai_model"]["input_shape"][:2]
            )
            image_array = tf.keras.preprocessing.image.img_to_array(image)
            image_array = np.expand_dims(image_array, axis=0)
            image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)

            # Perform distributed inference
            start_time = time.time()
            results = await self.distributed_inference.distributed_inference(image_array)
            latency = (time.time() - start_time) * 1000

            # Record metrics
            self.performance_monitor.record_request(latency)

            # Process results (top 5 predictions)
            top_5_indices = np.argsort(results[0])[-5:][::-1]
            predictions = []

            for i, class_idx in enumerate(top_5_indices):
                predictions.append({
                    "class_id": int(class_idx),
                    "confidence": float(results[0][class_idx]),
                    "rank": i + 1
                })

            return {
                "success": True,
                "predictions": predictions,
                "latency_ms": latency,
                "processing_node": self.node_id,
                "cluster_status": self.performance_monitor.get_cluster_status()
            }

        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            self.performance_monitor.record_request(0, success=False)
            return {
                "success": False,
                "error": str(e),
                "processing_node": self.node_id
            }

    async def get_cluster_health(self) -> Dict:
        """Get comprehensive cluster health status"""
        return {
            "node_id": self.node_id,
            "uptime_seconds": time.time() - self.start_time,
            "performance_metrics": self.performance_monitor.get_cluster_status(),
            "model_shards": {
                shard_id: {
                    "loaded": shard.is_loaded,
                    "layers": len(shard.layer_indices)
                }
                for shard_id, shard in self.model_distributor.model_shards.items()
            },
            "cluster_config": self.config
        }

# Example usage and testing
async def main():
    """Test the distributed AI cluster"""
    cluster = RaspberryPiCluster("test-node-1")

    # Initialize cluster
    cluster.initialize_cluster()

    # Mock distributed inference for testing
    cluster.distributed_inference = DistributedInference(
        "test-node-1",
        None,
        cluster.model_distributor
    )

    # Test with a sample image (you would replace this with actual image processing)
    print("Distributed AI Cluster initialized successfully!")
    print(f"Active shards: {len(cluster.model_distributor.model_shards)}")

    # Get cluster health
    health = await cluster.get_cluster_health()
    print(f"Cluster health: {json.dumps(health, indent=2)}")

if __name__ == "__main__":
    asyncio.run(main())
