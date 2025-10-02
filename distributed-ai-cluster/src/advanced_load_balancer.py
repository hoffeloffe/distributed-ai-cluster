#!/usr/bin/env python3
"""
Advanced Load Balancer for Distributed AI Cluster
Intelligent workload distribution with multiple algorithms
"""

import time
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class LoadBalancingAlgorithm(Enum):
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_RESPONSE_TIME = "weighted_response_time"
    RESOURCE_AWARE = "resource_aware"
    PREDICTIVE = "predictive"

@dataclass
class WorkerMetrics:
    """Comprehensive worker performance metrics"""
    worker_id: str
    cpu_usage: float
    memory_usage: float
    active_connections: int
    average_response_time: float
    error_rate: float
    throughput: float  # requests per second
    last_updated: float

class AdvancedLoadBalancer:
    """Advanced load balancer with multiple algorithms"""

    def __init__(self):
        self.worker_metrics: Dict[str, WorkerMetrics] = {}
        self.round_robin_counter = 0
        self.algorithm = LoadBalancingAlgorithm.LEAST_CONNECTIONS

    def update_worker_metrics(self, worker_id: str, metrics: Dict):
        """Update metrics for a worker node"""
        self.worker_metrics[worker_id] = WorkerMetrics(
            worker_id=worker_id,
            cpu_usage=metrics.get("cpu_percent", 0),
            memory_usage=metrics.get("memory_percent", 0),
            active_connections=metrics.get("active_connections", 0),
            average_response_time=metrics.get("average_latency", 0),
            error_rate=metrics.get("error_rate", 0),
            throughput=metrics.get("throughput", 0),
            last_updated=time.time()
        )

    def select_worker_round_robin(self, available_workers: List[str]) -> Optional[str]:
        """Simple round-robin selection"""
        if not available_workers:
            return None

        worker = available_workers[self.round_robin_counter % len(available_workers)]
        self.round_robin_counter += 1
        return worker

    def select_worker_least_connections(self, available_workers: List[str]) -> Optional[str]:
        """Select worker with fewest active connections"""
        if not available_workers:
            return None

        best_worker = None
        min_connections = float('inf')

        for worker_id in available_workers:
            if worker_id in self.worker_metrics:
                connections = self.worker_metrics[worker_id].active_connections
                if connections < min_connections:
                    min_connections = connections
                    best_worker = worker_id

        return best_worker or available_workers[0]

    def select_worker_weighted_response_time(self, available_workers: List[str]) -> Optional[str]:
        """Select worker based on response time (faster = more weight)"""
        if not available_workers:
            return None

        # Calculate weights (inverse of response time)
        weights = {}
        total_weight = 0

        for worker_id in available_workers:
            if worker_id in self.worker_metrics:
                response_time = self.worker_metrics[worker_id].average_response_time
                # Avoid division by zero, use small epsilon
                weight = 1.0 / max(response_time, 0.001)
            else:
                weight = 1.0  # Default weight for unknown workers

            weights[worker_id] = weight
            total_weight += weight

        # Weighted random selection
        if total_weight == 0:
            return random.choice(available_workers)

        rand_val = random.random() * total_weight
        cumulative = 0

        for worker_id, weight in weights.items():
            cumulative += weight
            if rand_val <= cumulative:
                return worker_id

        return available_workers[0]

    def select_worker_resource_aware(self, available_workers: List[str]) -> Optional[str]:
        """Select worker based on resource utilization"""
        if not available_workers:
            return None

        best_worker = None
        best_score = float('-inf')

        for worker_id in available_workers:
            if worker_id in self.worker_metrics:
                metrics = self.worker_metrics[worker_id]

                # Calculate composite score (lower resource usage = higher score)
                cpu_score = (100 - metrics.cpu_usage) / 100
                memory_score = (100 - metrics.memory_usage) / 100
                error_penalty = max(0, 1 - metrics.error_rate)

                # Weighted combination
                composite_score = (cpu_score * 0.4 + memory_score * 0.4 + error_penalty * 0.2)
                composite_score += metrics.throughput * 0.1  # Bonus for high throughput

                if composite_score > best_score:
                    best_score = composite_score
                    best_worker = worker_id

        return best_worker or available_workers[0]

    def select_worker_predictive(self, available_workers: List[str], request_complexity: str = "medium") -> Optional[str]:
        """Predictive selection based on request type and worker capabilities"""
        if not available_workers:
            return None

        # Complexity multipliers
        complexity_multipliers = {
            "simple": 0.5,
            "medium": 1.0,
            "complex": 2.0
        }

        complexity = complexity_multipliers.get(request_complexity, 1.0)

        best_worker = None
        best_predicted_time = float('inf')

        for worker_id in available_workers:
            if worker_id in self.worker_metrics:
                metrics = self.worker_metrics[worker_id]

                # Predict processing time based on current load and complexity
                predicted_time = (metrics.average_response_time *
                                (1 + metrics.cpu_usage / 100) *
                                complexity)

                if predicted_time < best_predicted_time:
                    best_predicted_time = predicted_time
                    best_worker = worker_id

        return best_worker or available_workers[0]

    def select_worker(self, available_workers: List[str], algorithm: str = None, request_complexity: str = "medium") -> Optional[str]:
        """Main load balancing decision"""
        if not available_workers:
            return None

        algorithm = algorithm or self.algorithm.value

        if algorithm == LoadBalancingAlgorithm.ROUND_ROBIN.value:
            return self.select_worker_round_robin(available_workers)
        elif algorithm == LoadBalancingAlgorithm.LEAST_CONNECTIONS.value:
            return self.select_worker_least_connections(available_workers)
        elif algorithm == LoadBalancingAlgorithm.WEIGHTED_RESPONSE_TIME.value:
            return self.select_worker_weighted_response_time(available_workers)
        elif algorithm == LoadBalancingAlgorithm.RESOURCE_AWARE.value:
            return self.select_worker_resource_aware(available_workers)
        elif algorithm == LoadBalancingAlgorithm.PREDICTIVE.value:
            return self.select_worker_predictive(available_workers, request_complexity)
        else:
            # Default to least connections
            return self.select_worker_least_connections(available_workers)

    def get_load_balancing_stats(self) -> Dict:
        """Get load balancing statistics"""
        if not self.worker_metrics:
            return {"total_workers": 0, "algorithms": {}}

        stats = {
            "total_workers": len(self.worker_metrics),
            "current_algorithm": self.algorithm.value,
            "worker_distribution": {},
            "algorithms": {}
        }

        # Calculate distribution
        for algorithm in LoadBalancingAlgorithm:
            distribution = {}
            for worker_id in self.worker_metrics.keys():
                available = list(self.worker_metrics.keys())
                selected = self.select_worker(available, algorithm.value)
                distribution[selected] = distribution.get(selected, 0) + 1

            stats["algorithms"][algorithm.value] = distribution

        return stats

class ModelVersionManager:
    """Handle model versioning and hot-swapping"""

    def __init__(self):
        self.current_version = "1.0.0"
        self.model_versions: Dict[str, Dict] = {}
        self.version_history: List[Dict] = []

    def register_model_version(self, version: str, model_path: str, metadata: Dict):
        """Register a new model version"""
        self.model_versions[version] = {
            "model_path": model_path,
            "metadata": metadata,
            "registered_at": time.time(),
            "status": "active"
        }

        self.version_history.append({
            "version": version,
            "action": "registered",
            "timestamp": time.time()
        })

        logger.info(f"Registered model version {version}")
        return True

    def switch_model_version(self, version: str) -> bool:
        """Switch to a different model version across all workers"""
        if version not in self.model_versions:
            logger.error(f"Model version {version} not found")
            return False

        # Validate version is ready
        if self.model_versions[version]["status"] != "active":
            logger.error(f"Model version {version} is not active")
            return False

        old_version = self.current_version
        self.current_version = version

        self.version_history.append({
            "version": version,
            "action": "switched_from",
            "previous_version": old_version,
            "timestamp": time.time()
        })

        logger.info(f"Switched from model version {old_version} to {version}")
        return True

    def get_model_info(self) -> Dict:
        """Get current model information"""
        return {
            "current_version": self.current_version,
            "available_versions": list(self.model_versions.keys()),
            "version_history": self.version_history[-10:],  # Last 10 changes
            "current_model_path": self.model_versions.get(self.current_version, {}).get("model_path")
        }

class PerformanceOptimizer:
    """Dynamic performance optimization"""

    def __init__(self):
        self.optimization_rules = [
            {
                "condition": lambda metrics: metrics.get("average_latency", 0) > 100,
                "action": "increase_batch_size",
                "description": "High latency detected, increasing batch size"
            },
            {
                "condition": lambda metrics: metrics.get("error_rate", 0) > 0.05,
                "action": "enable_circuit_breaker",
                "description": "High error rate, enabling circuit breaker"
            },
            {
                "condition": lambda metrics: metrics.get("cpu_usage", 0) > 90,
                "action": "reduce_concurrency",
                "description": "High CPU usage, reducing concurrent requests"
            }
        ]

    def analyze_and_optimize(self, cluster_metrics: Dict) -> List[Dict]:
        """Analyze metrics and suggest optimizations"""
        optimizations = []

        for rule in self.optimization_rules:
            if rule["condition"](cluster_metrics):
                optimizations.append({
                    "rule": rule["action"],
                    "description": rule["description"],
                    "timestamp": time.time(),
                    "metrics_trigger": cluster_metrics
                })

        return optimizations

class AdvancedClusterManager:
    """Enhanced cluster manager with advanced features"""

    def __init__(self):
        self.load_balancer = AdvancedLoadBalancer()
        self.model_manager = ModelVersionManager()
        self.performance_optimizer = PerformanceOptimizer()
        self.auto_scaling_enabled = True
        self.circuit_breaker_threshold = 0.1  # 10% error rate

    def handle_worker_heartbeat(self, worker_id: str, heartbeat_data: Dict):
        """Enhanced heartbeat processing with metrics update"""
        # Update load balancer metrics
        self.load_balancer.update_worker_metrics(worker_id, heartbeat_data)

        # Check for optimization opportunities
        cluster_metrics = self._aggregate_cluster_metrics()
        optimizations = self.performance_optimizer.analyze_and_optimize(cluster_metrics)

        for optimization in optimizations:
            self._apply_optimization(optimization)

    def _aggregate_cluster_metrics(self) -> Dict:
        """Aggregate metrics across all workers"""
        if not self.load_balancer.worker_metrics:
            return {}

        total_requests = sum(m.throughput for m in self.load_balancer.worker_metrics.values())
        avg_latency = sum(m.average_response_time for m in self.load_balancer.worker_metrics.values()) / len(self.load_balancer.worker_metrics)
        avg_cpu = sum(m.cpu_usage for m in self.load_balancer.worker_metrics.values()) / len(self.load_balancer.worker_metrics)
        avg_memory = sum(m.memory_usage for m in self.load_balancer.worker_metrics.values()) / len(self.load_balancer.worker_metrics)

        return {
            "total_throughput": total_requests,
            "average_latency": avg_latency,
            "average_cpu_usage": avg_cpu,
            "average_memory_usage": avg_memory,
            "worker_count": len(self.load_balancer.worker_metrics)
        }

    def _apply_optimization(self, optimization: Dict):
        """Apply performance optimization"""
        logger.info(f"Applying optimization: {optimization['description']}")

        if optimization["rule"] == "increase_batch_size":
            # Increase batch size across workers
            self._broadcast_config_update({"batch_size": "increase"})

        elif optimization["rule"] == "enable_circuit_breaker":
            # Enable circuit breaker for failing workers
            self._enable_circuit_breaker()

        elif optimization["rule"] == "reduce_concurrency":
            # Reduce concurrent requests per worker
            self._broadcast_config_update({"max_concurrent_requests": "decrease"})

    def _broadcast_config_update(self, config_update: Dict):
        """Broadcast configuration updates to all workers"""
        # Implementation would send config updates to all workers
        logger.info(f"Broadcasting config update: {config_update}")

    def _enable_circuit_breaker(self):
        """Enable circuit breaker for failing workers"""
        # Implementation would mark failing workers as circuit-broken
        logger.info("Circuit breaker enabled for failing workers")

    def get_advanced_stats(self) -> Dict:
        """Get comprehensive cluster statistics"""
        basic_metrics = self._aggregate_cluster_metrics()
        load_balancing_stats = self.load_balancer.get_load_balancing_stats()
        model_info = self.model_manager.get_model_info()

        return {
            "timestamp": time.time(),
            "basic_metrics": basic_metrics,
            "load_balancing": load_balancing_stats,
            "model_info": model_info,
            "optimizations_applied": len(self.performance_optimizer.analyze_and_optimize(basic_metrics))
        }

# Example usage and testing
def demonstrate_advanced_features():
    """Demonstrate advanced load balancing and optimization"""

    print("🚀 Advanced Distributed AI Features Demo")
    print("=" * 50)

    # Initialize advanced components
    cluster_manager = AdvancedClusterManager()

    # Simulate worker metrics
    worker_metrics_data = [
        {
            "worker_id": "worker-1",
            "cpu_percent": 45.0,
            "memory_percent": 60.0,
            "active_connections": 3,
            "average_latency": 25.0,
            "error_rate": 0.01,
            "throughput": 15.0
        },
        {
            "worker_id": "worker-2",
            "cpu_percent": 70.0,
            "memory_percent": 80.0,
            "active_connections": 1,
            "average_latency": 18.0,
            "error_rate": 0.02,
            "throughput": 22.0
        },
        {
            "worker_id": "worker-3",
            "cpu_percent": 30.0,
            "memory_percent": 45.0,
            "active_connections": 5,
            "average_latency": 35.0,
            "error_rate": 0.005,
            "throughput": 12.0
        }
    ]

    # Update metrics for all workers
    for metrics in worker_metrics_data:
        cluster_manager.load_balancer.update_worker_metrics(
            metrics["worker_id"],
            metrics
        )

    # Test different load balancing algorithms
    available_workers = ["worker-1", "worker-2", "worker-3"]

    print("\n🔄 Load Balancing Algorithm Comparison:")
    print("-" * 40)

    algorithms = [
        LoadBalancingAlgorithm.ROUND_ROBIN,
        LoadBalancingAlgorithm.LEAST_CONNECTIONS,
        LoadBalancingAlgorithm.WEIGHTED_RESPONSE_TIME,
        LoadBalancingAlgorithm.RESOURCE_AWARE,
        LoadBalancingAlgorithm.PREDICTIVE
    ]

    for algorithm in algorithms:
        selections = []
        for _ in range(9):  # Run multiple times to see distribution
            selected = cluster_manager.load_balancer.select_worker(
                available_workers,
                algorithm.value
            )
            selections.append(selected)

        distribution = {}
        for worker in selections:
            distribution[worker] = distribution.get(worker, 0) + 1

        print(f"{algorithm.value"25"} -> {distribution}")

    # Demonstrate model versioning
    print("\n📦 Model Version Management:")
    print("-" * 30)

    cluster_manager.model_manager.register_model_version(
        "1.0.0",
        "/models/mobilenet_v1.h5",
        {"accuracy": 0.85, "size_mb": 15}
    )

    cluster_manager.model_manager.register_model_version(
        "1.1.0",
        "/models/mobilenet_v2.h5",
        {"accuracy": 0.92, "size_mb": 18}
    )

    model_info = cluster_manager.model_manager.get_model_info()
    print(f"Current version: {model_info['current_version']}")
    print(f"Available versions: {model_info['available_versions']}")

    # Test performance optimization
    print("\n⚡ Performance Optimization Analysis:")
    print("-" * 40)

    cluster_metrics = cluster_manager._aggregate_cluster_metrics()
    print(f"Cluster metrics: {cluster_metrics}")

    optimizations = cluster_manager.performance_optimizer.analyze_and_optimize(cluster_metrics)
    print(f"Optimizations suggested: {len(optimizations)}")

    for opt in optimizations:
        print(f"  • {opt['description']}")

    # Show comprehensive stats
    print("\n📊 Comprehensive Cluster Statistics:")
    print("-" * 40)

    stats = cluster_manager.get_advanced_stats()
    print(f"Total workers: {stats['basic_metrics']['worker_count']}")
    print(f"Total throughput: {stats['basic_metrics']['total_throughput']:.1f} req/s")
    print(f"Average latency: {stats['basic_metrics']['average_latency']:.1f}ms")
    print(f"Active model version: {stats['model_info']['current_version']}")

    print("\n✅ Advanced Features Demo Complete!")

if __name__ == "__main__":
    demonstrate_advanced_features()
