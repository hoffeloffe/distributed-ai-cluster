#!/usr/bin/env python3
"""
Kubernetes Worker Node for Distributed AI Cluster
AI inference worker designed to run as a Kubernetes pod
"""

import asyncio
import json
import os
import socket
import time
import uuid
from typing import Dict, List, Optional, Any
import logging
from kubernetes import client, config

# Import our distributed AI framework
from cluster_framework import ClusterNode, WorkerNode, NetworkMessage, MessageType

logger = logging.getLogger(__name__)

class KubernetesWorkerNode(WorkerNode):
    """Kubernetes-native worker node with service discovery"""

    def __init__(self, node_id: str, master_service: str, namespace: str = "distributed-ai"):
        # Get pod IP for communication
        worker_ip = self._get_pod_ip()
        super().__init__(node_id, worker_ip, "localhost")  # Master IP will be resolved via service

        self.master_service = master_service
        self.namespace = namespace
        self.k8s_config_loaded = False
        self.model_loaded = False

    def _get_pod_ip(self) -> str:
        """Get the current pod's IP address"""
        try:
            pod_ip = os.environ.get('POD_IP')
            if pod_ip:
                return pod_ip

            hostname = socket.gethostname()
            return socket.gethostbyname(hostname)
        except Exception as e:
            logger.error(f"Failed to get pod IP: {e}")
            return "127.0.0.1"

    def _resolve_master_ip(self) -> Optional[str]:
        """Resolve master node IP through Kubernetes service"""
        try:
            if not self.k8s_config_loaded:
                return None

            # Get the master service
            service = self.k8s_api.read_namespaced_service(
                name=self.master_service,
                namespace=self.namespace
            )

            # For ClusterIP services, we need to resolve the service IP
            if service.spec.cluster_ip:
                return service.spec.cluster_ip

            # For headless services or other types
            return service.spec.cluster_ip

        except Exception as e:
            logger.error(f"Failed to resolve master IP: {e}")
            return None

    def _load_kubernetes_config(self):
        """Load Kubernetes configuration"""
        try:
            config.load_incluster_config()
            self.k8s_config_loaded = True
            logger.info("Loaded in-cluster Kubernetes configuration")
        except config.ConfigException:
            try:
                config.load_kube_config()
                self.k8s_config_loaded = True
                logger.info("Loaded kubeconfig for local development")
            except config.ConfigException as e:
                logger.error(f"Failed to load Kubernetes configuration: {e}")
                self.k8s_config_loaded = False

    def load_ai_model(self, model_path: str):
        """Load AI model for inference"""
        try:
            # Import here to handle different environments
            import tensorflow as tf

            logger.info(f"Loading AI model from {model_path}")

            # Load model with error handling
            self.model = tf.keras.models.load_model(model_path)
            self.model_loaded = True

            # Warm up the model with a dummy prediction
            dummy_input = tf.random.normal([1] + [224, 224, 3])
            _ = self.model.predict(dummy_input, verbose=0)

            logger.info("✅ AI model loaded and warmed up successfully")

        except ImportError:
            logger.error("TensorFlow not available")
            self.model_loaded = False
        except Exception as e:
            logger.error(f"Failed to load AI model: {e}")
            self.model_loaded = False

    async def start(self):
        """Start the Kubernetes worker node"""
        self.is_running = True
        logger.info(f"Starting Kubernetes worker node {self.node_id} at {self.node_ip}")

        # Load Kubernetes configuration
        self._load_kubernetes_config()

        if self.k8s_config_loaded:
            self._init_k8s_client()

        # Load AI model
        model_path = "/app/models/efficientnet_b0.h5"
        if os.path.exists(model_path):
            self.load_ai_model(model_path)
        else:
            logger.warning(f"Model file not found at {model_path}")

        # Register with master
        await self._register_with_master()

        # Start all services
        await asyncio.gather(
            self._k8s_message_listener(),
            self._enhanced_heartbeat_service(),
            self._inference_service(),
            self._health_check_service()
        )

    def _init_k8s_client(self):
        """Initialize Kubernetes API client"""
        try:
            self.k8s_api = client.CoreV1Api()
            logger.info("Kubernetes API client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Kubernetes client: {e}")

    async def _register_with_master(self):
        """Register this worker with the Kubernetes master service"""
        try:
            # Resolve master IP through service discovery
            master_ip = self._resolve_master_ip()
            if not master_ip:
                logger.error("Could not resolve master IP")
                return

            # Update master IP for communication
            self.master_ip = master_ip

            registration_msg = NetworkMessage(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.NODE_DISCOVERY,
                source_node=self.node_id,
                target_node="master",
                payload={
                    "worker_ip": self.node_ip,
                    "worker_pod": os.environ.get('POD_NAME', self.node_id),
                    "capabilities": ["computer_vision", "tensorflow", "kubernetes"],
                    "hardware": {
                        "cpu_cores": int(os.environ.get('CPU_LIMIT', '2')),
                        "memory_gb": int(os.environ.get('MEMORY_LIMIT', '4')),
                        "namespace": self.namespace
                    },
                    "model_loaded": self.model_loaded,
                    "model_type": "efficientnet_b0"
                },
                timestamp=time.time()
            )

            success = await self.send_message(self.master_ip, registration_msg)

            if success:
                logger.info(f"✅ Successfully registered with master at {self.master_ip}")
            else:
                logger.error("❌ Failed to register with master")

        except Exception as e:
            logger.error(f"Registration failed: {e}")

    async def _k8s_message_listener(self):
        """Kubernetes-aware message listener"""
        while self.is_running:
            try:
                # This would listen for messages from master
                # Implementation depends on your communication protocol
                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Message listener error: {e}")
                await asyncio.sleep(5)

    async def _enhanced_heartbeat_service(self):
        """Enhanced heartbeat service for Kubernetes"""
        while self.is_running:
            try:
                if not self.master_ip:
                    await asyncio.sleep(5)
                    continue

                # Enhanced metrics collection
                system_metrics = await self._collect_system_metrics()

                heartbeat_msg = NetworkMessage(
                    message_id=str(uuid.uuid4()),
                    message_type=MessageType.HEARTBEAT,
                    source_node=self.node_id,
                    target_node="master",
                    payload={
                        "timestamp": time.time(),
                        "pod_name": os.environ.get('POD_NAME', self.node_id),
                        "pod_ip": self.node_ip,
                        "namespace": self.namespace,
                        "system_metrics": system_metrics,
                        "inference_stats": self.inference_stats,
                        "model_loaded": self.model_loaded,
                        "k8s_health": await self._get_k8s_health()
                    },
                    timestamp=time.time()
                )

                await self.send_message(self.master_ip, heartbeat_msg)
                self.inference_stats["last_heartbeat"] = time.time()

                await asyncio.sleep(5)

            except Exception as e:
                logger.error(f"Enhanced heartbeat service error: {e}")
                await asyncio.sleep(5)

    async def _collect_system_metrics(self) -> Dict:
        """Collect comprehensive system metrics"""
        try:
            # Use psutil for system metrics
            import psutil

            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            # Network metrics
            network = psutil.net_io_counters()

            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_gb": memory.used / (1024**3),
                "disk_percent": disk.percent,
                "disk_used_gb": disk.used / (1024**3),
                "network_sent_mb": network.bytes_sent / (1024**2),
                "network_recv_mb": network.bytes_recv / (1024**2),
                "uptime": time.time() - psutil.boot_time()
            }
        except ImportError:
            logger.warning("psutil not available for metrics collection")
            return {"error": "metrics_unavailable"}
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            return {"error": str(e)}

    async def _get_k8s_health(self) -> Dict:
        """Get Kubernetes-specific health metrics"""
        if not self.k8s_config_loaded:
            return {"k8s_status": "config_not_loaded"}

        try:
            # Get pod status
            pod_name = os.environ.get('POD_NAME', 'unknown')
            if pod_name != 'unknown':
                pod = self.k8s_api.read_namespaced_pod(pod_name, self.namespace)
                pod_status = pod.status.phase

                return {
                    "k8s_status": "healthy",
                    "pod_status": pod_status,
                    "pod_ready": all(
                        condition.status == "True"
                        for condition in pod.status.conditions or []
                        if condition.type == "Ready"
                    )
                }
            else:
                return {"k8s_status": "pod_name_unknown"}

        except Exception as e:
            return {"k8s_status": "error", "error": str(e)}

    async def _inference_service(self):
        """Kubernetes-aware inference service"""
        while self.is_running:
            try:
                if not self.model_loaded:
                    logger.debug("Model not loaded, waiting...")
                    await asyncio.sleep(10)
                    continue

                # In a real implementation, this would:
                # 1. Listen for inference requests from master
                # 2. Process the requests using the loaded model
                # 3. Return results to master or directly to clients

                # For now, simulate inference work
                await asyncio.sleep(1)

                # Update inference statistics
                self.inference_stats["total_requests"] += 1
                self.inference_stats["successful_inferences"] += 1

            except Exception as e:
                logger.error(f"Inference service error: {e}")
                await asyncio.sleep(5)

    async def _health_check_service(self):
        """Kubernetes health check service"""
        while self.is_running:
            try:
                # Update health status
                await self._update_health_status()
                await asyncio.sleep(30)  # Health check every 30 seconds

            except Exception as e:
                logger.error(f"Health check service error: {e}")
                await asyncio.sleep(30)

    async def _update_health_status(self):
        """Update worker health status"""
        try:
            if self.k8s_config_loaded:
                # In a real implementation, you might update a ConfigMap
                # or send status to a monitoring system
                pass

        except Exception as e:
            logger.error(f"Failed to update health status: {e}")

    async def perform_inference(self, input_data: Any) -> Dict:
        """Perform AI inference on input data"""
        if not self.model_loaded:
            return {
                "success": False,
                "error": "Model not loaded",
                "node_id": self.node_id
            }

        try:
            start_time = time.time()

            # Perform inference using the loaded model
            # This is a placeholder - you'd implement actual model inference
            # results = self.model.predict(input_data)

            inference_time = (time.time() - start_time) * 1000

            # Update statistics
            self.inference_stats["total_requests"] += 1
            self.inference_stats["successful_inferences"] += 1

            if self.inference_stats["total_requests"] > 0:
                total_latency = self.inference_stats.get("total_latency", 0) + inference_time
                self.inference_stats["total_latency"] = total_latency
                self.inference_stats["average_latency"] = total_latency / self.inference_stats["total_requests"]

            return {
                "success": True,
                "inference_time_ms": inference_time,
                "node_id": self.node_id,
                "model_loaded": self.model_loaded,
                "timestamp": time.time()
            }

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            self.inference_stats["total_requests"] += 1

            return {
                "success": False,
                "error": str(e),
                "node_id": self.node_id
            }

class KubernetesWorkerManager:
    """Manages Kubernetes worker node lifecycle"""

    def __init__(self, worker_id: str, master_service: str = "ai-master-service"):
        self.worker_id = worker_id
        self.master_service = master_service
        self.worker_node = None

    def create_worker(self) -> KubernetesWorkerNode:
        """Create a Kubernetes worker node"""
        self.worker_node = KubernetesWorkerNode(
            node_id=self.worker_id,
            master_service=self.master_service
        )
        return self.worker_node

    async def run_worker(self):
        """Run the worker node"""
        if not self.worker_node:
            logger.error("No worker node created")
            return

        try:
            await self.worker_node.start()
        except KeyboardInterrupt:
            logger.info("Shutting down worker...")
            await self.worker_node.stop()
        except Exception as e:
            logger.error(f"Worker failed: {e}")
            if self.worker_node:
                await self.worker_node.stop()

# Example usage
async def main():
    """Run a Kubernetes worker node"""

    # Create worker manager
    manager = KubernetesWorkerManager("k8s-worker-1")

    # Create and start worker
    worker = manager.create_worker()
    await manager.run_worker()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
