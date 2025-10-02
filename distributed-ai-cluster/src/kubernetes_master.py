#!/usr/bin/env python3
"""
Kubernetes Master Node for Distributed AI Cluster
Coordinates worker nodes and manages model distribution in Kubernetes environment
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
from kubernetes.client.rest import ApiException

# Import our distributed AI framework
from cluster_framework import ClusterNode, MasterNode, NetworkMessage, MessageType

logger = logging.getLogger(__name__)

class KubernetesMasterNode(MasterNode):
    """Kubernetes-native master node with service discovery"""

    def __init__(self, node_id: str, namespace: str = "distributed-ai"):
        # Get Kubernetes service IP instead of static IP
        master_ip = self._get_pod_ip()
        super().__init__(node_id, master_ip)

        self.namespace = namespace
        self.k8s_config_loaded = False
        self.worker_pods: Dict[str, Dict] = {}

    def _get_pod_ip(self) -> str:
        """Get the current pod's IP address"""
        try:
            # Try environment variable first (works in containers)
            pod_ip = os.environ.get('POD_IP')
            if pod_ip:
                return pod_ip

            # Fallback to hostname resolution
            hostname = socket.gethostname()
            return socket.gethostbyname(hostname)
        except Exception as e:
            logger.error(f"Failed to get pod IP: {e}")
            return "127.0.0.1"

    def _load_kubernetes_config(self):
        """Load Kubernetes configuration"""
        try:
            # Try in-cluster config first (for pods)
            config.load_incluster_config()
            self.k8s_config_loaded = True
            logger.info("Loaded in-cluster Kubernetes configuration")
        except config.ConfigException:
            try:
                # Fallback to kubeconfig file (for local development)
                config.load_kube_config()
                self.k8s_config_loaded = True
                logger.info("Loaded kubeconfig for local development")
            except config.ConfigException as e:
                logger.error(f"Failed to load Kubernetes configuration: {e}")
                self.k8s_config_loaded = False

    async def start(self):
        """Start the Kubernetes master node"""
        self.is_running = True
        logger.info(f"Starting Kubernetes master node {self.node_id} in namespace {self.namespace}")

        # Load Kubernetes configuration
        self._load_kubernetes_config()

        # Initialize Kubernetes API client
        if self.k8s_config_loaded:
            self._init_k8s_client()

        # Start all services
        await asyncio.gather(
            self._kubernetes_service_discovery(),
            self._enhanced_heartbeat_service(),
            self._k8s_task_distribution_service(),
            self._model_sync_service(),
            self._monitoring_service()
        )

    def _init_k8s_client(self):
        """Initialize Kubernetes API client"""
        try:
            self.k8s_api = client.CoreV1Api()
            self.apps_api = client.AppsV1Api()
            logger.info("Kubernetes API client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Kubernetes client: {e}")

    async def _kubernetes_service_discovery(self):
        """Discover worker nodes through Kubernetes services"""
        while self.is_running:
            try:
                if not self.k8s_config_loaded:
                    await asyncio.sleep(10)
                    continue

                # Get all pods in the distributed-ai namespace
                pods = self.k8s_api.list_namespaced_pod(
                    namespace=self.namespace,
                    label_selector="app=distributed-ai-cluster,component=worker-node"
                )

                current_worker_ips = set()

                for pod in pods.items:
                    if pod.status.phase == "Running":
                        pod_ip = pod.status.pod_ip
                        if pod_ip:
                            pod_name = pod.metadata.name
                            current_worker_ips.add(pod_ip)

                            # Register or update worker
                            if pod_name not in self.worker_nodes:
                                self.worker_nodes[pod_name] = {
                                    "ip": pod_ip,
                                    "last_seen": time.time(),
                                    "status": "healthy"
                                }
                                logger.info(f"Discovered worker pod: {pod_name} at {pod_ip}")
                            else:
                                self.worker_nodes[pod_name]["last_seen"] = time.time()
                                self.worker_nodes[pod_name]["status"] = "healthy"

                # Mark missing workers as unhealthy
                for worker_name in list(self.worker_nodes.keys()):
                    if self.worker_nodes[worker_name]["ip"] not in current_worker_ips:
                        self.worker_nodes[worker_name]["status"] = "unhealthy"
                        logger.warning(f"Worker {worker_name} appears unhealthy")

                # Clean up dead workers (not seen for 60 seconds)
                current_time = time.time()
                dead_workers = [
                    name for name, info in self.worker_nodes.items()
                    if current_time - info.get("last_seen", 0) > 60
                ]

                for dead_worker in dead_workers:
                    logger.warning(f"Removing dead worker: {dead_worker}")
                    del self.worker_nodes[dead_worker]

                logger.debug(f"Current active workers: {len([w for w in self.worker_nodes.values() if w.get('status') == 'healthy'])}")

                await asyncio.sleep(10)  # Check every 10 seconds

            except ApiException as e:
                logger.error(f"Kubernetes API error during service discovery: {e}")
                await asyncio.sleep(30)
            except Exception as e:
                logger.error(f"Service discovery error: {e}")
                await asyncio.sleep(10)

    async def _enhanced_heartbeat_service(self):
        """Enhanced heartbeat service for Kubernetes"""
        while self.is_running:
            try:
                # Get current cluster status
                cluster_status = await self._get_cluster_status()

                # Send heartbeat to all workers
                heartbeat_msg = NetworkMessage(
                    message_id=str(uuid.uuid4()),
                    message_type=MessageType.HEARTBEAT,
                    source_node=self.node_id,
                    target_node="broadcast",
                    payload={
                        "timestamp": time.time(),
                        "cluster_status": cluster_status,
                        "master_ip": self.node_ip
                    },
                    timestamp=time.time()
                )

                success_count = await self.broadcast_message(heartbeat_msg)
                logger.debug(f"Enhanced heartbeat sent to {success_count} workers")

                await asyncio.sleep(5)

            except Exception as e:
                logger.error(f"Enhanced heartbeat service error: {e}")
                await asyncio.sleep(5)

    async def _get_cluster_status(self) -> Dict:
        """Get comprehensive cluster status"""
        return {
            "total_workers": len(self.worker_nodes),
            "healthy_workers": len([w for w in self.worker_nodes.values() if w.get("status") == "healthy"]),
            "total_requests": sum(w.get("total_requests", 0) for w in self.worker_nodes.values()),
            "average_latency": self._calculate_average_latency(),
            "model_version": getattr(self, 'model_version', 0),
            "uptime": time.time() - getattr(self, 'start_time', time.time())
        }

    def _calculate_average_latency(self) -> float:
        """Calculate average latency across all workers"""
        latencies = [w.get("average_latency", 0) for w in self.worker_nodes.values()]
        return sum(latencies) / len(latencies) if latencies else 0.0

    async def _k8s_task_distribution_service(self):
        """Kubernetes-aware task distribution"""
        while self.is_running:
            try:
                # Get tasks from queue (implement task queuing)
                # For now, simulate task distribution
                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Kubernetes task distribution error: {e}")
                await asyncio.sleep(5)

    async def _monitoring_service(self):
        """Kubernetes-native monitoring service"""
        while self.is_running:
            try:
                # Collect metrics from all workers
                await self._collect_worker_metrics()
                await asyncio.sleep(15)  # Collect metrics every 15 seconds

            except Exception as e:
                logger.error(f"Monitoring service error: {e}")
                await asyncio.sleep(30)

    async def _collect_worker_metrics(self):
        """Collect metrics from worker nodes"""
        for worker_name, worker_info in self.worker_nodes.items():
            if worker_info.get("status") == "healthy":
                try:
                    # In a real implementation, this would collect metrics via HTTP API
                    # For now, we'll simulate metrics collection
                    worker_info["last_metrics_collection"] = time.time()
                    worker_info["cpu_usage"] = 45.0  # Simulated
                    worker_info["memory_usage"] = 60.0  # Simulated
                    worker_info["inference_count"] = worker_info.get("inference_count", 0) + 1

                except Exception as e:
                    logger.error(f"Failed to collect metrics from {worker_name}: {e}")

    async def scale_workers(self, target_replicas: int):
        """Scale the number of worker nodes"""
        if not self.k8s_config_loaded:
            logger.error("Cannot scale workers: Kubernetes config not loaded")
            return False

        try:
            # Patch the deployment to change replica count
            deployment_patch = {
                "spec": {
                    "replicas": target_replicas
                }
            }

            self.apps_api.patch_namespaced_deployment_scale(
                name="ai-worker-nodes",
                namespace=self.namespace,
                body=deployment_patch
            )

            logger.info(f"Scaled worker nodes to {target_replicas} replicas")
            return True

        except ApiException as e:
            logger.error(f"Failed to scale workers: {e}")
            return False

class KubernetesClusterManager:
    """Kubernetes-native cluster manager"""

    def __init__(self, namespace: str = "distributed-ai"):
        self.namespace = namespace
        self.master_node = None
        self.k8s_config = {}

    def load_config(self, config_path: str = "config/k8s_config.json"):
        """Load Kubernetes cluster configuration"""
        try:
            with open(config_path, 'r') as f:
                self.k8s_config = json.load(f)
            logger.info(f"Loaded Kubernetes configuration from {config_path}")
        except FileNotFoundError:
            logger.error(f"Kubernetes config file {config_path} not found")
            self.k8s_config = {}

    def create_master_node(self, node_id: str) -> KubernetesMasterNode:
        """Create Kubernetes master node"""
        self.master_node = KubernetesMasterNode(node_id, self.namespace)
        return self.master_node

    async def deploy_cluster(self):
        """Deploy the entire cluster using Kubernetes"""
        if not self.master_node:
            logger.error("No master node configured")
            return

        logger.info(f"Deploying distributed AI cluster in namespace: {self.namespace}")

        # The actual deployment is handled by Kubernetes manifests
        # This method monitors the deployment status

        deployment_healthy = await self._wait_for_deployment_ready()
        if deployment_healthy:
            logger.info("✅ Distributed AI cluster deployment completed successfully!")

            # Start the master node logic
            await self.master_node.start()
        else:
            logger.error("❌ Cluster deployment failed")
            return False

        return True

    async def _wait_for_deployment_ready(self, timeout: int = 300) -> bool:
        """Wait for Kubernetes deployment to be ready"""
        if not self.master_node.k8s_config_loaded:
            logger.warning("Kubernetes config not loaded, skipping deployment check")
            return True

        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                # Check if worker deployment is ready
                deployment = self.master_node.apps_api.read_namespaced_deployment(
                    name="ai-worker-nodes",
                    namespace=self.namespace
                )

                ready_replicas = deployment.status.ready_replicas or 0
                desired_replicas = deployment.spec.replicas or 1

                if ready_replicas >= desired_replicas:
                    logger.info(f"Deployment ready: {ready_replicas}/{desired_replicas} worker nodes")
                    return True

                logger.info(f"Waiting for deployment: {ready_replicas}/{desired_replicas} ready")
                await asyncio.sleep(10)

            except ApiException as e:
                logger.debug(f"Deployment not yet available: {e}")
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Error checking deployment status: {e}")
                await asyncio.sleep(10)

        logger.error(f"Deployment timeout after {timeout} seconds")
        return False

    async def get_cluster_metrics(self) -> Dict:
        """Get comprehensive cluster metrics"""
        metrics = {
            "timestamp": time.time(),
            "namespace": self.namespace,
            "master_node": self.master_node.node_id if self.master_node else "none",
            "worker_count": len(self.master_node.worker_nodes) if self.master_node else 0,
            "cluster_status": "unknown"
        }

        if self.master_node:
            cluster_status = await self.master_node._get_cluster_status()
            metrics.update(cluster_status)
            metrics["cluster_status"] = "healthy" if cluster_status.get("healthy_workers", 0) > 0 else "degraded"

        return metrics

# Example usage for Kubernetes deployment
async def main():
    """Deploy distributed AI cluster on Kubernetes"""

    # Load configuration
    manager = KubernetesClusterManager()
    manager.load_config()

    # Create master node
    master = manager.create_master_node("k8s-master-1")

    # Deploy cluster
    success = await manager.deploy_cluster()

    if success:
        logger.info("🎉 Distributed AI cluster deployed successfully on Kubernetes!")

        # Keep running for monitoring
        try:
            while True:
                metrics = await manager.get_cluster_metrics()
                logger.info(f"Cluster metrics: {json.dumps(metrics, indent=2)}")
                await asyncio.sleep(60)
        except KeyboardInterrupt:
            logger.info("Shutting down cluster...")
            await master.stop()
    else:
        logger.error("❌ Failed to deploy cluster")
        exit(1)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
