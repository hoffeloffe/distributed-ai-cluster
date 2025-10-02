#!/usr/bin/env python3
"""
Distributed AI Cluster for Raspberry Pi
Core communication and coordination framework
"""

import asyncio
import json
import pickle
import socket
import struct
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MessageType(Enum):
    HEARTBEAT = "heartbeat"
    MODEL_UPDATE = "model_update"
    INFERENCE_REQUEST = "inference_request"
    INFERENCE_RESPONSE = "inference_response"
    NODE_DISCOVERY = "node_discovery"
    TASK_ASSIGNMENT = "task_assignment"
    METRICS_UPDATE = "metrics_update"

@dataclass
class NetworkMessage:
    message_id: str
    message_type: MessageType
    source_node: str
    target_node: str
    payload: Dict[str, Any]
    timestamp: float

    def to_bytes(self) -> bytes:
        """Serialize message for network transmission"""
        data = {
            'message_id': self.message_id,
            'message_type': self.message_type.value,
            'source_node': self.source_node,
            'target_node': self.target_node,
            'payload': self.payload,
            'timestamp': self.timestamp
        }
        return pickle.dumps(data)

    @classmethod
    def from_bytes(cls, data: bytes) -> 'NetworkMessage':
        """Deserialize message from network"""
        unpacked = pickle.loads(data)
        return cls(
            message_id=unpacked['message_id'],
            message_type=MessageType(unpacked['message_type']),
            source_node=unpacked['source_node'],
            target_node=unpacked['target_node'],
            payload=unpacked['payload'],
            timestamp=unpacked['timestamp']
        )

class ClusterNode(ABC):
    """Base class for cluster nodes"""

    def __init__(self, node_id: str, node_ip: str, config_path: str = "config/cluster_config.json"):
        self.node_id = node_id
        self.node_ip = node_ip
        self.config = self._load_config(config_path)
        self.peers: Dict[str, str] = {}  # node_id -> ip_address
        self.is_running = False

    def _load_config(self, config_path: str) -> Dict:
        """Load cluster configuration"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return {
                "communication_port": 8888,
                "heartbeat_port": 8889,
                "model_sync_port": 8890
            }

    @abstractmethod
    async def start(self):
        """Start the node"""
        pass

    @abstractmethod
    async def stop(self):
        """Stop the node"""
        pass

    async def send_message(self, target_ip: str, message: NetworkMessage) -> bool:
        """Send message to target node"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((target_ip, self.config["communication_port"]))

            # Send message length first (4 bytes)
            data = message.to_bytes()
            length = struct.pack('!I', len(data))

            sock.send(length + data)
            sock.close()
            return True
        except Exception as e:
            logger.error(f"Failed to send message to {target_ip}: {e}")
            return False

    async def broadcast_message(self, message: NetworkMessage) -> int:
        """Broadcast message to all peers"""
        success_count = 0
        for peer_ip in self.peers.values():
            if await self.send_message(peer_ip, message):
                success_count += 1
        return success_count

class MasterNode(ClusterNode):
    """Master node that coordinates the cluster"""

    def __init__(self, node_id: str, node_ip: str):
        super().__init__(node_id, node_ip)
        self.worker_nodes: Dict[str, Dict] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.model_version = 0

    async def start(self):
        """Start the master node"""
        self.is_running = True
        logger.info(f"Starting master node {self.node_id} at {self.node_ip}")

        # Start all services
        await asyncio.gather(
            self._heartbeat_service(),
            self._worker_discovery_service(),
            self._task_distribution_service(),
            self._model_sync_service()
        )

    async def stop(self):
        """Stop the master node"""
        self.is_running = False
        logger.info(f"Stopping master node {self.node_id}")

    async def _heartbeat_service(self):
        """Monitor worker node health"""
        while self.is_running:
            try:
                # Send heartbeat to all workers
                heartbeat_msg = NetworkMessage(
                    message_id=str(uuid.uuid4()),
                    message_type=MessageType.HEARTBEAT,
                    source_node=self.node_id,
                    target_node="broadcast",
                    payload={"timestamp": time.time()},
                    timestamp=time.time()
                )

                success_count = await self.broadcast_message(heartbeat_msg)
                logger.debug(f"Heartbeat sent to {success_count} workers")

                # Check for dead nodes (no response for 30 seconds)
                current_time = time.time()
                dead_nodes = []
                for worker_id, worker_info in self.worker_nodes.items():
                    if current_time - worker_info.get("last_heartbeat", 0) > 30:
                        dead_nodes.append(worker_id)

                for dead_node in dead_nodes:
                    logger.warning(f"Worker {dead_node} appears to be dead")
                    del self.worker_nodes[dead_node]

                await asyncio.sleep(5)  # Check every 5 seconds
            except Exception as e:
                logger.error(f"Heartbeat service error: {e}")
                await asyncio.sleep(5)

    async def _worker_discovery_service(self):
        """Discover and register new worker nodes"""
        # Implementation for worker discovery
        pass

    async def _task_distribution_service(self):
        """Distribute AI tasks to worker nodes"""
        # Implementation for task distribution
        pass

    async def _model_sync_service(self):
        """Synchronize AI models across workers"""
        # Implementation for model synchronization
        pass

class WorkerNode(ClusterNode):
    """Worker node that performs AI inference"""

    def __init__(self, node_id: str, node_ip: str, master_ip: str):
        super().__init__(node_id, node_ip)
        self.master_ip = master_ip
        self.model_shard = None
        self.inference_stats = {
            "total_requests": 0,
            "successful_inferences": 0,
            "average_latency": 0.0,
            "last_heartbeat": time.time()
        }

    async def start(self):
        """Start the worker node"""
        self.is_running = True
        logger.info(f"Starting worker node {self.node_id} at {self.node_ip}")

        # Connect to master and register
        await self._register_with_master()

        # Start all services
        await asyncio.gather(
            self._message_listener(),
            self._heartbeat_service(),
            self._inference_service()
        )

    async def stop(self):
        """Stop the worker node"""
        self.is_running = False
        logger.info(f"Stopping worker node {self.node_id}")

    async def _register_with_master(self):
        """Register this worker with the master node"""
        registration_msg = NetworkMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.NODE_DISCOVERY,
            source_node=self.node_id,
            target_node="master",
            payload={
                "worker_ip": self.node_ip,
                "capabilities": ["image_classification", "tensorflow"],
                "hardware": {"cpu_cores": 4, "memory_gb": 8}
            },
            timestamp=time.time()
        )

        await self.send_message(self.master_ip, registration_msg)
        logger.info(f"Registered with master at {self.master_ip}")

    async def _heartbeat_service(self):
        """Send periodic heartbeats to master"""
        while self.is_running:
            try:
                heartbeat_msg = NetworkMessage(
                    message_id=str(uuid.uuid4()),
                    message_type=MessageType.HEARTBEAT,
                    source_node=self.node_id,
                    target_node="master",
                    payload={
                        "stats": self.inference_stats,
                        "timestamp": time.time()
                    },
                    timestamp=time.time()
                )

                await self.send_message(self.master_ip, heartbeat_msg)
                self.inference_stats["last_heartbeat"] = time.time()

                await asyncio.sleep(3)  # Send heartbeat every 3 seconds
            except Exception as e:
                logger.error(f"Heartbeat service error: {e}")
                await asyncio.sleep(3)

    async def _message_listener(self):
        """Listen for messages from master"""
        # Implementation for message listening
        pass

    async def _inference_service(self):
        """Perform AI inference on assigned tasks"""
        # Implementation for inference service
        pass

class ClusterManager:
    """Main cluster management interface"""

    def __init__(self, config_path: str = "config/cluster_config.json"):
        self.config = self._load_config(config_path)
        self.master_node = None
        self.worker_nodes = []

    def _load_config(self, config_path: str) -> Dict:
        """Load cluster configuration"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Config file {config_path} not found")
            return {}

    def create_master_node(self, node_id: str, node_ip: str) -> MasterNode:
        """Create and configure master node"""
        self.master_node = MasterNode(node_id, node_ip)
        return self.master_node

    def create_worker_node(self, node_id: str, node_ip: str, master_ip: str) -> WorkerNode:
        """Create and configure worker node"""
        worker = WorkerNode(node_id, node_ip, master_ip)
        self.worker_nodes.append(worker)
        return worker

    async def start_cluster(self):
        """Start the entire cluster"""
        if not self.master_node:
            logger.error("No master node configured")
            return

        logger.info("Starting distributed AI cluster...")

        # Start master node
        master_task = asyncio.create_task(self.master_node.start())

        # Start all worker nodes
        worker_tasks = []
        for worker in self.worker_nodes:
            worker_tasks.append(asyncio.create_task(worker.start()))

        # Wait for all nodes to finish
        await asyncio.gather(master_task, *worker_tasks)

    async def stop_cluster(self):
        """Stop the entire cluster"""
        logger.info("Stopping distributed AI cluster...")

        if self.master_node:
            await self.master_node.stop()

        for worker in self.worker_nodes:
            await worker.stop()

if __name__ == "__main__":
    # Example usage
    manager = ClusterManager()

    # Create master node
    master = manager.create_master_node("master-1", "192.168.1.100")

    # Create worker nodes
    for i in range(3):
        worker_id = f"worker-{i+1}"
        worker_ip = f"192.168.1.{101+i}"
        manager.create_worker_node(worker_id, worker_ip, "192.168.1.100")

    # Start the cluster
    try:
        asyncio.run(manager.start_cluster())
    except KeyboardInterrupt:
        logger.info("Shutting down cluster...")
        asyncio.run(manager.stop_cluster())
