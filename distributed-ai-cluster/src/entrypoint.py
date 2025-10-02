#!/usr/bin/env python3
"""Container entrypoint for distributed AI cluster."""

import asyncio
import logging
import os
import socket
import uuid

from cluster_framework import ClusterManager
from kubernetes_worker import KubernetesWorkerManager

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


def _resolve_ip() -> str:
    pod_ip = os.getenv("POD_IP")
    if pod_ip:
        return pod_ip
    try:
        return socket.gethostbyname(socket.gethostname())
    except OSError:
        return "127.0.0.1"


def _master_id() -> str:
    return os.getenv("MASTER_NODE_ID", "master-1")


def _worker_id() -> str:
    return os.getenv("WORKER_ID", f"worker-{uuid.uuid4().hex[:6]}")


async def run_master() -> None:
    node_id = _master_id()
    node_ip = _resolve_ip()
    logger.info("Starting master node %s at %s", node_id, node_ip)

    manager = ClusterManager()
    manager.create_master_node(node_id=node_id, node_ip=node_ip)

    try:
        await manager.start_cluster()
    except asyncio.CancelledError:
        logger.info("Master cancelled; shutting down")
    finally:
        await manager.stop_cluster()


async def run_worker() -> None:
    worker_id = _worker_id()
    master_service = os.getenv("MASTER_SERVICE_NAME", "distributed-ai-master")
    namespace = os.getenv("K8S_NAMESPACE", "distributed-ai")

    logger.info(
        "Starting worker %s targeting master service %s in namespace %s",
        worker_id,
        master_service,
        namespace,
    )

    manager = KubernetesWorkerManager(worker_id, master_service=master_service)
    manager.create_worker()
    await manager.run_worker()


def main() -> None:
    role = os.getenv("NODE_ROLE", "master").strip().lower()

    if role == "master":
        asyncio.run(run_master())
    elif role == "worker":
        asyncio.run(run_worker())
    else:
        logger.error("Unknown NODE_ROLE '%s'", role)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
