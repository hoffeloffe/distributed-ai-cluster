# Distributed AI System Communication Flow
# This diagram shows exactly how components interact

"""
MASTER NODE COORDINATION PROCESS
==============================

1. MASTER NODE STARTUP
   ┌─────────────────┐
   │ Kubernetes      │
   │ Master Pod      │◄─── Starts first
   │                 │
   │ • Loads config  │
   │ • Initializes   │
   │   K8s API       │
   │ • Binds to port │
   │   8888, 8080    │
   └─────────────────┘

2. WORKER DISCOVERY (Every 10 seconds)
   ┌─────────────────┐    ┌─────────────────┐
   │ Master Node     │───▶│ K8s API         │
   │                 │    │                 │
   │ list_namespaced_│    │ pod.status.phase│
   │ pod() with      │    │ = "Running"     │
   │ label selector  │    │ pod.status.pod_ │
   │                 │    │ ip != null      │
   └─────────────────┘    └─────────────────┘
           │                       │
           ▼                       ▼
   ┌─────────────────┐    ┌─────────────────┐
   │ Check existing  │    │ Register new    │
   │ worker registry │    │ workers in      │
   │                 │    │ worker_nodes{}  │
   └─────────────────┘    └─────────────────┘

3. WORKER REGISTRATION PROCESS
   ┌─────────────────┐    ┌─────────────────┐
   │ Worker Pod      │───▶│ Master Service  │
   │ Starts &        │    │ (ClusterIP)     │
   │ discovers       │    │                 │
   │ master via DNS  │    │ Port: 8888      │
   │                 │    └─────────────────┘
   │ NODE_DISCOVERY  │           │
   │ message with    │           │
   │ capabilities    │           │
   └─────────────────┘           │
           │                       │
           ▼                       ▼
   ┌─────────────────┐    ┌─────────────────┐
   │ Master validates│    │ Master adds     │
   │ worker auth     │    │ worker to       │
   │ & capabilities  │    │ active pool     │
   └─────────────────┘    └─────────────────┘

HEARTBEAT & HEALTH MONITORING
=============================

4. HEARTBEAT LOOP (Every 5 seconds)
   ┌─────────────────┐    ┌─────────────────┐
   │ Master Node     │───▶│ All Workers     │
   │                 │    │                 │
   │ HEARTBEAT msg   │    │ • System metrics│
   │ with cluster    │    │ • Load stats    │
   │ status          │    │ • Health status │
   └─────────────────┘    └─────────────────┘
           │
           ▼
   ┌─────────────────┐
   │ Workers respond │
   │ with current    │
   │ status & metrics│
   └─────────────────┘

AI INFERENCE REQUEST FLOW
=========================

5. CLIENT REQUEST ARRIVAL
   ┌─────────────────┐    ┌─────────────────┐
   │ External Client │───▶│ Nginx Ingress   │
   │ (curl, browser) │    │ Controller      │
   │                 │    │                 │
   │ POST /api/      │    │ Routes to       │
   │ inference       │    │ master service  │
   └─────────────────┘    └─────────────────┘

6. LOAD BALANCING DECISION
   ┌─────────────────┐
   │ Master Node     │
   │                 │
   │ • Checks worker │
   │   load/capacity │
   │ • Selects best  │
   │   worker        │
   │ • Routes request│
   └─────────────────┘

7. DISTRIBUTED INFERENCE
   ┌─────────────────┐    ┌─────────────────┐
   │ Master Node     │───▶│ Worker Node 1   │
   │                 │    │                 │
   │ INFERENCE_      │    │ • Loads model   │
   │ REQUEST with    │    │   shard         │
   │ input data      │    │ • Processes     │
   │                 │    │   data          │
   │                 │    │ • Returns       │
   │                 │    │   results       │
   └─────────────────┘    └─────────────────┘
           │
           ▼ (if model sharding enabled)
   ┌─────────────────┐    ┌─────────────────┐
   │ Master combines │◄───│ Worker Node 2   │
   │ results from    │    │                 │
   │ all shards      │    │ • Process shard │
   │                 │    │ • Return partial│
   │                 │    │   results       │
   └─────────────────┘    └─────────────────┘

MONITORING & OBSERVABILITY
==========================

8. METRICS COLLECTION (Every 15 seconds)
   ┌─────────────────┐    ┌─────────────────┐
   │ Prometheus      │◄───│ Master &        │
   │                 │    │ Workers expose  │
   │ • Scrapes       │    │ metrics on      │
   │   /metrics      │    │ port 9090       │
   │ • Stores time   │    └─────────────────┘
   │   series data   │           │
   └─────────────────┘           │
           │                       │
           ▼                       ▼
   ┌─────────────────┐    ┌─────────────────┐
   │ Grafana         │◄───│ Jaeger traces   │
   │ Dashboards      │    │ requests across │
   │ visualize       │    │ services        │
   │ cluster health  │    └─────────────────┘
   └─────────────────┘

NETWORK PROTOCOL DETAILS
========================

MESSAGE FORMAT:
{
  "message_id": "uuid4-string",
  "message_type": "HEARTBEAT|INFERENCE_REQUEST|MODEL_UPDATE",
  "source_node": "worker-1",
  "target_node": "master",
  "payload": {
    "data": "...",
    "timestamp": 1643123456.789,
    "metadata": {...}
  },
  "timestamp": 1643123456.789
}

COMMUNICATION PORTS:
- 8888: Main cluster communication (gRPC/TCP)
- 8889: Heartbeat monitoring (UDP for low latency)
- 8890: Model synchronization (TCP)
- 8080: HTTP API and monitoring (REST)

ERROR HANDLING:
- Automatic retry with exponential backoff
- Circuit breaker pattern for failed nodes
- Graceful degradation when nodes fail
- Dead node detection and removal
"""
