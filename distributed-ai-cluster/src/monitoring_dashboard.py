#!/usr/bin/env python3
"""
Distributed AI Cluster Monitoring Dashboard
Real-time web dashboard for monitoring cluster performance
"""

import asyncio
import json
import time
from typing import Dict, List, Optional
import aiohttp
from aiohttp import web
import aiofiles
import psutil
import GPUtil
import socket
import threading
import webbrowser
import logging

logger = logging.getLogger(__name__)

class ClusterMonitor:
    """Monitor cluster performance and health"""

    def __init__(self, cluster_config: Dict):
        self.cluster_config = cluster_config
        self.node_metrics: Dict[str, Dict] = {}
        self.start_time = time.time()

    async def collect_system_metrics(self) -> Dict:
        """Collect local system metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_freq = psutil.cpu_freq()

            # Memory metrics
            memory = psutil.virtual_memory()

            # Network metrics
            network = psutil.net_io_counters()

            # GPU metrics (if available)
            gpu_metrics = {}
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    gpu_metrics = {
                        "gpu_utilization": gpu.load * 100,
                        "gpu_memory_used": gpu.memoryUsed,
                        "gpu_memory_total": gpu.memoryTotal,
                        "gpu_temperature": gpu.temperature
                    }
            except:
                pass  # No GPU available

            # Disk metrics
            disk = psutil.disk_usage('/')

            return {
                "timestamp": time.time(),
                "uptime_seconds": time.time() - self.start_time,
                "cpu": {
                    "percent": cpu_percent,
                    "frequency_mhz": cpu_freq.current if cpu_freq else 0
                },
                "memory": {
                    "total_gb": memory.total / (1024**3),
                    "used_gb": memory.used / (1024**3),
                    "percent": memory.percent
                },
                "network": {
                    "bytes_sent_mb": network.bytes_sent / (1024**2),
                    "bytes_recv_mb": network.bytes_recv / (1024**2)
                },
                "disk": {
                    "total_gb": disk.total / (1024**3),
                    "used_gb": disk.used / (1024**3),
                    "percent": disk.percent
                },
                **gpu_metrics
            }
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            return {}

    async def collect_cluster_metrics(self) -> Dict:
        """Collect metrics from all cluster nodes"""
        cluster_data = {
            "timestamp": time.time(),
            "total_nodes": 0,
            "active_nodes": 0,
            "total_requests": 0,
            "average_latency": 0,
            "nodes": {}
        }

        # Try to collect metrics from each configured node
        for node_ip in self.cluster_config.get("worker_nodes", []):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"http://{node_ip}:8080/metrics") as response:
                        if response.status == 200:
                            node_data = await response.json()
                            cluster_data["nodes"][node_ip] = node_data
                            cluster_data["active_nodes"] += 1
                            cluster_data["total_requests"] += node_data.get("total_requests", 0)
            except Exception as e:
                logger.debug(f"Could not collect metrics from {node_ip}: {e}")

        cluster_data["total_nodes"] = len(self.cluster_config.get("worker_nodes", []))

        # Calculate average latency
        if cluster_data["total_requests"] > 0:
            total_latency = sum(
                node.get("average_latency", 0) * node.get("total_requests", 0)
                for node in cluster_data["nodes"].values()
            )
            cluster_data["average_latency"] = total_latency / cluster_data["total_requests"]

        return cluster_data

class WebDashboard:
    """Web dashboard for cluster monitoring"""

    def __init__(self, cluster_monitor: ClusterMonitor, host: str = "0.0.0.0", port: int = 8080):
        self.cluster_monitor = cluster_monitor
        self.host = host
        self.port = port
        self.app = web.Application()
        self.setup_routes()

    def setup_routes(self):
        """Set up web routes"""
        self.app.router.add_get('/', self.index_handler)
        self.app.router.add_get('/metrics', self.metrics_handler)
        self.app.router.add_get('/api/cluster-status', self.cluster_status_handler)
        self.app.router.add_get('/api/node-metrics', self.node_metrics_handler)
        self.app.router.add_static('/static', path='static')

    async def index_handler(self, request):
        """Serve the main dashboard page"""
        html_content = await self._generate_dashboard_html()
        return web.Response(text=html_content, content_type='text/html')

    async def metrics_handler(self, request):
        """Serve metrics JSON endpoint"""
        metrics = await self.cluster_monitor.collect_system_metrics()
        return web.json_response(metrics)

    async def cluster_status_handler(self, request):
        """Serve cluster status JSON endpoint"""
        cluster_metrics = await self.cluster_monitor.collect_cluster_metrics()
        return web.json_response(cluster_metrics)

    async def node_metrics_handler(self, request):
        """Serve individual node metrics"""
        node_id = request.query.get('node_id', 'local')
        if node_id == 'local':
            metrics = await self.cluster_monitor.collect_system_metrics()
        else:
            # Try to get metrics from specific node
            metrics = {"error": "Node metrics not available"}

        return web.json_response(metrics)

    async def _generate_dashboard_html(self) -> str:
        """Generate the HTML dashboard"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Distributed AI Cluster Monitor</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            min-height: 100vh;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        .header {
            text-align: center;
            margin-bottom: 40px;
        }
        .header h1 {
            font-size: 3em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }
        .stat-card {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 25px;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.2);
            transition: transform 0.3s ease;
        }
        .stat-card:hover {
            transform: translateY(-5px);
        }
        .stat-value {
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .stat-label {
            font-size: 0.9em;
            opacity: 0.8;
        }
        .charts-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 40px;
        }
        .chart-card {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        .nodes-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }
        .node-card {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        .node-status {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-online { background-color: #4CAF50; }
        .status-offline { background-color: #f44336; }
        .status-warning { background-color: #ff9800; }
        .refresh-btn {
            background: rgba(255, 255, 255, 0.2);
            border: none;
            color: white;
            padding: 10px 20px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 16px;
            margin-bottom: 20px;
        }
        .refresh-btn:hover {
            background: rgba(255, 255, 255, 0.3);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Distributed AI Cluster Monitor</h1>
            <p>Real-time performance monitoring for your Raspberry Pi AI cluster</p>
            <button class="refresh-btn" onclick="refreshData()">🔄 Refresh Data</button>
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value" id="total-nodes">0</div>
                <div class="stat-label">Total Nodes</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="active-nodes">0</div>
                <div class="stat-label">Active Nodes</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="total-requests">0</div>
                <div class="stat-label">Total Requests</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="avg-latency">0ms</div>
                <div class="stat-label">Avg Latency</div>
            </div>
        </div>

        <div class="charts-container">
            <div class="chart-card">
                <h3>CPU & Memory Usage</h3>
                <canvas id="systemChart" width="400" height="200"></canvas>
            </div>
            <div class="chart-card">
                <h3>Network Traffic</h3>
                <canvas id="networkChart" width="400" height="200"></canvas>
            </div>
        </div>

        <div class="charts-container">
            <div class="chart-card">
                <h3>Request Performance</h3>
                <canvas id="performanceChart" width="400" height="200"></canvas>
            </div>
            <div class="chart-card">
                <h3>Cluster Health</h3>
                <canvas id="healthChart" width="400" height="200"></canvas>
            </div>
        </div>

        <h2>Node Details</h2>
        <div class="nodes-grid" id="nodes-container">
            <!-- Node cards will be populated by JavaScript -->
        </div>
    </div>

    <script>
        // Dashboard JavaScript will go here
        console.log('Distributed AI Cluster Monitor loaded');

        // Auto-refresh data every 5 seconds
        setInterval(refreshData, 5000);

        async function refreshData() {
            try {
                // Fetch cluster status
                const clusterResponse = await fetch('/api/cluster-status');
                const clusterData = await clusterResponse.json();

                // Fetch local metrics
                const metricsResponse = await fetch('/api/node-metrics');
                const metricsData = await metricsResponse.json();

                updateStats(clusterData);
                updateCharts(metricsData);
                updateNodes(clusterData);

            } catch (error) {
                console.error('Failed to refresh data:', error);
            }
        }

        function updateStats(data) {
            document.getElementById('total-nodes').textContent = data.total_nodes;
            document.getElementById('active-nodes').textContent = data.active_nodes;
            document.getElementById('total-requests').textContent = data.total_requests;
            document.getElementById('avg-latency').textContent = Math.round(data.average_latency) + 'ms';
        }

        function updateCharts(data) {
            // Update charts with new data
            console.log('Updating charts with data:', data);
        }

        function updateNodes(data) {
            const container = document.getElementById('nodes-container');
            container.innerHTML = '';

            Object.entries(data.nodes).forEach(([nodeId, nodeData]) => {
                const nodeCard = document.createElement('div');
                nodeCard.className = 'node-card';

                nodeCard.innerHTML = `
                    <h4>Node: ${nodeId}</h4>
                    <p><span class="node-status status-online"></span>Online</p>
                    <p>Requests: ${nodeData.total_requests || 0}</p>
                    <p>Avg Latency: ${Math.round(nodeData.average_latency || 0)}ms</p>
                `;

                container.appendChild(nodeCard);
            });
        }

        // Initial data load
        refreshData();
    </script>
</body>
</html>
        """

async def create_monitoring_dashboard(config_path: str = "config/cluster_config.json"):
    """Create and start the monitoring dashboard"""

    # Load cluster configuration
    try:
        with open(config_path, 'r') as f:
            cluster_config = json.load(f)
    except FileNotFoundError:
        logger.error(f"Config file {config_path} not found")
        cluster_config = {}

    # Create cluster monitor
    monitor = ClusterMonitor(cluster_config)

    # Create web dashboard
    dashboard = WebDashboard(monitor)

    # Start the web server
    logger.info("Starting monitoring dashboard on http://localhost:8080")
    logger.info("Open your browser and navigate to http://localhost:8080")

    # Auto-open browser (optional)
    try:
        webbrowser.open("http://localhost:8080")
    except:
        pass

    # Start the server
    runner = web.AppRunner(dashboard.app)
    await runner.setup()
    site = web.TCPSite(runner, dashboard.host, dashboard.port)
    await site.start()

    logger.info("Monitoring dashboard started successfully!")

    return dashboard, monitor

if __name__ == "__main__":
    # Start the monitoring dashboard
    async def main():
        dashboard, monitor = await create_monitoring_dashboard()
        logger.info("Dashboard is running. Press Ctrl+C to stop.")

        # Keep the dashboard running
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down dashboard...")

    asyncio.run(main())
