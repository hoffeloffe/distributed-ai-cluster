#!/usr/bin/env python3
"""
Advanced Monitoring & Alerting System for Distributed AI Cluster
Prometheus metrics, Grafana dashboards, and intelligent alerting
"""

import time
import json
import psutil
import threading
from typing import Dict, List, Optional, Callable
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

# Prometheus client for metrics export
try:
    from prometheus_client import (
        CollectorRegistry, Gauge, Counter, Histogram, Summary,
        generate_latest, CONTENT_TYPE_LATEST, start_http_server
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("Prometheus client not available")

logger = logging.getLogger(__name__)

@dataclass
class ClusterMetrics:
    """Comprehensive cluster performance metrics"""
    timestamp: float

    # Node-level metrics
    total_nodes: int
    active_nodes: int
    unhealthy_nodes: int

    # Performance metrics
    total_requests: int
    requests_per_second: float
    average_latency_ms: float
    p95_latency_ms: float
    error_rate: float

    # Resource utilization
    average_cpu_usage: float
    average_memory_usage: float
    average_disk_usage: float
    average_network_io: float

    # AI-specific metrics
    total_inference_requests: int
    inference_success_rate: float
    average_inference_time_ms: float
    model_load_time_ms: float

    # Cache metrics
    cache_hits: int
    cache_misses: int
    cache_hit_rate: float

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {k: v for k, v in asdict(self).items()}

class PrometheusMetricsExporter:
    """Export cluster metrics to Prometheus"""

    def __init__(self, port: int = 9090):
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus metrics disabled - client not available")
            return

        self.port = port
        self.registry = CollectorRegistry()

        # Define Prometheus metrics
        self.cluster_nodes = Gauge(
            'distributed_ai_cluster_nodes_total',
            'Total number of nodes in the cluster',
            registry=self.registry
        )

        self.active_nodes = Gauge(
            'distributed_ai_active_nodes',
            'Number of active nodes in the cluster',
            registry=self.registry
        )

        self.total_requests = Counter(
            'distributed_ai_requests_total',
            'Total number of requests processed',
            registry=self.registry
        )

        self.request_latency = Histogram(
            'distributed_ai_request_latency_seconds',
            'Request processing latency in seconds',
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0],
            registry=self.registry
        )

        self.inference_requests = Counter(
            'distributed_ai_inference_requests_total',
            'Total number of AI inference requests',
            registry=self.registry
        )

        self.inference_latency = Histogram(
            'distributed_ai_inference_latency_seconds',
            'AI inference processing time in seconds',
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0],
            registry=self.registry
        )

        self.cpu_usage = Gauge(
            'distributed_ai_cpu_usage_percent',
            'CPU usage percentage across cluster',
            registry=self.registry
        )

        self.memory_usage = Gauge(
            'distributed_ai_memory_usage_percent',
            'Memory usage percentage across cluster',
            registry=self.registry
        )

        self.error_rate = Gauge(
            'distributed_ai_error_rate',
            'Error rate across cluster operations',
            registry=self.registry
        )

        self.cache_hit_rate = Gauge(
            'distributed_ai_cache_hit_rate',
            'Cache hit rate for inference results',
            registry=self.registry
        )

        # Start HTTP server for metrics
        try:
            start_http_server(port, registry=self.registry)
            logger.info(f"✅ Prometheus metrics server started on port {port}")
        except Exception as e:
            logger.error(f"❌ Failed to start Prometheus metrics server: {e}")

    def update_metrics(self, metrics: ClusterMetrics):
        """Update all Prometheus metrics"""
        if not PROMETHEUS_AVAILABLE:
            return

        try:
            # Node metrics
            self.cluster_nodes.set(metrics.total_nodes)
            self.active_nodes.set(metrics.active_nodes)

            # Performance metrics
            self.total_requests.inc(metrics.total_requests)
            self.request_latency.observe(metrics.average_latency_ms / 1000)

            # AI metrics
            self.inference_requests.inc(metrics.total_inference_requests)
            self.inference_latency.observe(metrics.average_inference_time_ms / 1000)

            # Resource metrics
            self.cpu_usage.set(metrics.average_cpu_usage)
            self.memory_usage.set(metrics.average_memory_usage)
            self.error_rate.set(metrics.error_rate)
            self.cache_hit_rate.set(metrics.cache_hit_rate)

        except Exception as e:
            logger.error(f"Failed to update Prometheus metrics: {e}")

class GrafanaDashboardGenerator:
    """Generate Grafana dashboards for cluster monitoring"""

    def __init__(self):
        self.dashboard_template = {
            "dashboard": {
                "id": None,
                "title": "Distributed AI Cluster Monitor",
                "tags": ["distributed-ai", "kubernetes", "machine-learning"],
                "timezone": "browser",
                "panels": [],
                "time": {
                    "from": "now-1h",
                    "to": "now"
                },
                "refresh": "30s"
            }
        }

    def generate_dashboard_json(self) -> str:
        """Generate complete Grafana dashboard JSON"""

        panels = [
            self._create_overview_panel(),
            self._create_performance_panel(),
            self._create_resource_panel(),
            self._create_ai_metrics_panel(),
            self._create_error_panel(),
            self._create_node_status_panel()
        ]

        dashboard = self.dashboard_template.copy()
        dashboard["dashboard"]["panels"] = panels

        return json.dumps(dashboard, indent=2)

    def _create_overview_panel(self) -> Dict:
        """Create overview metrics panel"""
        return {
            "id": 1,
            "title": "Cluster Overview",
            "type": "stat",
            "targets": [
                {
                    "expr": "distributed_ai_cluster_nodes_total",
                    "refId": "A",
                    "legendFormat": "Total Nodes"
                },
                {
                    "expr": "distributed_ai_active_nodes",
                    "refId": "B",
                    "legendFormat": "Active Nodes"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "color": {"mode": "palette-classic"},
                    "thresholds": {
                        "steps": [
                            {"color": "green", "value": 0},
                            {"color": "yellow", "value": 80},
                            {"color": "red", "value": 90}
                        ]
                    }
                }
            },
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
        }

    def _create_performance_panel(self) -> Dict:
        """Create performance metrics panel"""
        return {
            "id": 2,
            "title": "Request Performance",
            "type": "timeseries",
            "targets": [
                {
                    "expr": "rate(distributed_ai_requests_total[5m])",
                    "refId": "A",
                    "legendFormat": "Requests/sec"
                },
                {
                    "expr": "histogram_quantile(0.95, distributed_ai_request_latency_seconds_bucket)",
                    "refId": "B",
                    "legendFormat": "P95 Latency"
                }
            ],
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
        }

    def _create_resource_panel(self) -> Dict:
        """Create resource utilization panel"""
        return {
            "id": 3,
            "title": "Resource Utilization",
            "type": "timeseries",
            "targets": [
                {
                    "expr": "distributed_ai_cpu_usage_percent",
                    "refId": "A",
                    "legendFormat": "CPU Usage %"
                },
                {
                    "expr": "distributed_ai_memory_usage_percent",
                    "refId": "B",
                    "legendFormat": "Memory Usage %"
                }
            ],
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8}
        }

    def _create_ai_metrics_panel(self) -> Dict:
        """Create AI-specific metrics panel"""
        return {
            "id": 4,
            "title": "AI Inference Metrics",
            "type": "timeseries",
            "targets": [
                {
                    "expr": "rate(distributed_ai_inference_requests_total[5m])",
                    "refId": "A",
                    "legendFormat": "Inference/sec"
                },
                {
                    "expr": "histogram_quantile(0.95, distributed_ai_inference_latency_seconds_bucket)",
                    "refId": "B",
                    "legendFormat": "P95 Inference Time"
                }
            ],
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8}
        }

    def _create_error_panel(self) -> Dict:
        """Create error rate panel"""
        return {
            "id": 5,
            "title": "Error Rate",
            "type": "timeseries",
            "targets": [
                {
                    "expr": "distributed_ai_error_rate",
                    "refId": "A",
                    "legendFormat": "Error Rate"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "thresholds": {
                        "steps": [
                            {"color": "green", "value": 0},
                            {"color": "yellow", "value": 0.05},
                            {"color": "red", "value": 0.1}
                        ]
                    }
                }
            },
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 16}
        }

    def _create_node_status_panel(self) -> Dict:
        """Create node status panel"""
        return {
            "id": 6,
            "title": "Node Status",
            "type": "table",
            "targets": [
                {
                    "expr": "distributed_ai_active_nodes",
                    "refId": "A",
                    "instant": True
                }
            ],
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 16}
        }

class AlertManager:
    """Intelligent alerting system"""

    def __init__(self):
        self.alert_rules = [
            {
                "name": "high_latency",
                "condition": lambda metrics: metrics.average_latency_ms > 100,
                "severity": "warning",
                "message": "High average latency detected: {:.2f}ms",
                "cooldown": 300  # 5 minutes
            },
            {
                "name": "high_error_rate",
                "condition": lambda metrics: metrics.error_rate > 0.05,
                "severity": "critical",
                "message": "High error rate detected: {:.2%}",
                "cooldown": 60  # 1 minute
            },
            {
                "name": "node_down",
                "condition": lambda metrics: metrics.active_nodes < metrics.total_nodes * 0.8,
                "severity": "critical",
                "message": "Node failure detected: {}/{} nodes active",
                "cooldown": 30  # 30 seconds
            },
            {
                "name": "resource_exhaustion",
                "condition": lambda metrics: metrics.average_cpu_usage > 90 or metrics.average_memory_usage > 90,
                "severity": "warning",
                "message": "Resource exhaustion: CPU {:.1f}%, Memory {:.1f}%",
                "cooldown": 120  # 2 minutes
            }
        ]

        self.alert_history: List[Dict] = []
        self.last_alert_times: Dict[str, float] = {}

    def check_alerts(self, metrics: ClusterMetrics) -> List[Dict]:
        """Check all alert conditions"""
        current_time = time.time()
        triggered_alerts = []

        for rule in self.alert_rules:
            rule_name = rule["name"]

            # Check cooldown period
            last_alert = self.last_alert_times.get(rule_name, 0)
            if current_time - last_alert < rule["cooldown"]:
                continue

            # Check condition
            if rule["condition"](metrics):
                alert = {
                    "name": rule_name,
                    "severity": rule["severity"],
                    "message": rule["message"].format(
                        metrics.average_latency_ms if "latency" in rule_name else
                        metrics.error_rate if "error" in rule_name else
                        f"{metrics.active_nodes}/{metrics.total_nodes}" if "node" in rule_name else
                        metrics.average_cpu_usage if "CPU" in rule["message"] else metrics.average_memory_usage
                    ),
                    "timestamp": current_time,
                    "metrics": metrics.to_dict()
                }

                triggered_alerts.append(alert)
                self.last_alert_times[rule_name] = current_time

                logger.warning(f"🚨 ALERT [{rule['severity'].upper()}]: {alert['message']}")

        self.alert_history.extend(triggered_alerts)
        return triggered_alerts

class ClusterMonitor:
    """Comprehensive cluster monitoring system"""

    def __init__(self, port: int = 9090):
        self.port = port
        self.metrics_history: List[ClusterMetrics] = []
        self.max_history_size = 1000

        # Initialize monitoring components
        self.prometheus_exporter = PrometheusMetricsExporter(port)
        self.grafana_dashboard = GrafanaDashboardGenerator()
        self.alert_manager = AlertManager()

        # Start background monitoring
        self.monitoring_thread = None
        self.is_monitoring = False

    def start_monitoring(self, metrics_callback: Optional[Callable] = None):
        """Start continuous monitoring"""
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, args=(metrics_callback,))
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        logger.info("✅ Cluster monitoring started")

    def stop_monitoring(self):
        """Stop monitoring"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("⏹️ Cluster monitoring stopped")

    def _monitoring_loop(self, metrics_callback: Optional[Callable]):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Collect current metrics (this would be replaced with real data collection)
                current_metrics = self._collect_current_metrics()

                # Store in history
                self.metrics_history.append(current_metrics)
                if len(self.metrics_history) > self.max_history_size:
                    self.metrics_history = self.metrics_history[-self.max_history_size:]

                # Update Prometheus metrics
                self.prometheus_exporter.update_metrics(current_metrics)

                # Check alerts
                alerts = self.alert_manager.check_alerts(current_metrics)

                # Call external callback if provided
                if metrics_callback:
                    metrics_callback(current_metrics, alerts)

                time.sleep(10)  # Collect metrics every 10 seconds

            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(30)  # Wait before retrying

    def _collect_current_metrics(self) -> ClusterMetrics:
        """Collect current cluster metrics"""
        # In a real implementation, this would collect from all workers
        # For demo purposes, we'll simulate realistic metrics

        # Simulate some realistic metrics based on time and load
        base_time = time.time()
        hour_of_day = (base_time % 86400) / 3600  # 0-24

        # Simulate higher load during "business hours" (9 AM - 6 PM)
        load_multiplier = 1.0
        if 9 <= hour_of_day <= 18:
            load_multiplier = 2.0 + (hour_of_day - 9) * 0.3

        # Generate realistic metrics
        total_nodes = 4
        active_nodes = max(1, int(total_nodes * (0.9 + 0.1 * (time.time() % 100) / 100)))

        return ClusterMetrics(
            timestamp=base_time,
            total_nodes=total_nodes,
            active_nodes=active_nodes,
            unhealthy_nodes=max(0, total_nodes - active_nodes),
            total_requests=int(1000 * load_multiplier * (time.time() % 10) / 10),
            requests_per_second=10.0 * load_multiplier,
            average_latency_ms=25.0 + 15.0 * (time.time() % 20) / 20,
            p95_latency_ms=50.0 + 30.0 * (time.time() % 15) / 15,
            error_rate=0.01 + 0.02 * (time.time() % 50) / 50,
            average_cpu_usage=40.0 + 30.0 * (time.time() % 25) / 25,
            average_memory_usage=50.0 + 25.0 * (time.time() % 30) / 30,
            average_disk_usage=30.0 + 20.0 * (time.time() % 40) / 40,
            average_network_io=100.0 + 50.0 * (time.time() % 20) / 20,
            total_inference_requests=int(800 * load_multiplier * (time.time() % 8) / 8),
            inference_success_rate=0.95 + 0.04 * (time.time() % 12) / 12,
            average_inference_time_ms=80.0 + 40.0 * (time.time() % 18) / 18,
            model_load_time_ms=500.0 + 200.0 * (time.time() % 22) / 22,
            cache_hits=int(600 * load_multiplier * (time.time() % 6) / 6),
            cache_misses=int(200 * load_multiplier * (time.time() % 8) / 8),
            cache_hit_rate=0.75 + 0.2 * (time.time() % 10) / 10
        )

    def get_current_metrics(self) -> Optional[ClusterMetrics]:
        """Get the most recent metrics"""
        return self.metrics_history[-1] if self.metrics_history else None

    def get_metrics_history(self, hours: int = 1) -> List[ClusterMetrics]:
        """Get metrics history for the specified time period"""
        cutoff_time = time.time() - (hours * 3600)
        return [m for m in self.metrics_history if m.timestamp >= cutoff_time]

    def export_grafana_dashboard(self, filename: str = "distributed_ai_dashboard.json"):
        """Export Grafana dashboard configuration"""
        dashboard_json = self.grafana_dashboard.generate_dashboard_json()

        with open(filename, 'w') as f:
            f.write(dashboard_json)

        logger.info(f"📊 Grafana dashboard exported to {filename}")
        return filename

class MonitoringDemo:
    """Demonstrate the monitoring system"""

    def __init__(self):
        self.monitor = ClusterMonitor()

    def run_demo(self):
        """Run monitoring demonstration"""
        print("📊 Advanced Monitoring & Alerting Demo")
        print("=" * 45)

        # Start monitoring
        self.monitor.start_monitoring(self._metrics_callback)

        # Run for 60 seconds to collect data
        print("⏳ Collecting metrics for 60 seconds...")
        time.sleep(60)

        # Stop monitoring
        self.monitor.stop_monitoring()

        # Show results
        current_metrics = self.monitor.get_current_metrics()
        if current_metrics:
            print("
📈 Final Metrics:"            print(f"   Active nodes: {current_metrics.active_nodes}/{current_metrics.total_nodes}")
            print(f"   Requests/sec: {current_metrics.requests_per_second:.1f}")
            print(f"   Avg latency: {current_metrics.average_latency_ms:.1f}ms")
            print(f"   Error rate: {current_metrics.error_rate:.2%}")
            print(f"   CPU usage: {current_metrics.average_cpu_usage:.1f}%")

        # Export Grafana dashboard
        print("
📊 Exporting Grafana Dashboard..."        dashboard_file = self.monitor.export_grafana_dashboard()
        print(f"✅ Dashboard configuration saved to: {dashboard_file}")

        print("
🚨 Alert History:"        for alert in self.monitor.alert_manager.alert_history[-5:]:  # Last 5 alerts
            print(f"   [{alert['severity'].upper()}] {alert['message']}")

        print("
✅ Monitoring Demo Complete!"
    def _metrics_callback(self, metrics: ClusterMetrics, alerts: List[Dict]):
        """Callback for real-time metrics display"""
        if alerts:
            print(f"🚨 ALERTS: {len(alerts)} triggered")

        # Show metrics every 30 seconds
        if int(metrics.timestamp) % 30 == 0:
            print(f"📊 {time.strftime('%H:%M:%S')} - Nodes: {metrics.active_nodes}/{metrics.total_nodes}, "
                  f"Latency: {metrics.average_latency_ms:.1f}ms, "
                  f"Errors: {metrics.error_rate:.2%}")

def main():
    """Run monitoring demonstration"""
    demo = MonitoringDemo()
    demo.run_demo()

if __name__ == "__main__":
    main()
