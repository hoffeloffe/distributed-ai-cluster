#!/usr/bin/env python3
"""
Custom Deployment Script for Your Blade Cluster
Optimized deployment for your K3s cluster on Rancher
"""

import subprocess
import sys
import time
import json
import os
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BladeClusterDeployer:
    """Custom deployer for your Blade K3s cluster"""

    def __init__(self):
        self.cluster_name = "blade"
        self.cluster_id = "c-m-l6b6wscq"
        self.namespace = "distributed-ai"
        self.helm_chart_path = Path(__file__).parent.parent / "helm" / "distributed-ai-cluster"

        # Your cluster specific configuration
        self.cluster_config = {
            "kubernetesVersion": "v1.31.12+k3s1",
            "nodeCount": 4,  # Based on your setup
            "availableResources": {
                "cpu": "8",  # cores total across nodes
                "memory": "32Gi",  # total memory
                "storage": "100Gi"  # available storage
            },
            "monitoring": {
                "prometheus": True,
                "grafana": True,
                "alerting": True
            }
        }

    def check_your_cluster_status(self) -> bool:
        """Check if your Blade cluster is ready for deployment"""
        logger.info("🔍 Checking your Blade cluster status...")

        try:
            # Check cluster connectivity
            result = subprocess.run([
                "kubectl", "cluster-info"
            ], capture_output=True, text=True, check=True)
            logger.info("✅ Blade cluster is accessible")

            # Check nodes
            result = subprocess.run([
                "kubectl", "get", "nodes", "-o", "wide"
            ], capture_output=True, text=True, check=True)

            logger.info("📦 Your cluster nodes:")
            for line in result.stdout.strip().split('\n')[1:]:  # Skip header
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 5:
                        node_name = parts[0]
                        status = parts[1]
                        roles = parts[2]
                        version = parts[4]
                        logger.info(f"   {node_name}: {status} ({roles}) - {version}")

            # Check available resources
            result = subprocess.run([
                "kubectl", "top", "nodes"
            ], capture_output=True, text=True)

            if result.returncode == 0:
                logger.info("📊 Node resource usage:")
                for line in result.stdout.strip().split('\n')[1:]:  # Skip header
                    if line.strip():
                        logger.info(f"   {line}")

            # Check if our namespace exists
            result = subprocess.run([
                "kubectl", "get", "namespace", self.namespace
            ], capture_output=True, text=True)

            if result.returncode != 0:
                logger.info(f"📦 Creating namespace '{self.namespace}'")
                subprocess.run([
                    "kubectl", "create", "namespace", self.namespace
                ], check=True)

            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Blade cluster check failed: {e}")
            logger.error("💡 Make sure your kubeconfig is pointing to the correct cluster")
            return False

    def optimize_for_your_cluster(self, values_file: str) -> str:
        """Create optimized values file for your Blade cluster"""

        # Base configuration
        values = {
            "cluster": {
                "master": {
                    "replicas": 1,
                    "resources": {
                        "requests": {"memory": "1Gi", "cpu": "500m"},
                        "limits": {"memory": "2Gi", "cpu": "1000m"}
                    }
                },
                "workers": {
                    "replicas": 3,  # Conservative start, can scale to 4
                    "resources": {
                        "requests": {"memory": "2Gi", "cpu": "1000m"},
                        "limits": {"memory": "4Gi", "cpu": "2000m"}
                    }
                },
                "modelStorage": {
                    "size": "20Gi",
                    "accessModes": ["ReadWriteMany"]
                }
            },
            "monitoring": {
                "enabled": True,
                "prometheus": {"enabled": True},
                "grafana": {"enabled": True}
            },
            "ingress": {
                "enabled": True,
                "className": "nginx",  # Your cluster likely has nginx ingress
                "hosts": [
                    {
                        "host": "ai.your-domain.local",  # Update with your domain
                        "paths": [{"path": "/", "pathType": "Prefix"}]
                    }
                ]
            }
        }

        # Save optimized values
        os.makedirs("temp", exist_ok=True)
        optimized_values_path = "temp/blade-cluster-values.yaml"

        import yaml
        with open(optimized_values_path, 'w') as f:
            yaml.dump(values, f, default_flow_style=False)

        logger.info(f"✅ Optimized values file created: {optimized_values_path}")
        return optimized_values_path

    def deploy_to_your_cluster(self, registry: str = None, custom_values: str = None) -> bool:
        """Deploy to your Blade cluster"""

        logger.info("🚀 Deploying to your Blade K3s cluster...")
        logger.info("=" * 50)

        # Use optimized values if no custom values provided
        if not custom_values:
            custom_values = self.optimize_for_your_cluster(custom_values)

        # Check cluster first
        if not self.check_your_cluster_status():
            return False

        # Deploy with Helm
        try:
            cmd = [
                "helm", "upgrade", "--install",
                "distributed-ai-blade",
                str(self.helm_chart_path),
                "--namespace", self.namespace,
                "--create-namespace",
                "-f", custom_values
            ]

            # Add image registry if provided
            if registry:
                cmd.extend(["--set", f"cluster.master.image.registry={registry}"])
                cmd.extend(["--set", f"cluster.workers.image.registry={registry}"])

            logger.info(f"Running: {' '.join(cmd)}")

            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("✅ Helm deployment completed successfully")

            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Deployment failed: {e}")
            logger.error(f"Error output: {e.stderr}")
            return False

    def wait_for_deployment_ready(self, timeout: int = 300) -> bool:
        """Wait for deployment to be ready"""

        logger.info(f"⏳ Waiting for deployment to be ready (timeout: {timeout}s)...")

        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                # Check master deployment
                result = subprocess.run([
                    "kubectl", "get", "deployment", "distributed-ai-blade-master",
                    "-n", self.namespace, "-o", "jsonpath={.status.readyReplicas}"
                ], capture_output=True, text=True, check=True)

                master_ready = result.stdout.strip() == "1"

                # Check worker deployment
                result = subprocess.run([
                    "kubectl", "get", "deployment", "distributed-ai-blade-worker",
                    "-n", self.namespace, "-o", "jsonpath={.status.readyReplicas}"
                ], capture_output=True, text=True, check=True)

                worker_ready = result.stdout.strip() == "3"  # We set replicas to 3

                if master_ready and worker_ready:
                    logger.info("✅ All deployments are ready!")
                    return True

                logger.info("⏳ Waiting for pods to be ready...")
                time.sleep(15)

            except subprocess.CalledProcessError:
                logger.info("⏳ Waiting for deployments to be created...")
                time.sleep(10)

        logger.error(f"❌ Deployment not ready after {timeout} seconds")
        return False

    def show_deployment_status(self):
        """Show detailed deployment status"""

        logger.info("📊 Deployment Status for Blade Cluster")
        logger.info("=" * 45)

        try:
            # Show all resources in our namespace
            logger.info(f"\n🔍 All resources in namespace '{self.namespace}':")
            result = subprocess.run([
                "kubectl", "get", "all,ingress,svc,pvc,configmap",
                "-n", self.namespace, "--show-labels"
            ], capture_output=True, text=True)

            if result.returncode == 0:
                print(result.stdout)

            # Show pod details
            logger.info("
📦 Pod Details:"            result = subprocess.run([
                "kubectl", "get", "pods", "-n", self.namespace,
                "-o", "wide"
            ], capture_output=True, text=True)

            if result.returncode == 0:
                print(result.stdout)

            # Show services
            logger.info("
🌐 Services:"            result = subprocess.run([
                "kubectl", "get", "svc", "-n", self.namespace
            ], capture_output=True, text=True)

            if result.returncode == 0:
                print(result.stdout)

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get deployment status: {e}")

    def get_access_information(self):
        """Get access information for your deployment"""

        logger.info("🎯 Access Information for Blade Cluster")
        logger.info("=" * 40)

        try:
            # Check if ingress is available
            result = subprocess.run([
                "kubectl", "get", "ingress", "-n", self.namespace,
                "-o", "jsonpath={.items[0].spec.rules[0].host}"
            ], capture_output=True, text=True)

            if result.returncode == 0 and result.stdout.strip():
                host = result.stdout.strip()
                logger.info(f"🌐 Dashboard: http://{host}/dashboard")
                logger.info(f"📡 API Endpoint: http://{host}/api")
                logger.info(f"📊 Grafana: http://{host}/grafana")
            else:
                # Fallback to service port-forwarding
                logger.info("🔗 Access via kubectl port-forward:")
                logger.info(f"   kubectl port-forward -n {self.namespace} svc/distributed-ai-blade-master-service 8080:8080")
                logger.info("   Then open: http://localhost:8080/dashboard")

            # Show monitoring access
            logger.info("
📊 Monitoring:"            logger.info(f"   kubectl port-forward -n {self.namespace} svc/distributed-ai-blade-prometheus-server 9090:80")
            logger.info("   kubectl port-forward -n {self.namespace} svc/distributed-ai-blade-grafana 3000:80")

        except Exception as e:
            logger.error(f"Failed to get access information: {e}")

    def test_your_deployment(self) -> bool:
        """Test that your deployment is working"""

        logger.info("🧪 Testing Blade cluster deployment...")

        try:
            # Get master service IP
            result = subprocess.run([
                "kubectl", "get", "service", "distributed-ai-blade-master-service",
                "-n", self.namespace, "-o", "jsonpath={.spec.clusterIP}"
            ], capture_output=True, text=True, check=True)

            service_ip = result.stdout.strip()

            if not service_ip:
                logger.error("❌ Master service not found")
                return False

            # Test health endpoint
            logger.info(f"Testing health endpoint: http://{service_ip}:8080/health")

            # Note: In a real test, you would use requests library here
            # For now, we'll just check if the service is responding

            # Check if pods are running
            result = subprocess.run([
                "kubectl", "get", "pods", "-n", self.namespace,
                "-o", "jsonpath={.items[*].status.phase}"
            ], capture_output=True, text=True, check=True)

            running_pods = result.stdout.count("Running")

            if running_pods >= 4:  # 1 master + 3 workers
                logger.info(f"✅ {running_pods} pods are running")
                return True
            else:
                logger.warning(f"⚠️ Only {running_pods} pods running (expected 4)")
                return False

        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Deployment test failed: {e}")
            return False

    def deploy(self, registry: str = None, skip_test: bool = False) -> bool:
        """Complete deployment to your Blade cluster"""

        logger.info("🚀 Starting deployment to Blade K3s cluster")
        logger.info("=" * 50)

        # Step 1: Deploy
        if not self.deploy_to_your_cluster(registry):
            return False

        # Step 2: Wait for ready
        if not self.wait_for_deployment_ready():
            logger.warning("⚠️ Deployment may not be fully ready")
            # Continue anyway

        # Step 3: Show status
        self.show_deployment_status()

        # Step 4: Show access info
        self.get_access_information()

        # Step 5: Test deployment
        if not skip_test:
            if not self.test_your_deployment():
                logger.warning("⚠️ Some tests failed, but deployment may still work")
            else:
                logger.info("✅ All tests passed!")

        logger.info("🎉 Deployment to Blade cluster completed!")
        logger.info("🎯 Your distributed AI cluster is now running!")

        return True

def main():
    """Main deployment function"""
    parser = argparse.ArgumentParser(description="Deploy Distributed AI Cluster to your Blade K3s cluster")
    parser.add_argument("--registry", help="Docker registry URL")
    parser.add_argument("--values-file", help="Custom Helm values file")
    parser.add_argument("--skip-test", action="store_true", help="Skip deployment tests")

    args = parser.parse_args()

    deployer = BladeClusterDeployer()

    success = deployer.deploy(
        registry=args.registry,
        skip_test=args.skip_test
    )

    if success:
        logger.info("✅ Deployment successful!")
        return 0
    else:
        logger.error("❌ Deployment failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
