#!/usr/bin/env python3
"""
Automated Deployment Script for Distributed AI Cluster on Rancher/Kubernetes
One-command deployment with validation and monitoring
"""

import subprocess
import sys
import time
import json
import os
from pathlib import Path
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RancherClusterDeployer:
    """Automated deployment for Rancher Kubernetes cluster"""

    def __init__(self, namespace: str = "distributed-ai", registry: str = None):
        self.namespace = namespace
        self.registry = registry or "your-registry.com"
        self.deployment_dir = Path(__file__).parent.parent

    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are met"""
        logger.info("🔍 Checking deployment prerequisites...")

        # Check kubectl
        try:
            result = subprocess.run(["kubectl", "version", "--client"],
                                  capture_output=True, text=True, check=True)
            logger.info("✅ kubectl is available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("❌ kubectl not found. Please install kubectl and configure it for your cluster")
            return False

        # Check Helm
        try:
            result = subprocess.run(["helm", "version"],
                                  capture_output=True, text=True, check=True)
            logger.info("✅ Helm is available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("❌ Helm not found. Please install Helm 3")
            return False

        # Check cluster connectivity
        try:
            result = subprocess.run(["kubectl", "cluster-info"],
                                  capture_output=True, text=True, check=True)
            logger.info("✅ Kubernetes cluster is accessible")
        except subprocess.CalledProcessError:
            logger.error("❌ Cannot connect to Kubernetes cluster. Please check your kubeconfig")
            return False

        # Check if namespace exists
        try:
            result = subprocess.run([
                "kubectl", "get", "namespace", self.namespace
            ], capture_output=True, text=True)

            if result.returncode == 0:
                logger.info(f"✅ Namespace '{self.namespace}' exists")
            else:
                logger.info(f"📦 Creating namespace '{self.namespace}'")
                subprocess.run([
                    "kubectl", "create", "namespace", self.namespace
                ], check=True)
                logger.info(f"✅ Created namespace '{self.namespace}'")

        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to check/create namespace: {e}")
            return False

        return True

    def validate_cluster_resources(self) -> bool:
        """Validate cluster has sufficient resources"""
        logger.info("🔍 Validating cluster resources...")

        try:
            # Check available nodes
            result = subprocess.run([
                "kubectl", "get", "nodes", "-o", "json"
            ], capture_output=True, text=True, check=True)

            nodes_data = json.loads(result.stdout)
            total_nodes = len(nodes_data["items"])

            if total_nodes < 2:
                logger.warning(f"⚠️ Only {total_nodes} nodes available. Recommend 3+ nodes for optimal performance")
            else:
                logger.info(f"✅ {total_nodes} nodes available")

            # Check available CPU and memory
            total_cpu = 0
            total_memory = 0

            for node in nodes_data["items"]:
                allocatable = node["status"]["allocatable"]
                cpu = allocatable.get("cpu", "0")
                memory = allocatable.get("memory", "0")

                # Convert to numbers (rough estimation)
                if cpu.endswith("m"):
                    total_cpu += int(cpu[:-1]) / 1000
                else:
                    total_cpu += int(cpu)

                # Memory in Gi
                if memory.endswith("Gi"):
                    total_memory += int(memory[:-2])
                elif memory.endswith("Mi"):
                    total_memory += int(memory[:-2]) / 1024

            logger.info(f"📊 Cluster resources: {total_cpu:.1f} CPU cores, {total_memory:.1f} Gi memory")

            # Check if we have enough resources for our deployment
            required_cpu = 10  # 4 workers * 2.5 CPU each
            required_memory = 24  # 4 workers * 6 Gi each

            if total_cpu < required_cpu:
                logger.warning(f"⚠️ Available CPU ({total_cpu:.1f}) may be insufficient for 4 workers (need {required_cpu})")
            else:
                logger.info(f"✅ Sufficient CPU resources available")

            if total_memory < required_memory:
                logger.warning(f"⚠️ Available memory ({total_memory:.1f} Gi) may be insufficient for 4 workers (need {required_memory} Gi)")
            else:
                logger.info(f"✅ Sufficient memory resources available")

            return True

        except Exception as e:
            logger.error(f"❌ Failed to validate cluster resources: {e}")
            return False

    def build_and_push_images(self, registry: str = None) -> bool:
        """Build Docker images and push to registry"""
        registry = registry or self.registry

        logger.info(f"🏗️ Building and pushing images to {registry}...")

        # Check if Dockerfile exists
        dockerfile_path = self.deployment_dir / "Dockerfile"
        if not dockerfile_path.exists():
            logger.error(f"❌ Dockerfile not found at {dockerfile_path}")
            return False

        # Build image
        image_tag = f"{registry}/distributed-ai-cluster:latest"

        try:
            logger.info(f"Building Docker image: {image_tag}")
            result = subprocess.run([
                "docker", "build",
                "-t", image_tag,
                str(self.deployment_dir)
            ], check=True, cwd=self.deployment_dir)

            logger.info("✅ Docker image built successfully")

            # Push image
            logger.info(f"Pushing image to registry: {image_tag}")
            result = subprocess.run([
                "docker", "push", image_tag
            ], check=True)

            logger.info("✅ Docker image pushed successfully")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to build/push Docker image: {e}")
            logger.error("💡 Make sure Docker is running and you have registry access")
            return False

    def deploy_with_helm(self, values_file: str = None) -> bool:
        """Deploy using Helm chart"""
        logger.info("🚀 Deploying with Helm...")

        helm_chart_path = self.deployment_dir / "helm" / "distributed-ai-cluster"

        if not helm_chart_path.exists():
            logger.error(f"❌ Helm chart not found at {helm_chart_path}")
            return False

        # Prepare Helm command
        cmd = [
            "helm", "upgrade", "--install",
            "distributed-ai-cluster",
            str(helm_chart_path),
            "--namespace", self.namespace,
            "--create-namespace"
        ]

        # Add custom values file if provided
        if values_file and Path(values_file).exists():
            cmd.extend(["-f", values_file])
            logger.info(f"Using custom values file: {values_file}")

        # Set image registry
        cmd.extend([
            "--set", f"image.registry={self.registry}",
            "--set", "cluster.workers.replicas=4"  # Use all 4 workers
        ])

        try:
            logger.info(f"Running Helm command: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)

            logger.info("✅ Helm deployment completed successfully")
            logger.info("📋 Deployment Summary:")

            # Show deployment status
            time.sleep(5)  # Wait for resources to be created

            # Check deployments
            result = subprocess.run([
                "kubectl", "get", "deployments", "-n", self.namespace
            ], capture_output=True, text=True)

            if result.returncode == 0:
                print(result.stdout)

            # Check pods
            result = subprocess.run([
                "kubectl", "get", "pods", "-n", self.namespace
            ], capture_output=True, text=True)

            if result.returncode == 0:
                print(result.stdout)

            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Helm deployment failed: {e}")
            logger.error(f"Error output: {e.stderr}")
            return False

    def validate_deployment(self) -> bool:
        """Validate that deployment is working correctly"""
        logger.info("🔍 Validating deployment...")

        max_wait_time = 300  # 5 minutes
        start_time = time.time()

        while time.time() - start_time < max_wait_time:
            try:
                # Check if deployments are ready
                result = subprocess.run([
                    "kubectl", "get", "deployments", "-n", self.namespace,
                    "-o", "json"
                ], capture_output=True, text=True, check=True)

                deployments = json.loads(result.stdout)

                all_ready = True
                for deployment in deployments["items"]:
                    ready_replicas = deployment["status"].get("readyReplicas", 0)
                    desired_replicas = deployment["spec"]["replicas"]

                    if ready_replicas < desired_replicas:
                        all_ready = False
                        break

                if all_ready:
                    logger.info("✅ All deployments are ready!")
                    break

                logger.info("⏳ Waiting for deployments to be ready...")
                time.sleep(10)

            except Exception as e:
                logger.error(f"❌ Validation failed: {e}")
                return False

        if not all_ready:
            logger.error(f"❌ Deployment validation failed after {max_wait_time} seconds")
            return False

        # Test API endpoint
        logger.info("🧪 Testing API endpoints...")

        try:
            # Get service IP
            result = subprocess.run([
                "kubectl", "get", "service", "ai-master-service", "-n", self.namespace,
                "-o", "jsonpath={.spec.clusterIP}"
            ], capture_output=True, text=True, check=True)

            service_ip = result.stdout.strip()

            if service_ip:
                logger.info(f"✅ Master service available at {service_ip}:8080")

                # Try to access health endpoint (if curl available)
                try:
                    import requests
                    response = requests.get(f"http://{service_ip}:8080/health", timeout=5)
                    if response.status_code == 200:
                        logger.info("✅ Health check passed")
                    else:
                        logger.warning(f"⚠️ Health check returned status {response.status_code}")
                except ImportError:
                    logger.info("ℹ️ Install requests library to enable health checks")
                except Exception as e:
                    logger.warning(f"⚠️ Health check failed: {e}")

            return True

        except subprocess.CalledProcessError:
            logger.warning("⚠️ Could not retrieve service IP for validation")
            return True  # Still consider deployment successful

    def get_access_information(self):
        """Get information on how to access the deployed system"""
        logger.info("📋 Deployment Access Information")
        logger.info("=" * 40)

        try:
            # Get ingress information
            result = subprocess.run([
                "kubectl", "get", "ingress", "-n", self.namespace,
                "-o", "jsonpath={.items[0].spec.rules[0].host}"
            ], capture_output=True, text=True)

            if result.returncode == 0 and result.stdout.strip():
                host = result.stdout.strip()
                logger.info(f"🌐 Dashboard URL: http://{host}/dashboard")
                logger.info(f"📡 API Endpoint: http://{host}/api")
            else:
                # Fallback to port-forwarding
                logger.info("🔗 Access via port-forwarding:")
                logger.info(f"   kubectl port-forward -n {self.namespace} svc/ai-master-service 8080:8080")
                logger.info("   Then open: http://localhost:8080/dashboard")

            # Show pod information
            result = subprocess.run([
                "kubectl", "get", "pods", "-n", self.namespace,
                "--no-headers"
            ], capture_output=True, text=True)

            if result.returncode == 0:
                logger.info("📦 Running Pods:")
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        parts = line.split()
                        if len(parts) >= 3:
                            pod_name = parts[0]
                            status = parts[2]
                            logger.info(f"   {pod_name}: {status}")

        except Exception as e:
            logger.error(f"Failed to get access information: {e}")

    def cleanup_failed_deployment(self):
        """Clean up failed deployment"""
        logger.info("🧹 Cleaning up failed deployment...")

        try:
            subprocess.run([
                "helm", "uninstall", "distributed-ai-cluster",
                "-n", self.namespace
            ], check=True)

            logger.info("✅ Cleanup completed")
        except subprocess.CalledProcessError:
            logger.warning("⚠️ Cleanup may have failed")

    def deploy(self, skip_build: bool = False, values_file: str = None) -> bool:
        """Complete deployment process"""
        logger.info("🚀 Starting Automated Deployment")
        logger.info("=" * 50)

        # Step 1: Prerequisites
        if not self.check_prerequisites():
            return False

        # Step 2: Resource validation
        if not self.validate_cluster_resources():
            logger.warning("⚠️ Resource validation failed, but continuing...")

        # Step 3: Build and push images (unless skipped)
        if not skip_build:
            if not self.build_and_push_images():
                return False
        else:
            logger.info("⏭️ Skipping Docker build/push")

        # Step 4: Deploy with Helm
        if not self.deploy_with_helm(values_file):
            self.cleanup_failed_deployment()
            return False

        # Step 5: Validate deployment
        if not self.validate_deployment():
            logger.warning("⚠️ Deployment validation failed")
            return False

        # Step 6: Show access information
        self.get_access_information()

        logger.info("🎉 Deployment completed successfully!")
        logger.info("🎯 Your distributed AI cluster is now running on Kubernetes!")

        return True

def main():
    """Main deployment function"""
    parser = argparse.ArgumentParser(description="Deploy Distributed AI Cluster to Rancher/Kubernetes")
    parser.add_argument("--namespace", default="distributed-ai", help="Kubernetes namespace")
    parser.add_argument("--registry", help="Docker registry URL")
    parser.add_argument("--values-file", help="Custom Helm values file")
    parser.add_argument("--skip-build", action="store_true", help="Skip Docker build/push")

    args = parser.parse_args()

    deployer = RancherClusterDeployer(args.namespace, args.registry)

    success = deployer.deploy(
        skip_build=args.skip_build,
        values_file=args.values_file
    )

    if success:
        logger.info("✅ Deployment successful! Check the access information above.")
        return 0
    else:
        logger.error("❌ Deployment failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
