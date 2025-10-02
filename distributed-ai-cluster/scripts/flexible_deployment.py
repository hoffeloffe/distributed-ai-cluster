#!/usr/bin/env python3
"""
Flexible Deployment Script - Direct or Fleet GitOps
Supports both direct deployment and Fleet GitOps for production management
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

class FlexibleDeployer:
    """Supports both direct deployment and Fleet GitOps"""

    def __init__(self):
        self.cluster_name = "blade"
        self.cluster_id = "c-m-l6b6wscq"
        self.namespace = "distributed-ai"

    def check_fleet_status(self) -> bool:
        """Check if Fleet is available in your cluster"""
        logger.info("🔍 Checking Fleet GitOps controller...")

        try:
            # Check if Fleet CRDs are installed
            result = subprocess.run([
                "kubectl", "get", "crd", "gitrepos.fleet.cattle.io"
            ], capture_output=True, text=True)

            if result.returncode == 0:
                logger.info("✅ Fleet GitOps controller is available")
                return True
            else:
                logger.info("ℹ️ Fleet GitOps controller not found")
                logger.info("💡 Install Fleet with: kubectl apply -f https://github.com/rancher/fleet/releases/latest/download/fleet-crd.yaml")
                logger.info("   Then: kubectl apply -f https://github.com/rancher/fleet/releases/latest/download/fleet.yaml")
                return False

        except subprocess.CalledProcessError:
            logger.info("ℹ️ Fleet not available")
            return False

    def deploy_with_fleet(self, git_repo_url: str, branch: str = "main") -> bool:
        """Deploy using Fleet GitOps"""
        logger.info(f"🚀 Deploying with Fleet GitOps from {git_repo_url}")

        # Create Fleet GitRepo resource
        fleet_manifest = f'''apiVersion: fleet.cattle.io/v1alpha1
kind: GitRepo
metadata:
  name: distributed-ai-cluster
  namespace: fleet-default
spec:
  repo: {git_repo_url}
  branch: {branch}
  paths:
    - fleet-gitops-setup.yaml
  targets:
    - clusterName: {self.cluster_name}
      clusterSelector: {{}}
'''

        # Save manifest temporarily
        with open("temp/fleet-gitrepo.yaml", "w") as f:
            f.write(fleet_manifest)

        try:
            # Apply Fleet GitRepo
            result = subprocess.run([
                "kubectl", "apply", "-f", "temp/fleet-gitrepo.yaml"
            ], check=True, capture_output=True, text=True)

            logger.info("✅ Fleet GitRepo created successfully")
            logger.info("⏳ Waiting for Fleet to sync...")
            time.sleep(30)

            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Fleet deployment failed: {e}")
            return False

    def deploy_direct_helm(self, registry: str = None) -> bool:
        """Deploy directly with Helm (traditional approach)"""
        logger.info("🚀 Deploying directly with Helm...")

        helm_chart_path = Path(__file__).parent.parent / "helm" / "distributed-ai-cluster"

        # Create optimized values for your cluster
        values_content = f'''cluster:
  master:
    replicas: 1
    image:
      registry: {registry or 'your-registry.com'}
      repository: distributed-ai-cluster/master
      tag: "latest"
    resources:
      requests:
        memory: "1Gi"
        cpu: "500m"
      limits:
        memory: "2Gi"
        cpu: "1000m"

  workers:
    replicas: 3
    image:
      registry: {registry or 'your-registry.com'}
      repository: distributed-ai-cluster/worker
      tag: "latest"
    resources:
      requests:
        memory: "2Gi"
        cpu: "1000m"
      limits:
        memory: "4Gi"
        cpu: "2000m"

  modelStorage:
    size: "20Gi"
    accessModes: ["ReadWriteMany"]

monitoring:
  enabled: true
  prometheus:
    enabled: true
  grafana:
    enabled: true

ingress:
  enabled: true
  className: "nginx"
  hosts:
    - host: "ai.your-domain.local"
      paths:
        - path: "/"
          pathType: "Prefix"

security:
  networkPolicy:
    enabled: true
  rbac:
    create: true
'''

        # Save values file
        os.makedirs("temp", exist_ok=True)
        values_file = "temp/direct-deployment-values.yaml"

        with open(values_file, "w") as f:
            f.write(values_content)

        try:
            # Deploy with Helm
            cmd = [
                "helm", "upgrade", "--install",
                "distributed-ai-blade",
                str(helm_chart_path),
                "--namespace", self.namespace,
                "--create-namespace",
                "-f", values_file
            ]

            logger.info(f"Running: {' '.join(cmd)}")

            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("✅ Direct Helm deployment completed")

            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Direct deployment failed: {e}")
            return False

    def wait_for_ready(self, timeout: int = 300) -> bool:
        """Wait for deployment to be ready"""
        logger.info(f"⏳ Waiting for deployment to be ready (timeout: {timeout}s)...")

        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                # Check deployments
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
                    return True

                logger.info("⏳ Waiting for pods to be ready...")
                time.sleep(15)

            except subprocess.CalledProcessError:
                logger.info("⏳ Waiting for deployments to be created...")
                time.sleep(10)

        logger.error(f"❌ Deployment not ready after {timeout} seconds")
        return False

    def show_deployment_info(self, deployment_method: str):
        """Show deployment information"""
        logger.info(f"📊 Deployment Information ({deployment_method})")
        logger.info("=" * 50)

        try:
            # Show resources
            logger.info(f"\n🔍 Resources in namespace '{self.namespace}':")
            result = subprocess.run([
                "kubectl", "get", "all,ingress,svc,pvc,configmap",
                "-n", self.namespace
            ], capture_output=True, text=True)

            if result.returncode == 0:
                print(result.stdout)

            # Show access information
            logger.info("\n🎯 Access Information:")
            if deployment_method == "fleet":
                logger.info("   🌐 GitOps Dashboard: Check your Git repository")
                logger.info("   📋 Fleet Status: kubectl get gitrepo distributed-ai-cluster -n fleet-default")
            else:
                logger.info("   🔗 Direct Access: kubectl port-forward -n distributed-ai svc/distributed-ai-blade-master-service 8080:8080")
                logger.info("   🌐 Then open: http://localhost:8080/dashboard")

            # Show monitoring access
            logger.info("\n📊 Monitoring:")
            logger.info(f"   kubectl port-forward -n {self.namespace} svc/distributed-ai-blade-prometheus-server 9090:80")
            logger.info(f"   kubectl port-forward -n {self.namespace} svc/distributed-ai-blade-grafana 3000:80")

        except Exception as e:
            logger.error(f"Failed to get deployment info: {e}")

    def deploy(self, method: str = "auto", registry: str = None, git_repo: str = None) -> bool:
        """Flexible deployment with method selection"""
        logger.info(f"🚀 Starting Flexible Deployment (method: {method})")
        logger.info("=" * 55)

        # Auto-detect best method if not specified
        if method == "auto":
            if self.check_fleet_status():
                method = "fleet"
                logger.info("✅ Fleet detected - using GitOps deployment")
            else:
                method = "direct"
                logger.info("ℹ️ Fleet not available - using direct Helm deployment")

        # Deploy based on selected method
        if method == "fleet":
            if not git_repo:
                logger.error("❌ Git repository URL required for Fleet deployment")
                logger.error("💡 Use: --git-repo https://github.com/your-username/distributed-ai-cluster.git")
                return False

            success = self.deploy_with_fleet(git_repo)
            deployment_method = "Fleet GitOps"

        elif method == "direct":
            success = self.deploy_direct_helm(registry)
            deployment_method = "Direct Helm"

        else:
            logger.error(f"❌ Unknown deployment method: {method}")
            return False

        if not success:
            return False

        # Wait for ready
        if not self.wait_for_ready():
            logger.warning("⚠️ Deployment may not be fully ready")

        # Show deployment info
        self.show_deployment_info(deployment_method)

        logger.info(f"🎉 {deployment_method} deployment completed successfully!")
        return True

def main():
    """Main deployment function"""
    parser = argparse.ArgumentParser(description="Flexible deployment for Distributed AI Cluster")
    parser.add_argument("--method", choices=["auto", "fleet", "direct"],
                       default="auto", help="Deployment method")
    parser.add_argument("--registry", help="Docker registry URL")
    parser.add_argument("--git-repo", help="Git repository URL for Fleet deployment")

    args = parser.parse_args()

    deployer = FlexibleDeployer()

    success = deployer.deploy(
        method=args.method,
        registry=args.registry,
        git_repo=args.git_repo
    )

    if success:
        logger.info("✅ Deployment successful!")
        return 0
    else:
        logger.error("❌ Deployment failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
