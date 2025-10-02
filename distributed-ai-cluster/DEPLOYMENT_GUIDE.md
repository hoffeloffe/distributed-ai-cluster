# 🚀 Distributed AI Cluster - Production Deployment Guide

## 🎯 Deployment Options for Your Blade Cluster

You now have **two excellent deployment strategies** for your distributed AI cluster. Choose based on your needs:

---

## **Option 1: Fleet GitOps (Recommended for Production)** ⭐

### **Why Fleet GitOps?**
- ✅ **Infrastructure as Code** - Everything in Git
- ✅ **Automatic Sync** - Changes deploy automatically
- ✅ **Rollback Support** - Easy rollbacks to previous versions
- ✅ **Multi-Cluster** - Manage multiple clusters from one repo
- ✅ **Version Control** - Full history of all changes
- ✅ **Rancher Integration** - Native Rancher GitOps support

### **Quick Setup:**
```bash
# 1. Push your code to GitHub
git add .
git commit -m "Initial distributed AI cluster"
git push origin main

# 2. Deploy with Fleet
python3 scripts/flexible_deployment.py \
  --method fleet \
  --git-repo https://github.com/your-username/distributed-ai-cluster.git

# 3. Monitor deployment
kubectl get gitrepo distributed-ai-cluster -n fleet-default -w
```

### **Fleet Structure:**
```
your-repo/
├── fleet-gitops-setup.yaml    # Fleet configuration
├── helm/
│   └── distributed-ai-cluster/  # Your Helm chart
└── manifests/
    ├── base/                  # Shared configuration
    └── overlays/
        └── production/        # Environment-specific
```

---

## **Option 2: Direct Helm (Great for Development)**

### **Why Direct Helm?**
- ✅ **Simple Setup** - One-command deployment
- ✅ **Immediate Results** - No Git repository needed
- ✅ **Easy Testing** - Quick iteration cycles
- ✅ **Direct Control** - Immediate deployment changes

### **Quick Setup:**
```bash
# 1. Build and push Docker image
docker build -t your-registry.com/distributed-ai-cluster:latest .
docker push your-registry.com/distributed-ai-cluster:latest

# 2. Deploy directly
python3 scripts/flexible_deployment.py \
  --method direct \
  --registry your-registry.com
```

---

## 🎯 **Which Should You Choose?**

| Scenario | Recommendation | Why? |
|----------|---------------|------|
| **Just Learning** | Direct Helm | Simpler to understand and debug |
| **Development** | Direct Helm | Fast iteration, immediate feedback |
| **Production** | **Fleet GitOps** | Version control, rollbacks, automation |
| **Multiple Clusters** | **Fleet GitOps** | Single source of truth for all clusters |
| **Team Collaboration** | **Fleet GitOps** | Git-based workflow for teams |

---

## 🚀 **Complete Setup Guide**

### **Prerequisites:**
```bash
# 1. Ensure kubectl points to your Blade cluster
kubectl cluster-info

# 2. Install Helm (if not already installed)
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# 3. For Fleet: Install Fleet in your cluster
kubectl apply -f https://github.com/rancher/fleet/releases/latest/download/fleet-crd.yaml
kubectl apply -f https://github.com/rancher/fleet/releases/latest/download/fleet.yaml
```

### **Option A: Fleet GitOps Setup**

```bash
# 1. Create GitHub repository
# 2. Push your code:
git init
git add .
git commit -m "Initial distributed AI cluster with Fleet GitOps"
git remote add origin https://github.com/your-username/distributed-ai-cluster.git
git push -u origin main

# 3. Deploy with Fleet
python3 scripts/flexible_deployment.py \
  --method fleet \
  --git-repo https://github.com/your-username/distributed-ai-cluster.git

# 4. Monitor
kubectl get gitrepo distributed-ai-cluster -n fleet-default
kubectl get pods -n distributed-ai -w
```

### **Option B: Direct Helm Setup**

```bash
# 1. Build Docker image
docker build -t your-registry.com/distributed-ai-cluster:latest .
docker push your-registry.com/distributed-ai-cluster:latest

# 2. Deploy directly
python3 scripts/flexible_deployment.py \
  --method direct \
  --registry your-registry.com

# 3. Check status
kubectl get pods -n distributed-ai
kubectl get svc -n distributed-ai
```

---

## 📊 **Access Your Deployment**

### **After Deployment:**

```bash
# Check all resources
kubectl get all,ingress,svc,pvc -n distributed-ai

# Access dashboard (choose one method):

# Method 1: Port forwarding
kubectl port-forward -n distributed-ai svc/distributed-ai-blade-master-service 8080:8080
# Then open: http://localhost:8080/dashboard

# Method 2: If using ingress (Fleet deployment)
# Access via your configured domain: http://ai.your-domain.com/dashboard

# Access monitoring
kubectl port-forward -n distributed-ai svc/distributed-ai-blade-grafana 3000:80
# Then open: http://localhost:3000 (admin/admin)

kubectl port-forward -n distributed-ai svc/distributed-ai-blade-prometheus-server 9090:80
# Then open: http://localhost:9090
```

---

## 🔧 **Scaling and Management**

### **Scale Your Workers:**
```bash
# Scale to 4 workers (use all your nodes)
kubectl scale deployment distributed-ai-blade-worker -n distributed-ai --replicas=4

# Check resource usage
kubectl top nodes
kubectl top pods -n distributed-ai
```

### **Update Configuration:**
```bash
# For Direct Helm - update and redeploy
helm upgrade distributed-ai-blade ./helm/distributed-ai-cluster \
  -n distributed-ai -f temp/direct-deployment-values.yaml

# For Fleet - push changes to Git and they auto-deploy
git add .
git commit -m "Updated worker configuration"
git push origin main
# Fleet automatically syncs changes!
```

---

## 🎯 **Production Readiness Features**

Your deployment includes:

### **📊 Monitoring & Alerting:**
- **Prometheus metrics** exported automatically
- **Grafana dashboards** with 6 monitoring panels
- **Intelligent alerts** for performance issues
- **Resource optimization** recommendations

### **🔒 Security:**
- **RBAC** (Role-Based Access Control)
- **Network policies** for traffic control
- **Security contexts** for pod security
- **Image pull secrets** for private registries

### **⚡ Performance:**
- **Advanced load balancing** (5 algorithms)
- **Model quantization** for size optimization
- **Caching** for improved response times
- **Auto-scaling** based on resource usage

---

## 🚨 **Troubleshooting**

### **Common Issues:**

```bash
# Check pod status
kubectl describe pods -n distributed-ai

# Check logs
kubectl logs -n distributed-ai deployment/distributed-ai-blade-master

# Check events
kubectl get events -n distributed-ai --sort-by=.metadata.creationTimestamp

# Check resource usage
kubectl top pods -n distributed-ai
kubectl top nodes
```

### **If Something Goes Wrong:**

**Direct Helm Deployment:**
```bash
# Check Helm status
helm list -n distributed-ai
helm status distributed-ai-blade -n distributed-ai

# Rollback if needed
helm rollback distributed-ai-blade 1 -n distributed-ai
```

**Fleet GitOps Deployment:**
```bash
# Check Fleet status
kubectl describe gitrepo distributed-ai-cluster -n fleet-default

# Check Fleet logs
kubectl logs -n cattle-fleet-system deployment/fleet-controller

# Force refresh
kubectl patch gitrepo distributed-ai-cluster -n fleet-default --type merge -p '{"metadata":{"annotations":{"fleet.cattle.io/latest-commit":"'"$(git rev-parse HEAD)"'"}}}'
```

---

## 🎉 **Next Steps**

1. **Choose your deployment method** (Fleet for production, Direct for development)
2. **Set up your Docker registry** if you haven't already
3. **Deploy your cluster** using the appropriate script
4. **Test the inference API** with sample images
5. **Monitor performance** using the Grafana dashboards
6. **Scale up** your worker nodes as needed

---

## 💼 **Portfolio Value**

This deployment demonstrates:

✅ **DevOps Excellence** - Kubernetes, Helm, GitOps, Rancher
✅ **Production Engineering** - Monitoring, security, scalability
✅ **System Architecture** - Distributed systems, microservices
✅ **AI/ML Operations** - Model deployment, inference optimization
✅ **Automation** - Infrastructure as code, CI/CD principles

**This is enterprise-grade deployment engineering!** 🚀
