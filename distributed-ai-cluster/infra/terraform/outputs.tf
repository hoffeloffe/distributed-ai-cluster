output "resource_group_name" {
  description = "Name of the resource group."
  value       = azurerm_resource_group.this.name
}

output "aks_cluster_name" {
  description = "Name of the AKS cluster (use with `az aks get-credentials`)."
  value       = azurerm_kubernetes_cluster.this.name
}

output "acr_login_server" {
  description = "ACR login server, e.g. distaiacrxxxxx.azurecr.io."
  value       = azurerm_container_registry.this.login_server
}

output "kube_config_raw" {
  description = "Raw kubeconfig for the cluster."
  value       = azurerm_kubernetes_cluster.this.kube_config_raw
  sensitive   = true
}
