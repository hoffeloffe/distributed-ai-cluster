variable "prefix" {
  description = "Short prefix used to name all resources (lowercase letters/digits)."
  type        = string
  default     = "distai"

  validation {
    condition     = can(regex("^[a-z][a-z0-9]{2,11}$", var.prefix))
    error_message = "prefix must be 3-12 chars, lowercase alphanumeric, starting with a letter."
  }
}

variable "location" {
  description = "Azure region to deploy into."
  type        = string
  default     = "westeurope"
}

variable "kubernetes_version" {
  description = "AKS Kubernetes version. Leave null to use the region default."
  type        = string
  default     = null
}

variable "node_count" {
  description = "Number of nodes in the default (system) node pool."
  type        = number
  default     = 2
}

variable "node_vm_size" {
  description = "VM size for the default node pool."
  type        = string
  default     = "Standard_B2s_v2"
}

variable "acr_sku" {
  description = "Azure Container Registry SKU."
  type        = string
  default     = "Standard"
}

variable "tags" {
  description = "Tags applied to all resources."
  type        = map(string)
  default = {
    project = "distributed-ai-cluster"
    managed = "terraform"
  }
}
