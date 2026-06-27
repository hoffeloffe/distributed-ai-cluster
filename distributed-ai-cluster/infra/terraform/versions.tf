terraform {
  required_version = ">= 1.6"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 4.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }

  # Remote state in Azure Blob Storage. Configured at init time so no secrets
  # live in source, e.g.:
  #   terraform init \
  #     -backend-config="resource_group_name=tfstate-rg" \
  #     -backend-config="storage_account_name=mytfstate" \
  #     -backend-config="container_name=tfstate" \
  #     -backend-config="key=distributed-ai-cluster.tfstate"
  backend "azurerm" {}
}
