# Azure infrastructure (Terraform)

Infrastructure-as-Code that provisions the Azure platform for
`distributed-ai-cluster`:

- **Resource group**
- **Azure Kubernetes Service (AKS)** cluster (system-assigned identity, Azure CNI)
- **Azure Container Registry (ACR)** with an `AcrPull` role assignment so AKS can
  pull images without image-pull secrets

State is stored remotely in Azure Blob Storage (configured at `init` time, see
below) so nothing sensitive lives in the repo.

## Layout

| File | Purpose |
| --- | --- |
| `versions.tf` | Provider + Terraform version constraints, remote backend |
| `providers.tf` | Provider configuration |
| `variables.tf` | Inputs (region, node size/count, ACR SKU, tags) |
| `main.tf` | Resource group, ACR, AKS, role assignment |
| `outputs.tf` | RG/cluster/ACR names + kubeconfig (sensitive) |
| `terraform.tfvars.example` | Copy to `terraform.tfvars` and edit |

## Prerequisites

- Azure CLI (`az login`) and Terraform >= 1.6
- An Azure subscription with permission to create resource groups and role
  assignments

## One-time: remote state backend

```bash
az group create -n tfstate-rg -l westeurope
az storage account create -n <globally-unique-name> -g tfstate-rg -l westeurope --sku Standard_LRS
az storage container create -n tfstate --account-name <globally-unique-name>
```

## Usage (local)

```bash
cd distributed-ai-cluster/infra/terraform
cp terraform.tfvars.example terraform.tfvars   # edit as needed

terraform init \
  -backend-config="resource_group_name=tfstate-rg" \
  -backend-config="storage_account_name=<globally-unique-name>" \
  -backend-config="container_name=tfstate" \
  -backend-config="key=distributed-ai-cluster.tfstate"

terraform plan
terraform apply
```

Then connect and deploy the chart:

```bash
az aks get-credentials \
  --resource-group "$(terraform output -raw resource_group_name)" \
  --name "$(terraform output -raw aks_cluster_name)"

helm upgrade --install distributed-ai ../../helm/distributed-ai-cluster \
  --namespace distributed-ai --create-namespace
```

## Verified live

This config has been applied to a real Azure subscription: it provisioned the
resource group, ACR (`Standard`), and a 2-node AKS cluster (`v1.34`,
`provisioningState: Succeeded`), with both nodes `Ready`. A test workload exposed
via a `LoadBalancer` received a public IP and served `HTTP 200`, and the
`AcrPull` role assignment was created so AKS can pull from ACR without secrets.

> **VM size note:** the default is `Standard_B2s_v2`. Some subscriptions (e.g.
> trial/MSDN) have **0 quota** for the `Bsv2` family or disallow legacy `Standard_B2s`
> entirely. Check `az vm list-usage --location <region>` and override
> `node_vm_size` (e.g. `Standard_D2s_v3`) if you hit a quota/availability error.

## CI/CD

- **`.github/workflows/terraform.yml`** — runs `fmt -check`, `init -backend=false`,
  and `validate` on every PR that touches this directory. No cloud credentials
  required, so it always runs for reviewers.
- **`.github/workflows/azure-deploy.yml`** — manual (`workflow_dispatch`) plan/apply
  that logs in with **Azure OIDC** (no stored secrets) and deploys the Helm chart
  to AKS.

### Configuring OIDC (keyless auth)

Create an app registration with a **federated credential** for this repo, grant it
`Contributor` on the subscription (and `User Access Administrator` for the role
assignment), then set these repo secrets:

```text
AZURE_CLIENT_ID
AZURE_TENANT_ID
AZURE_SUBSCRIPTION_ID
TFSTATE_RG
TFSTATE_STORAGE_ACCOUNT
TFSTATE_CONTAINER
```

Example app + federated credential:

```bash
az ad app create --display-name distributed-ai-cluster-gha
# capture appId, then:
az ad app federated-credential create --id <appId> --parameters '{
  "name": "gha-main",
  "issuer": "https://token.actions.githubusercontent.com",
  "subject": "repo:hoffeloffe/distributed-ai-cluster:ref:refs/heads/main",
  "audiences": ["api://AzureADTokenExchange"]
}'
```
