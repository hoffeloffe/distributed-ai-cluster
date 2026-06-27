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

> **VM size note:** the default is `Standard_D2s_v3` (it has quota on this
> subscription and is broadly available). Some subscriptions (e.g. trial/MSDN)
> have **0 quota** for the `Bsv2` family or disallow legacy `Standard_B2s`. If you
> want a cheaper burstable SKU where quota allows, set `node_vm_size` to
> `Standard_B2s_v2`. Check availability with `az vm list-usage --location <region>`.

## CI/CD

- **`.github/workflows/terraform.yml`** — runs `fmt -check`, `init -backend=false`,
  and `validate` on every PR that touches this directory. No cloud credentials
  required, so it always runs for reviewers.
- **`.github/workflows/azure-deploy.yml`** — manual (`workflow_dispatch`) Terraform
  `plan`/`apply` that logs in with **Azure OIDC** (no stored secrets). Runs against
  the `azure` environment (add required reviewers there for an approval gate).
  Optional `deploy_app` input runs `helm upgrade` (needs the chart + app image fixed).
- **`.github/workflows/acr-build.yml`** — builds the container and pushes it to ACR
  tagged by commit SHA (keyless via OIDC).

### Configuring OIDC (keyless auth)

The pipeline authenticates with a **federated credential** instead of a stored
secret. Set these on the repo (only IDs — none are secret values):

```text
Secrets:    AZURE_CLIENT_ID, AZURE_TENANT_ID, AZURE_SUBSCRIPTION_ID,
            TFSTATE_RG, TFSTATE_STORAGE_ACCOUNT, TFSTATE_CONTAINER
Variables:  ACR_LOGIN_SERVER   (e.g. distaiacrt6g0m.azurecr.io)
```

The identity needs `Contributor` (+ `User Access Administrator` for the AcrPull
role assignment) on the deploy scope, and `Storage Blob Data Contributor` on the
state storage account (the backend uses `use_azuread_auth=true`, so state access is
also keyless).

Create the app + federated credentials (one per trusted subject):

```bash
az ad app create --display-name distributed-ai-cluster-gha
APP_ID=$(az ad app list --display-name distributed-ai-cluster-gha --query "[0].appId" -o tsv)
az ad sp create --id "$APP_ID"

for SUB in \
  "repo:hoffeloffe/distributed-ai-cluster:ref:refs/heads/main" \
  "repo:hoffeloffe/distributed-ai-cluster:environment:azure"; do
  az ad app federated-credential create --id "$APP_ID" --parameters "{
    \"name\": \"gha-$(echo "$SUB" | tr ':/' '-')\",
    \"issuer\": \"https://token.actions.githubusercontent.com\",
    \"subject\": \"$SUB\",
    \"audiences\": [\"api://AzureADTokenExchange\"]
  }"
done
```
