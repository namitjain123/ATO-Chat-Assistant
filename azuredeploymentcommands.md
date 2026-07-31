# Azure deployment — PowerShell commands
# Run these in PowerShell from e:\enterprise-grade-chatbot
# NOTE: keep the SAME terminal window open for all phases — the $variables
# below only live for the current session. If you close it, re-run the
# variable block (but the random names will change, so save them — see Phase B end).

# ─────────────────────────────────────────────────────────────
# Phase A — one-time setup (Container Apps extension + providers)
# ─────────────────────────────────────────────────────────────
az extension add --name containerapp --upgrade
az provider register --namespace Microsoft.App
az provider register --namespace Microsoft.OperationalInsights
az provider register --namespace Microsoft.DBforPostgreSQL

# ─────────────────────────────────────────────────────────────
# Phase B — variables + core resources
# ─────────────────────────────────────────────────────────────
$RG          = "chatbot-rg"
$LOCATION    = "australiaeast"   # eastus is blocked by the student-subscription region policy
$ACR_NAME    = "ragchatbott"   # must be globally unique
$ENV_NAME    = "chatbot-env"
$PG_SERVER   = "chatbot-pg-new"  
$PG_ADMIN    = "chatbotadmin"
$PG_PASSWORD = "Namitj@123"   # 8-128 chars, 3 of: upper/lower/digit/special. CHANGE THIS.

# already logged in as Azure for Students — skip if 'az account show' works
# az login

az group create --name $RG --location $LOCATION

# Container Registry (stores your Docker images)
az acr create --resource-group $RG --name $ACR_NAME --sku Basic

# Container Apps environment (shared networking/logging boundary)
az containerapp env create --name $ENV_NAME --resource-group $RG --location $LOCATION

# SAVE your generated names so you can reuse them in a new terminal later:
Write-Host "ACR_NAME  = $ACR_NAME"
Write-Host "PG_SERVER = $PG_SERVER"

# ─────────────────────────────────────────────────────────────
# Phase C — Postgres (the LangGraph checkpointer DB)
# ─────────────────────────────────────────────────────────────
az postgres flexible-server create `
  --resource-group $RG `
  --name $PG_SERVER `
  --location $LOCATION `
  --admin-user $PG_ADMIN `
  --admin-password $PG_PASSWORD `
  --sku-name Standard_B1ms `
  --tier Burstable `
  --storage-size 32 `
  --version 16 `
  --public-access 0.0.0.0

az postgres flexible-server db create `
  --resource-group $RG --server-name $PG_SERVER --database-name chatbot_checkpoints

# Build the connection string psycopg expects (used as POSTGRES_URL secret in Phase E)
$POSTGRES_URL = "postgresql://$($PG_ADMIN):$($PG_PASSWORD)@$($PG_SERVER).postgres.database.azure.com:5432/chatbot_checkpoints?sslmode=require"
Write-Host "POSTGRES_URL = $POSTGRES_URL"

# ─────────────────────────────────────────────────────────────
# Phase D — build & push the image in the cloud (no local Docker needed)
# ─────────────────────────────────────────────────────────────
az acr build `
  --registry $ACR_NAME `
  --image chatbot-backend:v1 `
  --file Dockerfile .
