$ErrorActionPreference = "Stop"

# --- CONFIG ---
$ResourceGroup = "COM774_CW2"
$Workspace     = "COM774_CW2"
$Cluster       = "CW2Cluster"
$EndpointName  = "bgl-endpoint-final" 
$Deployment    = "blue-deployment"

# 4 Cores for Training (Fast), 2 Cores for Inference (Fits remaining quota)
$TrainingVM    = "Standard_DS3_v2" 
$InferenceVM   = "Standard_DS2_v2" 

Write-Host ">>> [START] Hybrid Pipeline Initiated..." -ForegroundColor Cyan

# 1. CREATE CLUSTER (Ephemeral)
Write-Host ">>> [1/5] Provisioning Training Cluster ($TrainingVM)..." -ForegroundColor Yellow
az ml compute create --name $Cluster --resource-group $ResourceGroup --workspace-name $Workspace --type amlcompute --size $TrainingVM --min-instances 0 --max-instances 1

# 2. TRAIN
Write-Host ">>> [2/5] Running Training Job..." -ForegroundColor Yellow

# Submit job and capture its name
$JobName = az ml job create `
    --file job.yaml `
    --resource-group $ResourceGroup `
    --workspace-name $Workspace `
    --query name `
    -o tsv

Write-Host ">>> Job submitted: $JobName" -ForegroundColor Green

# --- ADD THIS: BLOCK UNTIL JOB COMPLETES ---
Write-Host ">>> Waiting for training to complete..." -ForegroundColor Cyan
az ml job stream --name $JobName --resource-group $ResourceGroup --workspace-name $Workspace
# -------------------------------------------

# 3. REGISTER
$ModelName = "bgl-anomaly-rf"
# NOTE: model is under outputs/model → so artifact path is outputs/model
$ModelPath = "azureml://jobs/$JobName/outputs/artifacts/paths/outputs/model"

az ml model create `
    --name $ModelName `
    --path $ModelPath `
    --type mlflow_model `
    --resource-group $ResourceGroup `
    --workspace-name $Workspace


# 4. DESTROY CLUSTER (Releases 4 Cores)
Write-Host ">>> [4/5] DESTROYING CLUSTER to release quota..." -ForegroundColor Red
az ml compute delete --name $Cluster --resource-group $ResourceGroup --workspace-name $Workspace --yes
Write-Host ">>> Cluster Deleted. Waiting 60s for quota sync..." -ForegroundColor Cyan
Start-Sleep -Seconds 60

# 5. DEPLOY ENDPOINT (Uses 2 Cores)
Write-Host ">>> [5/5] Deploying Endpoint ($InferenceVM)..." -ForegroundColor Yellow

# Ensure Endpoint Shell Exists
az ml online-endpoint create --name $EndpointName --auth-mode key --resource-group $ResourceGroup --workspace-name $Workspace

# Get Latest Version
$LatestVersion = az ml model list --name $ModelName --resource-group $ResourceGroup --workspace-name $Workspace --query "[0].version" --output tsv

# Config
$DeployConfig = @"
`$schema: https://azuremlschemas.azureedge.net/latest/managedOnlineDeployment.schema.json
name: $Deployment
endpoint_name: $EndpointName
model: "azureml:$($ModelName):$LatestVersion"
instance_type: $InferenceVM
instance_count: 1
"@
Set-Content -Path "deploy_config.yaml" -Value $DeployConfig -Encoding UTF8

# Deploy
az ml online-deployment create --file "deploy_config.yaml" --all-traffic --resource-group $ResourceGroup --workspace-name $Workspace
Remove-Item "deploy_config.yaml"

Write-Host ">>> SUCCESS. Pipeline Complete." -ForegroundColor Cyan