# -----------------------------------------------------------------
# MASTER PIPELINE: Train -> Register -> Destroy -> Deploy
# -----------------------------------------------------------------
$ErrorActionPreference = "Stop"

# --- CONFIGURATION ---
$ResourceGroup = "EamonnUniversitty_COM774"
$Workspace     = "COM774_CW2"
$Cluster       = "CW2Cluster"
$JobFile       = "job.yaml"
$EndpointName  = "bgl-endpoint-final" 
$Deployment    = "blue-deployment"

# VM SIZING STRATEGY:
# Training: DS3_v2 (4 Cores) -> Fast training.
# Inference: DS3_v2 (4 Cores) -> High RAM (14GB) for stability.
# Quota Logic: 4 (Train) -> Destroy -> 0 -> 4 (Deploy). Max usage = 4. (Fits in 6).
$TrainingVM    = "Standard_DS3_v2" 
$InferenceVM   = "Standard_DS2_v2" 

Write-Host ">>> [START] Pipeline Initiated..." -ForegroundColor Cyan

# -----------------------------------------------------------------
# STEP 0: PROVISION TRAINING INFRASTRUCTURE
# -----------------------------------------------------------------
Write-Host ">>> [0/4] Provisioning Ephemeral Training Cluster..." -ForegroundColor Yellow
az ml compute create --name $Cluster `
  --resource-group $ResourceGroup `
  --workspace-name $Workspace `
  --type amlcompute `
  --size $TrainingVM `
  --min-instances 0 `
  --max-instances 1

# -----------------------------------------------------------------
# STEP 1: TRAIN
# -----------------------------------------------------------------
Write-Host ">>> [1/4] Submitting Training Job..." -ForegroundColor Yellow
az ml job create --file $JobFile --resource-group $ResourceGroup --workspace-name $Workspace --stream

if ($LASTEXITCODE -ne 0) { 
    Write-Error "Training Failed. Pipeline Aborted."
    exit 1 
}

# Retrieve Run ID
$JobName = az ml job list --resource-group $ResourceGroup --workspace-name $Workspace --query "[0].name" --output tsv
Write-Host ">>> Training Complete. Run ID: $JobName" -ForegroundColor Green

# -----------------------------------------------------------------
# STEP 2: REGISTER MODEL
# -----------------------------------------------------------------
Write-Host ">>> [2/4] Registering Model Artifact..." -ForegroundColor Yellow

$ModelName = "bgl-anomaly-rf"
$ModelPath = "azureml://jobs/$JobName/outputs/artifacts/paths/model"

# Auto-increment version
az ml model create --name $ModelName --path $ModelPath --type mlflow_model --resource-group $ResourceGroup --workspace-name $Workspace

if ($LASTEXITCODE -ne 0) { 
    Write-Error "Model Registration Failed. Aborting."
    exit 1 
}

# -----------------------------------------------------------------
# STEP 3: DESTROY CLUSTER (Split Lifecycle)
# -----------------------------------------------------------------
Write-Host ">>> [3/4] RELEASING QUOTA: Destroying Training Cluster..." -ForegroundColor Red
# This is critical. We must free up the 4 cores so we can use them for the Endpoint.
az ml compute delete --name $Cluster --resource-group $ResourceGroup --workspace-name $Workspace --yes
Write-Host ">>> Quota Released. Waiting 60s for backend sync..." -ForegroundColor Cyan
Start-Sleep -Seconds 60

# -----------------------------------------------------------------
# STEP 4: DEPLOY ENDPOINT
# -----------------------------------------------------------------
Write-Host ">>> [4/4] Provisioning Managed Online Endpoint..." -ForegroundColor Yellow

# Check Endpoint State
$OldEAP = $ErrorActionPreference
$ErrorActionPreference = "Continue" 
$EndpointExists = $false
az ml online-endpoint show --name $EndpointName --resource-group $ResourceGroup --workspace-name $Workspace > $null 2>&1
if ($LASTEXITCODE -eq 0) { $EndpointExists = $true }
$ErrorActionPreference = $OldEAP 

if (-not $EndpointExists) {
    Write-Host ">>> Creating new Endpoint Shell..."
    az ml online-endpoint create --name $EndpointName --auth-mode key --resource-group $ResourceGroup --workspace-name $Workspace
}

# --- FIX: GET LATEST MODEL VERSION ---
Write-Host ">>> Looking up latest model version..."
$LatestVersion = az ml model list --name $ModelName --resource-group $ResourceGroup --workspace-name $Workspace --query "[0].version" --output tsv
Write-Host ">>> Deploying Model Version: $LatestVersion to VM: $InferenceVM" -ForegroundColor Yellow

$DeployConfig = @"
`$schema: https://azuremlschemas.azureedge.net/latest/managedOnlineDeployment.schema.json
name: $Deployment
endpoint_name: $EndpointName
model: azureml:$($ModelName):$LatestVersion
instance_type: $InferenceVM
instance_count: 1
"@

Set-Content -Path "deploy_config.yaml" -Value $DeployConfig -Encoding UTF8

# Create deployment (or update if exists). Using --all-traffic to switch immediately.
az ml online-deployment create --file "deploy_config.yaml" --all-traffic --resource-group $ResourceGroup --workspace-name $Workspace

Remove-Item "deploy_config.yaml"

Write-Host ">>> SUCCESS: Pipeline Finished. REST API is Live." -ForegroundColor Cyan