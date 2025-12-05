# -----------------------------------------------------------------
# MASTER PIPELINE: Train -> Register -> Destroy -> Deploy
# -----------------------------------------------------------------
# Implements the "Split Lifecycle" pattern to fit within 6 vCPU quota.

$ErrorActionPreference = "Stop"

# --- CONFIGURATION ---
$ResourceGroup = "EamonnUniversitty_COM774"
$Workspace     = "COM774_CW2"
$Cluster       = "CW2Cluster"
$JobFile       = "job.yaml"
# Endpoint name must be unique in the region
$EndpointName  = "bgl-endpoint-final" 
$Deployment    = "blue-deployment"
# Training = Standard_DS3_v2 (4 cores)
# Inference = Standard_DS2_v2 (2 cores)
# Total Required = 6 cores (Fits quota ONLY if run sequentially)
$TrainingVM    = "Standard_DS3_v2"  # Change this to scale Training
$InferenceVM   = "Standard_DS2_v2"  # Change this to scale the Endpoint

Write-Host ">>> [START] Pipeline Initiated..." -ForegroundColor Cyan

# -----------------------------------------------------------------
# STEP 0: PROVISION TRAINING INFRASTRUCTURE
# -----------------------------------------------------------------
Write-Host ">>> [0/4] Provisioning Ephemeral Training Cluster..." -ForegroundColor Yellow

# Create/Update cluster with min-instances=0 to save costs/quota when idle
az ml compute create --name $Cluster `
  --resource-group $ResourceGroup `
  --workspace-name $Workspace `
  --type amlcompute `
  --size $TrainingVM `
  --min-instances 0 `
  --max-instances 1

# -----------------------------------------------------------------
# STEP 1: TRAIN (WITH QUALITY GATE)
# -----------------------------------------------------------------
Write-Host ">>> [1/4] Submitting Training Job..." -ForegroundColor Yellow

# Submit and Stream logs. If Python script fails (F1 < 0.20), this throws an error.
az ml job create --file $JobFile --resource-group $ResourceGroup --workspace-name $Workspace --stream

if ($LASTEXITCODE -ne 0) { 
    Write-Error "Training Failed or Quality Gate (F1 < 0.20) triggered. Pipeline Aborted."
    exit 1 
}

# Retrieve the Run ID of the successful job
$JobName = az ml job list --resource-group $ResourceGroup --workspace-name $Workspace --query "[0].name" --output tsv
Write-Host ">>> Training Complete. Run ID: $JobName" -ForegroundColor Green

# -----------------------------------------------------------------
# STEP 2: REGISTER MODEL
# -----------------------------------------------------------------
Write-Host ">>> [2/4] Registering Model Artifact..." -ForegroundColor Yellow

$ModelName = "bgl-anomaly-rf"
$ModelPath = "azureml://jobs/$JobName/outputs/artifacts/paths/model"

az ml model create --name $ModelName --path $ModelPath --type mlflow_model --resource-group $ResourceGroup --workspace-name $Workspace --force

# -----------------------------------------------------------------
# STEP 3: DESTROY CLUSTER TO RELEASE QUOTA
# -----------------------------------------------------------------
Write-Host ">>> [3/4] RELEASING QUOTA: Destroying Training Cluster..." -ForegroundColor Red

# We MUST delete the 4-core cluster to allow the 2-core AZURE Endpoint to spin up.
# Without this, there will be insufficient memory/quota to deploy the inference VM.
az ml compute delete --name $Cluster --resource-group $ResourceGroup --workspace-name $Workspace --yes

Write-Host ">>> Quota Released." -ForegroundColor Green

# -----------------------------------------------------------------
# STEP 4: DEPLOY ENDPOINT
# -----------------------------------------------------------------
Write-Host ">>> [4/4] Provisioning Managed Online Endpoint..." -ForegroundColor Yellow

# Create Endpoint (The "Shell" - Costs nothing until deployment added)
# We check if it exists first to avoid errors
$EndpointExists = az ml online-endpoint show --name $EndpointName --resource-group $ResourceGroup --workspace-name $Workspace 2>$null
if (-not $EndpointExists) {
    az ml online-endpoint create --name $EndpointName --auth-mode key --resource-group $ResourceGroup --workspace-name $Workspace
}

# Deploy the Model (This provisions the VM and eats the quota)
Write-Host ">>> Deploying Model to Inference VM ($InferenceVM)..."

$DeployConfig = @"
`$schema: https://azuremlschemas.azureedge.net/latest/managedOnlineDeployment.schema.json
name: $Deployment
endpoint_name: $EndpointName
model: azureml:$($ModelName):1
instance_type: $InferenceVM
instance_count: 1
"@

Set-Content -Path "deploy_config.yaml" -Value $DeployConfig -Encoding UTF8

# This step takes 6-10 minutes
az ml online-deployment create --file "deploy_config.yaml" --all-traffic --resource-group $ResourceGroup --workspace-name $Workspace

Remove-Item "deploy_config.yaml"

Write-Host ">>> SUCCESS: Pipeline Finished. REST API is Live." -ForegroundColor Cyan