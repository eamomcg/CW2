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

# FIX: Removed '--force'. Azure will auto-increment version (e.g. 1 -> 2)
az ml model create --name $ModelName --path $ModelPath --type mlflow_model --resource-group $ResourceGroup --workspace-name $Workspace

if ($LASTEXITCODE -ne 0) { 
    Write-Error "Model Registration Failed. Aborting before cluster destruction."
    exit 1 
}

# -----------------------------------------------------------------
# STEP 3: DESTROY CLUSTER (Split Lifecycle)
# -----------------------------------------------------------------
Write-Host ">>> [3/4] RELEASING QUOTA: Destroying Training Cluster..." -ForegroundColor Red
az ml compute delete --name $Cluster --resource-group $ResourceGroup --workspace-name $Workspace --yes
Write-Host ">>> Quota Released." -ForegroundColor Green

# -----------------------------------------------------------------
# STEP 4: DEPLOY ENDPOINT
# -----------------------------------------------------------------
Write-Host ">>> [4/4] Provisioning Managed Online Endpoint..." -ForegroundColor Yellow

# FIX: Safely check for endpoint existence without crashing on warnings
$OldEAP = $ErrorActionPreference
$ErrorActionPreference = "Continue" # Temporarily allow warnings
$EndpointExists = $false

# Try to show the endpoint. If it works (Exit Code 0), it exists.
az ml online-endpoint show --name $EndpointName --resource-group $ResourceGroup --workspace-name $Workspace > $null 2>&1
if ($LASTEXITCODE -eq 0) { $EndpointExists = $true }

$ErrorActionPreference = $OldEAP # Reset error handling

if (-not $EndpointExists) {
    Write-Host ">>> Creating new Endpoint: $EndpointName"
    az ml online-endpoint create --name $EndpointName --auth-mode key --resource-group $ResourceGroup --workspace-name $Workspace
} else {
    Write-Host ">>> Endpoint exists. Skipping creation."
}

# Deploy
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
az ml online-deployment create --file "deploy_config.yaml" --all-traffic --resource-group $ResourceGroup --workspace-name $Workspace
Remove-Item "deploy_config.yaml"

Write-Host ">>> SUCCESS: Pipeline Finished. REST API is Live." -ForegroundColor Cyan