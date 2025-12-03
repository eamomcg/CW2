# -----------------------------------------------------------------
# COM774 CW2: Local MLOps Pipeline (PowerShell Edition)
# -----------------------------------------------------------------
# Validated for "Azure for Students" constraints.
# Replaces GitHub Actions due to Permission Restrictions.

# Stop on first error to prevent cascading failures
$ErrorActionPreference = "Stop"

# --- CONFIGURATION (CHECK THESE VALUES) ---
$ResourceGroup = "EamonnUniversitty_COM774"
$Workspace     = "COM774_CW2"
$Cluster       = "CW2Cluster"
$JobFile       = "job.yaml"

# Randomize endpoint name slightly to ensure uniqueness if you run it multiple times
$EndpointName  = "bgl-endpoint-v" + (Get-Random -Minimum 100 -Maximum 999)
$Deployment    = "blue-deployment"
$InstanceType  = "Standard_DS2_v2"           # <--- Change if you hit Quota limits

Write-Host ">>> [1/4] Starting Pipeline Execution..." -ForegroundColor Cyan

# -----------------------------------------------------------------
# STEP 1: Submit Training Job
# -----------------------------------------------------------------
Write-Host ">>> Submitting Training Job to $Cluster..." -ForegroundColor Yellow

# We use --stream to block the script until training finishes (Simulating CI/CD blocking)
$JobRun = az ml job create --file $JobFile --resource-group $ResourceGroup --workspace-name $Workspace --stream

if ($LASTEXITCODE -ne 0) { Write-Error "Training failed. Stopping pipeline."; exit 1 }

# Retrieve the Job Name (Run ID) from the last run to use in registration
# Note: We query the 'latest' job to get the ID we just finished
$JobName = az ml job list --resource-group $ResourceGroup --workspace-name $Workspace --query "[0].name" --output tsv

Write-Host ">>> Job Success. Run ID: $JobName" -ForegroundColor Green

# -----------------------------------------------------------------
# STEP 2: Register Model
# -----------------------------------------------------------------
Write-Host ">>> Registering Model..." -ForegroundColor Yellow

$ModelName = "bgl-anomaly-rf"
# Standard MLflow path in Azure. If this fails, check the "Outputs" tab in Azure Portal.
$ModelPath = "azureml://jobs/$JobName/outputs/artifacts/paths/model"

az ml model create --name $ModelName --path $ModelPath --type mlflow_model --resource-group $ResourceGroup --workspace-name $Workspace

Write-Host ">>> Model Registered: $ModelName" -ForegroundColor Green

# -----------------------------------------------------------------
# STEP 3: Create Endpoint
# -----------------------------------------------------------------
Write-Host ">>> Creating Managed Endpoint: $EndpointName..." -ForegroundColor Yellow

az ml online-endpoint create --name $EndpointName --auth-mode key --resource-group $ResourceGroup --workspace-name $Workspace

# -----------------------------------------------------------------
# STEP 4: Deploy Model
# -----------------------------------------------------------------
Write-Host ">>> Deploying Model to Hardware ($InstanceType)..." -ForegroundColor Yellow

# Dynamically create the deployment YAML to point to the model we just registered
$DeployConfig = @"
`$schema: https://azuremlschemas.azureedge.net/latest/managedOnlineDeployment.schema.json
name: $Deployment
endpoint_name: $EndpointName
model: azureml:$($ModelName):1
instance_type: $InstanceType
instance_count: 1
"@

# Save temp config
Set-Content -Path "temp_deploy.yaml" -Value $DeployConfig -Encoding UTF8

# Execute Deployment
az ml online-deployment create --file "temp_deploy.yaml" --all-traffic --resource-group $ResourceGroup --workspace-name $Workspace

# Cleanup temp file
Remove-Item "temp_deploy.yaml"

Write-Host ">>> Pipeline Complete! Endpoint is live." -ForegroundColor Cyan
Write-Host ">>> Test URL and Keys available in Azure Portal -> Endpoints."