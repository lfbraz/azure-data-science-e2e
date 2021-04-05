# Databricks notebook source
# MAGIC %run /Shared/churn-model/utils

# COMMAND ----------

# Workspace config
workspace_name = '<YOUR-WORKSPACE>'
resource_group = '<YOUR-RESOURCE-GROUP>'
subscription_id = '<YOUR-SUBSCRIPTION-ID>'

# Model
model_name = 'churn-model'
model_description = 'Model to predict churn'
model_path = '/dbfs/models/churn-prediction'

# Environment
environment_name = 'xgboost_env'
conda_dep_file = '/dbfs/models/churn-prediction/conda.yaml'
entry_script = '/dbfs/models/churn-prediction/score.py'

#Endpoint - PROD
endpoint_name = 'api-churn-prod'
aks_name = 'aks-e2e-1'

workspace = get_workspace(workspace_name, resource_group, subscription_id)

model_azure = register_model(workspace, model_name, model_description, model_path)

inference_config = get_inference_config(environment_name, conda_dep_file, entry_script)
service = deploy_aks(workspace, model_azure, endpoint_name, inference_config, aks_name)