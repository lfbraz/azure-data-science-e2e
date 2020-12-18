# Databricks notebook source
# MAGIC %run /Shared/churn-model/utils

# COMMAND ----------

# Workspace config
workspace_name = 'e2e-datascience-aml'
resource_group = 'RG-E2E-DATA-SCIENCE'
subscription_id = 'f56912be-98e5-44e3-9e64-54bc52cef4a7'

# Model
model_name = 'churn-model'
model_description = 'Model to predict churn'
model_path = '/dbfs/models/churn-prediction'

# Environment
environment_name = 'xgboost_env'
conda_dep_file = '/dbfs/models/churn-prediction/conda.yaml'
entry_script = '/dbfs/models/churn-prediction/score.py'

#Endpoint - DEV
endpoint_name = 'api-churn-dev'
print()

workspace = get_workspace(workspace_name, resource_group, subscription_id)

model_azure = register_model(workspace, model_name, model_description, model_path)

inference_config = get_inference_config(environment_name, conda_dep_file, entry_script)
service = deploy_aci(workspace, model_azure, endpoint_name, inference_config)

# COMMAND ----------

