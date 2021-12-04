# Databricks notebook source
# MAGIC %run ../src/utils

# COMMAND ----------

# Workspace config
workspace_name = dbutils.secrets.get(scope = "azure-key-vault", key = "workspace-name")
resource_group = dbutils.secrets.get(scope = "azure-key-vault", key = "resource-group")
subscription_id = dbutils.secrets.get(scope = "azure-key-vault", key = "subscription-id")

# Model
model_name = 'ChurnModel'
model_description = 'Model to predict churn'
model_path = '/dbfs/models/churn-prediction'

workspace = get_workspace(workspace_name, resource_group, subscription_id)

model_azure = register_model(workspace, model_name, model_description, model_path)
