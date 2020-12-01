# Databricks notebook source
# MAGIC %md
# MAGIC ## Deploy churn prediction model
# MAGIC In this notebook we will demonstrate how to get the model generated [here]() to deploy it. We need to follow these steps:
# MAGIC 
# MAGIC - Get an already trained model
# MAGIC - Instantiate an Azure ML Workspace
# MAGIC - Build an image with the best model packaged
# MAGIC - Deploy the model to ACI (Azure Container Instance)
# MAGIC - Deploy the model to AKS (Azure Kubernetes Services)

# COMMAND ----------

# MAGIC %md
# MAGIC ## First lets get the model
# MAGIC Return the best model from `churn-prediction` experiment. We will use the same notebook **model-churn-prediction** and return the `model_uri`.

# COMMAND ----------

# MAGIC %run ./model-churn-prediction

# COMMAND ----------

# MAGIC %md And load the `xgboost` using the `model_uri` returned from MLFlow tracking.

# COMMAND ----------

import mlflow

model = mlflow.xgboost.load_model(model_uri)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Get Azure Machine Learning Workspace
# MAGIC We will use Azure Machine Learning to deliver the API `endpoints` that will consume the Machine Learning models. To be able to interact with Azure ML we will use [Azure Machine Learning Python SDK](https://docs.microsoft.com/en-us/python/api/overview/azure/ml/?view=azure-ml-py), with it its possible to create new workspaces (or use existing ones) to facilitate the deployment process.
# MAGIC 
# MAGIC Its required to fill the variables `WORKSPACE_NAME`, `WORKSPACE_LOCATION`, `RESOURCE_GROUP` and `SUBSCRIPTION_ID` with your subscription data.
# MAGIC 
# MAGIC As default will be required the `Interactive Login` auth. For production scenarios an app registration with `Service Principal` is required. In the [documentation] (https://docs.microsoft.com/en-us/azure/machine-learning/how-to-setup-authentication#set-up-service-principal-authentication) we have more details about the different kind of authentications.

# COMMAND ----------

# MAGIC %md First install the [`azureml-sdk`](https://pypi.org/project/azureml-sdk/)

# COMMAND ----------

# MAGIC %md And now we can use it to instantiate the Azure ML Workspace

# COMMAND ----------

import azureml
from azureml.core import Workspace
import mlflow.azureml

workspace_name = '<YOUR-WORKSPACE-NAME>'
resource_group = '<YOUR-RESOURCE-GROUP>'
subscription_id = '<YOUR-SUBSCRIPTION-ID>'

workspace = Workspace.get(name = workspace_name,
                          resource_group = resource_group,
                          subscription_id = subscription_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register the model
# MAGIC Now we instantiate the Azure ML Workspace we can register the model. First we will persist it to the dbfs (to be able to pass the path as a parameters to Azure ML Register)

# COMMAND ----------

import shutil
model_path = '/dbfs/models/churn-prediction'

# Delete old files
shutil.rmtree(model_path)

# Persist the XGBoost model
mlflow.xgboost.save_model(model, model_path)

# COMMAND ----------

from azureml.core.model import Model

model_name = 'churn-model'
model_description = 'Modelo de predição de churn utilizando XGBoost'

model_azure = Model.register(model_path = model_path,
                             model_name = model_name,
                             description = model_description,
                             workspace = workspace,
                             tags={'Framework': "XGBoost", 'Tipo': "Classificação"}
                             )

# COMMAND ----------

# MAGIC %md A new model version was generated in the Azure ML Workspace. We can use it to deploy an API with ACI or AKS.

# COMMAND ----------

# MAGIC %md
# MAGIC #Deploy
# MAGIC Now with the model registered we can choose between two deployment types: `ACI` (Azure Container Instance) or `AKS` (Azure Kubernetes Service).
# MAGIC 
# MAGIC For development scenarios it is better to use `ACI` and for production `AKS` will have more options related to scalability and security. Please see more details in this [page](https://docs.microsoft.com/en-us/azure/architecture/reference-architectures/ai/mlops-python).

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Entry script
# MAGIC But before deploy the model, it is important to define an **`entry script`** named score.py. It will be responsible to load the model when the deployed service starts and for receiving data, passing it to the model, and then returning a response as well (see this [link](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-existing-model#define-inference-configuration)).

# COMMAND ----------

# MAGIC %%writefile /dbfs/models/churn-prediction/score.py
# MAGIC 
# MAGIC import mlflow
# MAGIC import json
# MAGIC import pandas as pd
# MAGIC import os
# MAGIC import xgboost as xgb
# MAGIC import time
# MAGIC 
# MAGIC # Called when the deployed service starts
# MAGIC def init():
# MAGIC     global model
# MAGIC     global train_stats
# MAGIC 
# MAGIC     # Get the path where the deployed model can be found.
# MAGIC     model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), './churn-prediction')
# MAGIC     
# MAGIC     # Load model
# MAGIC     model = mlflow.xgboost.load_model(model_path)
# MAGIC 
# MAGIC # Handle requests to the service
# MAGIC def run(data):
# MAGIC   # JSON request.
# MAGIC   # {"Cylinders":0, "Displacement":0.0, "Horsepower":0.0, "Weight":0.0, "Acceleration":0.5, "Model Year":0, "USA":0.0, "Europe":0.0, "Japan":0.0}
# MAGIC   
# MAGIC   info = {"payload": data}
# MAGIC   print(json.dumps(info))
# MAGIC     
# MAGIC   data = pd.read_json(data, orient = 'split')
# MAGIC   data_xgb = xgb.DMatrix(data)
# MAGIC 
# MAGIC   # Return the prediction
# MAGIC   prediction = predict(data_xgb)
# MAGIC   print ("Prediction created at: " + time.strftime("%H:%M:%S"))
# MAGIC   
# MAGIC   return prediction
# MAGIC 
# MAGIC def predict(data):
# MAGIC   prediction = model.predict(data)[0]
# MAGIC   return {"churn-prediction": str(prediction)}

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Inference config
# MAGIC We must now add some inference configs to be used in the endpoint. We can add required packages and an environment that can be registered in the Azure ML Workspace.
# MAGIC 
# MAGIC Here we will use the same `conda.yaml` file that is already registered from MLFlow process. We will add the `azureml-defaults` package that can be used in the inference process.

# COMMAND ----------

from azureml.core.model import InferenceConfig
from azureml.core.environment import Environment
from azureml.core.conda_dependencies import CondaDependencies

# Create the environment
env = Environment(name='xgboost_env')

conda_dep = CondaDependencies('/dbfs/models/churn-prediction/conda.yaml')

# Define the packages needed by the model and scripts
conda_dep.add_pip_package("azureml-defaults")

# Adds dependencies to PythonSection of myenv
env.python.conda_dependencies=conda_dep

inference_config = InferenceConfig(entry_script="/dbfs/models/churn-prediction/score.py",
                                   environment=env)

# COMMAND ----------

# MAGIC %md Now with the inference config we can proceed with the deployment

# COMMAND ----------

# MAGIC %md
# MAGIC ###ACI - Azure Container Instance
# MAGIC Follow we will demonstrate how to create an `endpoint` using the image created before and delivering with `ACI`.

# COMMAND ----------

from azureml.core.webservice import AciWebservice, Webservice
from azureml.exceptions import WebserviceException
from azureml.core.model import Model

endpoint_name = 'api-churn-dev'

deployment_config = AciWebservice.deploy_configuration(cpu_cores = 1, memory_gb = 1)
service = Model.deploy(workspace, endpoint_name, [model_azure], inference_config, deployment_config, overwrite=True)
service.wait_for_deployment(show_output = True)

print('A API {} foi gerada no estado {}'.format(service.scoring_uri, service.state))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load some data to test the endpoint
# MAGIC We will use the same dataset used to train the model only for testing purposes.

# COMMAND ----------

import requests

payload1='{"columns":["Idade","RendaMensal","PercentualUtilizacaoLimite","QtdTransacoesNegadas","AnosDeRelacionamentoBanco","JaUsouChequeEspecial","QtdEmprestimos","NumeroAtendimentos","TMA","IndiceSatisfacao","Saldo","CLTV"],"data":[[21,9703,1.0,5.0,12.0,0.0,1.0,100,300,2,6438,71]]}'

payload2='{"columns":["Idade","RendaMensal","PercentualUtilizacaoLimite","QtdTransacoesNegadas","AnosDeRelacionamentoBanco","JaUsouChequeEspecial","QtdEmprestimos","NumeroAtendimentos","TMA","IndiceSatisfacao","Saldo","CLTV"],"data":[[21,9703,1.0,5.0,12.0,0.0,1.0,1,5,5,6438,71]]}'

# COMMAND ----------

# MAGIC %md
# MAGIC ## Call the API
# MAGIC Make a request to the API using `query_input`. The API url can be obtained throught `dev_webservice.scoring_uri` generated from deployment process.

# COMMAND ----------

headers = {
  'Content-Type': 'application/json'
}

response1 = requests.request("POST", service.scoring_uri, headers=headers, data=payload)
response2 = requests.request("POST", service.scoring_uri, headers=headers, data=payload2)

print(response1.text)
print(response2.text)

# COMMAND ----------

# MAGIC %md
# MAGIC It is also possible to use API using any client to make HTTP requests (curl, postman, etc.).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Azure Kubernetes Services (AKS)
# MAGIC For production scenarios it is better to deploy using AKS because we have more benefits about security and scalability.
# MAGIC 
# MAGIC In this scenario is possible to follow two ways: Creating a new AKS cluster or targeting to an existing one. In this tutorial we will use a existing cluster.

# COMMAND ----------

from azureml.core.webservice import AksWebservice
from azureml.core.compute import AksCompute

endpoint_name = 'api-churn-prod'
aks_name = 'aks-e2e-ds'

aks_target = AksCompute(workspace, aks_name)
aks_config = AksWebservice.deploy_configuration(compute_target_name=aks_name)

aks_service = Model.deploy(workspace=workspace,
                           name=endpoint_name,
                           models=[model_azure],
                           inference_config=inference_config,
                           deployment_config=aks_config,
                           deployment_target=aks_target)

aks_service.wait_for_deployment(show_output = True)
print(aks_service.state)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Call the API (with AKS)

# COMMAND ----------

prod_service_key = aks_service.get_keys()[0] if len(aks_service.get_keys()) > 0 else None

headers["Authorization"] = "Bearer {service_key}".format(service_key=prod_service_key)

response1 = requests.request("POST", aks_service.scoring_uri, headers=headers, data=payload)
response2 = requests.request("POST", aks_service.scoring_uri, headers=headers, data=payload2)

print(response1.text)
print(response2.text)