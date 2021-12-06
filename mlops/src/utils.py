# Databricks notebook source
import pandas as pd
import numpy as np
# Azure libs
from azureml.core.webservice import AciWebservice,  AksWebservice, Webservice
from azureml.core.image import Image
from azureml.core import Workspace
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.compute import AksCompute
from azureml.exceptions import WebserviceException
from azureml.core.model import Model, InferenceConfig
from azureml.core.environment import Environment
from azureml.core.conda_dependencies import CondaDependencies

# SKLearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# MLFlow
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.azureml
from mlflow.tracking.client import MlflowClient
from mlflow.entities import ViewType

import shutil
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

# Shap
import shap
import matplotlib.pyplot as plt
    
def get_dataset(filename):
    data = spark.read.parquet(filename)
    return data.toPandas()

def preprocessing(dataset):
    numeric_columns = []
    for col in dataset.columns:
        if(dataset[col].dtypes!='object'):
            numeric_columns.append(col)

    dataset = dataset.dropna()
    return dataset, numeric_columns

def split_dataset(dataset, seed, test_size=0.33):
    train_dataset, test_dataset = train_test_split(dataset, random_state=seed, test_size=test_size)
    return train_dataset, test_dataset

def get_X_y(train, test, target_column, numeric_columns, drop_columns):
    X_train = train[numeric_columns].drop(drop_columns, axis=1)
    X_test = test[numeric_columns].drop(drop_columns, axis=1)

    y_train = train[target_column]
    y_test = test[target_column]
    return X_train, X_test, y_train, y_test

def persist_shap(model, X_train):
    shap_values = shap.TreeExplainer(model).shap_values(X_train)
    shap.summary_plot(shap_values, X_train, show=False)
    plt.savefig('/dbfs/mnt/documents/images/shap.png')

def train_model(X_train, y_train, X_test, y_test):
    mlflow.set_experiment('/churn-prediction')

    with mlflow.start_run(run_name='mlops-train') as run:
        train = xgb.DMatrix(data=X_train, label=y_train)
        test = xgb.DMatrix(data=X_test, label=y_test)

        # Pass in the test set so xgb can track an evaluation metric. XGBoost terminates training when the evaluation metric
        # is no longer improving.
        model = xgb.train(params=params, dtrain=train, num_boost_round=1000,\
                           evals=[(test, "test")], early_stopping_rounds=50)

        mlflow.xgboost.log_model(model, 'model')
        persist_shap(model, X_train)
        mlflow.log_artifact('/dbfs/mnt/documents/images/shap.png')
        run_id = run.info.run_id

    return "runs:/" + run_id + "/model"

def validate_model(model, X_test, y_test):
    predictions_test = model.predict_proba(X_test)[:,1]
    auc_score = roc_auc_score(y_test, predictions_test)
    return auc_score

# COMMAND ----------

def get_model_uri(experiment_name, run_name):
    experiment = MlflowClient().get_experiment_by_name(experiment_name)
    experiment_ids = eval('[' + experiment.experiment_id + ']')

    query = f"tag.mlflow.runName = '{run_name}'"
    run = MlflowClient().search_runs(experiment_ids, query, ViewType.ALL)[0]

    return "runs:/" + run.info.run_id + "/model"

def load_model(model_uri):
    model = mlflow.xgboost.load_model(model_uri)
    return model

def persist_model(model, model_path):
    shutil.rmtree(model_path)

    # Persist the XGBoost model
    mlflow.xgboost.save_model(model, model_path)

def get_model(workspace, model_name):
    model_azure = Model(workspace, model_name)
    return model_azure
    
def register_model(workspace, model_name, model_description, model_path):
    model_azure = Model.register(model_path = model_path,
                               model_name = model_name,
                               description = model_description,
                               workspace = workspace,
                               tags={})
    return model_azure

def get_workspace(workspace_name, resource_group, subscription_id):
    svc_pr = ServicePrincipalAuthentication(
      tenant_id = dbutils.secrets.get(scope = "azure-key-vault", key = "tenant-id"),
      service_principal_id = dbutils.secrets.get(scope = "azure-key-vault", key = "client-id"),
      service_principal_password = dbutils.secrets.get(scope = "azure-key-vault", key = "client-secret"))

    workspace = Workspace.get(name = workspace_name,
                            resource_group = resource_group,
                            subscription_id = subscription_id,
                            auth=svc_pr)

    return workspace

def get_inference_config(environment_name, conda_file, entry_script):
    # Create the environment
    env = Environment(name=environment_name)

    conda_dep = CondaDependencies(conda_file)

    # Define the packages needed by the model and scripts
    conda_dep.add_pip_package("azureml-defaults")
    conda_dep.add_pip_package("azureml-monitoring")
    conda_dep.add_pip_package("xgboost")

    # Adds dependencies to PythonSection of myenv
    env.python.conda_dependencies=conda_dep

    inference_config = InferenceConfig(entry_script=entry_script,
                                     environment=env)

    return inference_config

def deploy_aci(workspace, model_azure, endpoint_name, inference_config):
    deployment_config = AciWebservice.deploy_configuration(cpu_cores = 1, memory_gb = 1, auth_enabled=True)
    service = Model.deploy(workspace, endpoint_name, [model_azure], inference_config, deployment_config, overwrite=True)
    service.wait_for_deployment(show_output = True)

    print(f"Endpoint : {endpoint_name} was successfully deployed to ACI")
    print(f"Endpoint : {service.scoring_uri} created")
    return service

def deploy_aks(workspace, model_azure, endpoint_name, inference_config, aks_name):
    aks_target = AksCompute(workspace, aks_name)
    aks_config = AksWebservice.deploy_configuration(enable_app_insights = True, collect_model_data=True)

    aks_service = Model.deploy(workspace=workspace,
                             name=endpoint_name,
                             models=[model_azure],
                             inference_config=inference_config,
                             deployment_config=aks_config,
                             deployment_target=aks_target,
                             overwrite=True)

    aks_service.wait_for_deployment(show_output = True)

    print(f"Endpoint : {endpoint_name} was successfully deployed to AKS")
    print(f"Endpoint : {aks_service.scoring_uri} created")
    print('')
