# Databricks notebook source
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
# MAGIC   return {"churn-prediction": str(int(prediction))}