# Databricks notebook source
# MAGIC %md
# MAGIC # Churn model - Training a simple classification model
# MAGIC In this tutorial we will cover how to implement a simple Machine Learning model to predict Customer's Churn following these steps:
# MAGIC 
# MAGIC - Import data into Databricks File System (DBFS)
# MAGIC - Explore and visualize data using popular libraries
# MAGIC - Run an optimization to get the best hyperparameters to the model
# MAGIC - Persist the best model using MLFlow
# MAGIC 
# MAGIC **Pre-Requisites**
# MAGIC * Conclude the steps before in the Lab 0
# MAGIC 
# MAGIC ** This tutorial was inspired in the [databricks doc](https://docs.databricks.com/applications/machine-learning/train-model/scikit-learn.html)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Get the data
# MAGIC We will use a learning purposes dataset created with simulated data. This dataset can be found [here](https://raw.githubusercontent.com/lfbraz/azure-databricks-mlops/master/dataset/dados_clientes.csv). We will download this dataset to `dbfs` to be able to import data to dataframes.

# COMMAND ----------

Customer = spark.read.parquet('/dbfs/Dataset/Customer')

# COMMAND ----------

# MAGIC %md
# MAGIC We will develop a `sklearn` model so it will be easier to work with a Pandas Dataframe instead of a spark one (of course, when we are working in an spark environment it would be better to work with spark dataframes because of performance aspects, however for this tutorial we will use Pandas for learning reasons)

# COMMAND ----------

customer_data = Customer.toPandas()

# COMMAND ----------

customer_data.head(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Data Visualization
# MAGIC 
# MAGIC Before training a model, explore the dataset using Seaborn and Matplotlib.

# COMMAND ----------

# MAGIC %md
# MAGIC Plot a histogram of the dependent variable, **IndiceSatisfacao** (An index that measure how satisfied a customer is with the service).

# COMMAND ----------

import seaborn as sns
sns.distplot(customer_data.IndiceSatisfacao, kde=False)

# COMMAND ----------

# MAGIC %md
# MAGIC Now we can analyse how the numeric features correlate each other, `pandas` have a native method `corr()` that use `pearson` correlation by default (you can alter this behavior using the *method* parameter).

# COMMAND ----------

import matplotlib as plt

numeric_columns = []
for col in customer_data.columns:
  if(customer_data[col].dtypes!='object'):
    numeric_columns.append(col)
    
corr = customer_data[numeric_columns].corr()

ax1 = sns.heatmap(corr, cbar=0, linewidths=5, square=True, cmap='Reds')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preprocessing Data
# MAGIC Prior to training a model, check for missing values and split the data into training and validation sets.
# MAGIC In the last steps of this lab we already checked this point, so we cannot have any missing data

# COMMAND ----------

customer_data.isna().any()

# COMMAND ----------

# MAGIC %md
# MAGIC Indeed there is no missing data

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train the model

# COMMAND ----------

# MAGIC %md
# MAGIC First of all let's split train and test data

# COMMAND ----------

from sklearn.model_selection import train_test_split
SEED = 2020

train, test = train_test_split(customer_data, random_state=SEED, test_size=0.33)

# COMMAND ----------

train.head(1)

# COMMAND ----------

TARGET_COLUMN = 'Churn'
drop_columns = [TARGET_COLUMN, 'CodigoCliente'] 

X_train = train[numeric_columns].drop(drop_columns, axis=1)
X_test = test[numeric_columns].drop(drop_columns, axis=1)

y_train = train[TARGET_COLUMN]
y_test = test[TARGET_COLUMN]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Building a Baseline Model
# MAGIC This task seems well suited to a random forest classifier, since the output is binary and there may be interactions between multiple variables.
# MAGIC 
# MAGIC The following code builds a simple classifier using scikit-learn.

# COMMAND ----------

import numpy as np
from sklearn.ensemble import RandomForestClassifier

n_estimators = 3

model = RandomForestClassifier(n_estimators=n_estimators, random_state=np.random.RandomState(SEED))
model.fit(X_train, y_train)

# COMMAND ----------

# predict_proba returns [prob_negative, prob_positive], so slice the output with [:, 1]
predictions_test = model.predict_proba(X_test)[:,1]


# COMMAND ----------

y_pred = model.predict(X_test)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Model Evaluation

# COMMAND ----------

# MAGIC %md
# MAGIC Get False and True Positive rate

# COMMAND ----------

from sklearn.metrics import confusion_matrix
  
c_matrix_log = confusion_matrix(
  y_test, y_pred, labels=[0, 1])
sns.heatmap(
    c_matrix_log, annot=True, fmt="d")

# COMMAND ----------

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

print('Results: ',
          str(model).split('(')[0], '\nPrecisão: {0:.3}'.format(
              precision_score(y_test, y_pred)), '\nRecall: {0:.3f}'.format(
                  recall_score(y_test, y_pred)), '\nAUC: {0:.3f}'.format(
                      roc_auc_score(y_test, y_pred)),
          '\nAcurácia: {0:.3f}'.format(accuracy_score(y_test, y_pred)), '\n')

# COMMAND ----------

# MAGIC %md
# MAGIC Plot ROC Curve

# COMMAND ----------

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

fpr_rf, tpr_rf, _ = roc_curve(y_test, predictions_test)

plt.figure(1)
plt.plot([0, 1], [0, 1], '--')
plt.plot(fpr_rf, tpr_rf, label = 'ROC Curve (AUC: {0:.2f})'.format(
                      roc_auc_score(y_test, y_pred)))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC Curve')
plt.legend(loc='best')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Seems to be a good model to begin. 😃
# MAGIC 
# MAGIC But let's try to make some improvements ...
# MAGIC 
# MAGIC First we will create an experiment to be able to track the **experiment**

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create an experiment to track our models
# MAGIC MLflow Experiments can be created and organized in the Databricks Workspace just like a notebook, library, or folder. Simply click on an experiment to see a list of its runs or compare them. Crucially, with Managed MLflow, Experiments are integrated with Databricks’ standard role-based access controls to set sharing permissions.

# COMMAND ----------

import mlflow

experiment_name = '/churn-prediction'

if(not(mlflow.get_experiment_by_name(experiment_name))):
  mlflow.create_experiment(experiment_name)

# COMMAND ----------

# MAGIC %md
# MAGIC Now we will try to optimize the hyperparameters of the model. We will use `hyperopt` (also inspired in this [doc](https://docs.databricks.com/applications/machine-learning/train-model/scikit-learn.html)) to be able to get the best parameters of our model. Mode details about `hyperopt` can be seen in this [link](https://docs.microsoft.com/en-us/azure/databricks/applications/machine-learning/automl-hyperparam-tuning/#hyperopt-overview). 
# MAGIC 
# MAGIC We will use `XGBoost` to try to increase the performance of our predictions as well.

# COMMAND ----------

from hyperopt import fmin, tpe, hp, SparkTrials, Trials, STATUS_OK
from hyperopt.pyll import scope
from math import exp
import mlflow.xgboost
from mlflow.entities import ViewType
import numpy as np
import xgboost as xgb

mlflow.set_experiment(experiment_name)
run_name = 'XGBoost-model'

search_space = {
  'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
  'learning_rate': hp.loguniform('learning_rate', -3, 0),
  'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
  'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
  'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
  #'objective': 'binary:logistic',
  'objective': 'binary:hinge',
  'seed': SEED,
}

def train_model(params):
  # With MLflow autologging, hyperparameters and the trained model are automatically logged to MLflow.
  mlflow.xgboost.autolog()

  with mlflow.start_run(nested=True):
    train = xgb.DMatrix(data=X_train, label=y_train)
    test = xgb.DMatrix(data=X_test, label=y_test)
    
    # Pass in the test set so xgb can track an evaluation metric. XGBoost terminates training when the evaluation metric
    # is no longer improving.
    booster = xgb.train(params=params, dtrain=train, num_boost_round=1000,\
                        evals=[(test, "test")], early_stopping_rounds=50)
    predictions_test = booster.predict(test)
    auc_score = roc_auc_score(y_test, predictions_test)
    mlflow.log_metric('auc', auc_score)

    # Set the loss to -1*auc_score so fmin maximizes the auc_score
    return {'status': STATUS_OK, 'loss': -1*auc_score, 'booster': booster.attributes()}

# Greater parallelism will lead to speedups, but a less optimal hyperparameter sweep. 
# A reasonable value for parallelism is the square root of max_evals.
spark_trials = SparkTrials(parallelism=10)

# Run fmin within an MLflow run context so that each hyperparameter configuration is logged as a child run of a parent
# run called "xgboost_models" .
with mlflow.start_run(run_name=run_name):
  best_params = fmin(
    fn=train_model, 
    space=search_space, 
    algo=tpe.suggest, 
    max_evals=20,
    trials=spark_trials, 
    rstate=np.random.RandomState(SEED)
  )

# COMMAND ----------

# MAGIC %md
# MAGIC ### And the best run 😎

# COMMAND ----------

best_run = mlflow.search_runs(order_by=['metrics.auc DESC'],
                              run_view_type=ViewType.ACTIVE_ONLY,
                             ).iloc[0]

print(f'AUC of Best Run: {round(best_run["metrics.auc"], 3)}')

model_uri = "runs:/" + best_run["run_id"] + "/model"
print(f'model_uri: {model_uri}')
