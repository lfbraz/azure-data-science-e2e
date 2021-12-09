# Databricks notebook source
# MAGIC %run ../src/utils

# COMMAND ----------

seed = 2022
target = 'Churn'

drop_columns = [target, 'CodigoCliente']

# Get the Train Dataset
dataset = get_dataset('/dbfs/Dataset/Customer')

# Preprocessing Features
dataset, numeric_columns = preprocessing(dataset)

# Split train and test
train_dataset, test_dataset = split_dataset(dataset, seed)

# Parameters we got from the best interaction
params = {'early_stopping_rounds': 50, 
          'learning_rate': 0.2261, 
          'max_depth': 64, 
          'maximize': False, 
          'min_child_weight': 19.22, 
          'num_boost_round': 1000, 
          'reg_alpha': 0.01, 
          'reg_lambda': 0.348, 
          'verbose_eval': False,
          'seed': seed}

# Get X, y
X_train, X_test, y_train, y_test = get_X_y(train_dataset, test_dataset, target, numeric_columns, drop_columns)

# Train model and Load
model_uri = train_model(X_train, y_train, X_test, y_test)
model = load_model(model_uri)

# Persist in the DBFS
persist_model(model, '/dbfs/models/churn-prediction')

print('Model trained')
