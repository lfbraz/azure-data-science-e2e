# Lab 0: Create a Machine Learning model to predict customer`s churn

In this lab you will configure the Azure environment to develop the Machine Learning model. We can use different Azure Service for this task (Azure Machine Learning, Azure Synapse, etc.). In this lab we will show how to use Azure Databricks to accomplish this task.

[Azure Databricks](https://azure.microsoft.com/pt-br/services/databricks/) allows you to build artificial intelligence (AI) solutions in an Apache Spark™ environment in minutes, autoscale, and collaborate on shared projects in an interactive workspace. Azure Databricks supports Python, Scala, R, Java, and SQL, as well as data science frameworks and libraries including TensorFlow, PyTorch, and scikit-learn.

## Microsoft Learn & Technical Documentation

The following Azure services will be used in this lab. If you need further training resources or access to technical documentation please find in the table below links to Microsoft Learn and to each service's Technical Documentation.

Azure Service | Microsoft Learn | Technical Documentation|
--------------|-----------------|------------------------|
Azure Databricks | [Perform data science with Azure Databricks](https://docs.microsoft.com/en-us/learn/paths/perform-data-science-azure-databricks/) | [Azure Databricks Technical Documentation](https://docs.microsoft.com/en-us/azure/databricks/)
Azure Data Lake Storage Gen2 | [Large Scale Data Processing with Azure Data Lake Storage Gen2](https://docs.microsoft.com/en-us/learn/paths/data-processing-with-azure-adls/) | [Azure Data Lake Storage Gen2 Technical Documentation](https://docs.microsoft.com/en-us/azure/storage/blobs/data-lake-storage-introduction)

## Lab Architecture

![](/images/data-science-architecture-lab-0.png)

Step     | Description
-------- | -----
![1](/images/Black1.png) | Create a Azure Databricks Workspace
![2](/images/Black2.png) | Create a Spark Cluster
![3](/images/Black3.png) | Import Dataset to Databricks Filesystem (DBFS)
![4](/images/Black4.png) | Prepare Dataset
![4](/images/Black5.png) | Develop a Machine Learning model using the prepared Dataset

### Create an Azure Databricks Workspace

<br>**IMPORTANT**: Don't forget to create a Resource Group to be used in this entire Lab. For instructions of how to do that, please see this [link](https://docs.microsoft.com/en-us/azure/azure-resource-manager/management/manage-resource-groups-portal). Azure Databricks will be created into this resource group.

1. First, in your *resource group* click in **+ADD** and type **Azure Databricks** to find the resource:

![1](/images/lab-0/1-add-adb.PNG)

So click in **Create** button

2. Provide the details: `Workspace Name`, `Region` and `Pricing Tier` and click on **Next**
3. On **Networking** we can leave the default since we will not deploy this service in a Virtual Network (VNet)
4. Put any Tag if you want
5. And click on **Create**

![1](/images/lab-0/2-create-adb-workspace.PNG)

The deployment process will start and after some minutes we will have the Azure Databricks created in the Resource Group

### Launch the Workspace

Now go to your resource group and click on the new resource created (with the `Workspace Name` you provided), so click on **Launch Workspace**

![1](/images/lab-0/3-launch-adb-workspace.PNG)

And ..

![1](/images/lab-0/3b-launch-adb-workspace.PNG)

### Create a spark cluster

Let's create a spark cluster. Click on **Clusters** and **Create Cluster** the following screen will appear, keep the default settings and provide a `Cluster Name`

![1](/images/lab-0/4-create-spark-cluster.PNG)

With that, just click on **Create Cluster** and wait for around *7 minutes*

### Import data
Now, with the cluster created, we can add data to be able to work with it in the notebooks in our Workspace. There are several ways to do that and in this tutorial we will show how to upload data using Databricks UI.

First, download the [dataset](https://raw.githubusercontent.com/lfbraz/azure-data-science-e2e/main/dataset/dados_clientes.csv) to your local machine (just click in *Save as* and choose your local folder)

This is a .csv file with fake customer's data so we can use it in this lab to practice how to work with datasets and how to use it to develop machine learning models.

Now you have the dataset in your local machine, we can upload it to Azure Databricks.

In the home of your workspace click on **Import & Explore Data**

![1](/images/lab-0/5a-upload-dataset.PNG)

Browse to your .csv file and after click on **Create Table with UI**

![](/images/lab-0/5b-upload-dataset.PNG)

Select your spark cluster and click on **Preview Table**

![](/images/lab-0/5c-upload-dataset.PNG)

It's important to select `First row is header` and `Infer schema` options to be able to load the csv correctly.

After that, click on **Create Table**. It will create a table registered in the internal Hive metastore (already created with the Databricks cluster). If you want to check for more Hive metastore options please take a look in [this](https://docs.microsoft.com/en-us/azure/databricks/kb/metastore/).

It is also important to note that the .csv file was uploaded to *databricks filesystem* (dbfs):

![](/images/lab-0/5d-upload-dataset.PNG)

This path can be also used to import the file directly from the notebooks. So let's try it! 😁

#### Read the data from the notebook

Now we can create a notebook to be able to import and explore the data. So let's create a notebook. Click in the **Workspace** icon and right click in the blank space below your user:

![](/images/lab-0/5-create-notebook.PNG)

So just click on *Create -> Notebook*.

Here you will provide a name and choose the best language to work on. You can choose Python, Scala, R or Spark SQL. This is only the *default* language to work with, but you can choose another one using `%language`(like `%sql`, `%python`, etc.) in each cell inside the notebooks.

Now let's import the data using `python` language (pyspark). Why Python? Just because we like Python 🤣.

**IMPORTANT**|
-------------|
**Take a look in this [notebook](/labs/lab%200/notebooks/read-data.ipynb).**|

You can also import this notebook to you own workspace. Just righ-click on the black space below your username and choose *Import -> URL* and put the path.

We can also import/export data from/to Data Lakes (for example an Azure Data Lake storage). If will want to see more details how to do that take a look in this [notebook](https://github.com/lfbraz/azure-databricks/blob/master/notebooks/read-from-adls.ipynb)

### Prepare the data

An important step of a Data Science project is regarding to **Prepare and transform the data** before creating the models. It is important to check for inconsistences and try to explore the biggest value from the data you have. So let's see how to do some basic transformations.

**IMPORTANT**|
-------------|
**Take a look in this [notebook](/labs/lab%200/notebooks/simple-etl-with-spark.ipynb).**|

### Putting all together - Develop the Machine Learning model

Now it is timeeeeeeeeee !!! 😎

Let's create a machine learning model using the data we prepared in the last section.

**IMPORTANT**|
-------------|
**Take a look in this [notebook](/labs/lab%200/notebooks/model-churn-prediction.ipynb).**|
