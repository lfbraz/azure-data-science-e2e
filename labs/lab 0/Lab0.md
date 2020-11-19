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
![2](/images/Black2.png) | Import Dataset to Databricks Filesystem (DBFS)
![3](/images/Black3.png) | Prepare Dataset
![4](/images/Black4.png) | Develop a Machine Learning model using the prepared Dataset

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

### Import data
Now we have the Azure Databricks Workspace we can add data to be able to work with it in the notebooks in our Workspace. There are several ways to do that and in this part we will show how to upload data using Databricks UI.

First, download the [dataset](https://raw.githubusercontent.com/lfbraz/azure-data-science-e2e/main/dataset/dados_clientes.csv) to your local machine (just click in *Save as* and choose your local folder)


