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

![](/images/data-science-architecture.png)

Step     | Description
-------- | -----
![1](/images/Black1.png) | Build an Azure Data Factory Pipeline to copy data from an Azure SQL Database table
![2](/images/Black2.png) | Use Azure Data Lake Storage Gen2 as a staging area for Polybase
![3](/images/Black3.png) | Load data to an Azure Synapse Analytics table using Polybase
![4](/images/Black4.png) | Visualize data from Azure Synapse Analytics using Power BI