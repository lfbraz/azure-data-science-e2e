# Lab 1: Deploy the machine learning model

In this lab you will deploy a real-time endpoint to consume the machine learning model trained in the previous lab (take a look in the [Lab 0](../lab%200/Lab0.md)).

Azure Machine Learning provides robust MLOps capabilities that integrate with existing DevOps processes and help manage the complete ML lifecycle.

To know more please take a look in this [link](https://azure.microsoft.com/en-us/services/machine-learning/mlops/).

## Microsoft Learn & Technical Documentation

The following Azure services will be used in this lab. If you need further training resources or access to technical documentation please find in the table below links to Microsoft Learn and to each service's Technical Documentation.

Azure Service | Microsoft Learn | Technical Documentation|
--------------|-----------------|------------------------|
Azure Machine Learning - Overview | [Build AI solutions with Azure Machine Learning](https://docs.microsoft.com/en-us/learn/paths/build-ai-solutions-with-azure-ml-service/) | [Azure Machine Learning Technical Documentation (overview)](https://docs.microsoft.com/en-us/azure/machine-learning/)
Azure Machine Learning - Deployment | [Deploy real-time machine learning services with Azure Machine Learning](https://docs.microsoft.com/en-us/learn/modules/register-and-deploy-model-with-amls//) | [Azure Machine Learning Technical Documentation (deploy)](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-and-where?tabs=azcli)

## Lab Architecture

![](/images/data-science-architecture-lab-1.png)

Step     | Description
-------- | -----
![1](/images/Black1.png) | Create a Azure Machine Learning Workspace
![2](/images/Black2.png) | Deploy to Azure Container Instance (ACI)
![3](/images/Black3.png) | Deploy to Azure Kubernetes Service (AKS)
![4](/images/Black4.png) | *BONUS*: Deploy to Azure Functions (preview)

### Create an Azure Machine Learning Workspace

1. First, in your *resource group* click in **+ADD** and type **Azure Machine Learning** to find the resource:

![1](/images/lab-1/1-add-aml.PNG)

So click in **Create** button

1. Provide the details: `Workspace Name` and `Region`. Azure Machine Learning needs a Storage Account, Key Vault, App Insights and a Container Registry. For this Lab let's use the default settings and click on **Next**. **IMPORTANT:** Don't delete these resources after creating the AML Workspace, they are part of the Workspace and delete them can make the service unusable.  
2. On **Networking** we can leave the default since we will not use a Private endpoint.
3. On **Advanced** leave the default "Microsoft-managed-keys" and click on **Next**.
4. Put any Tag if you want
5. And click on **Create**

The deployment process will start and after some minutes we will have the Azure Machine Learning created in the Resource Group

### Launch the Workspace

Now go to your resource group and click on the new resource created (with the `Workspace Name` you provided), so click on **Launch Workspace**

![](/images/lab-1/2-launch-aml-workspace.PNG)

In the Azure Machine Learning Workspace we have a lof of interesting features, among them some useful resources to lead with MLOps aspects. In this lab we will use **Model registry**, **Compute** and **Endpoints** capabilities that can help us during the deployment process integrated with Azure Databricks.

