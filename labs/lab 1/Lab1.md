# Lab 1: Deploy the machine learning model

In this lab you will deploy a real-time endpoint to consume the machine learning model trained in the previous lab (take a look in the [Lab 0](../lab-0/Lab0.md)).

Azure Machine Learning provides robust MLOps capabilities that integrate with existing DevOps processes and help manage the complete ML lifecycle.

To know more please take a look in this [link](https://azure.microsoft.com/en-us/services/machine-learning/mlops/).

## Microsoft Learn & Technical Documentation

The following Azure services will be used in this lab. If you need further training resources or access to technical documentation please find in the table below links to Microsoft Learn and to each service's Technical Documentation.

Azure Service | Microsoft Learn | Technical Documentation|
--------------|-----------------|------------------------|
Azure Machine Learning - Overview | [Build AI solutions with Azure Machine Learning](https://docs.microsoft.com/en-us/learn/paths/build-ai-solutions-with-azure-ml-service/) | [Azure Machine Learning Technical Documentation (overview)](https://docs.microsoft.com/en-us/azure/machine-learning/)
Azure Machine Learning - Deployment | [Deploy real-time machine learning services with Azure Machine Learning](https://docs.microsoft.com/en-us/learn/modules/register-and-deploy-model-with-amls//) | [Azure Machine Learning Technical Documentation (deploy)](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-and-where?tabs=azcli)

## Lab Architecture

![1](/images/data-science-architecture-lab-1.png)

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

![1](/images/lab-1/2-launch-aml-workspace.PNG)

In the Azure Machine Learning Workspace we have a lof of interesting features, among them some useful resources to lead with MLOps aspects. In this lab we will use **Model registry**, **Compute** and **Endpoints** capabilities that can help us during the deployment process integrated with Azure Databricks.

![1](/images/lab-1/3-mlops-capabilities.PNG)

### Deploy to Azure Container Instance (ACI) and Azure Kubernetes Service (AKS)

Now we have an Azure Machine Learning Workspace we can use it to integrate with our Azure Databricks Workspace to be able to build a MLOps process.

For this, we can use MLFlow together with Azure ML SDK, so it will be possible to integrate the best of this two worlds. 😋

**IMPORTANT**|
-------------|
**Take a look in this [notebook](/labs/lab%201/notebooks/deploy-model-churn-prediction.ipynb).**|

You can also import this notebook to your own workspace. Just right-click on the blank space below your username and choose *Import -> File* and put the path of the file. You can download to your local machine and upload to Databricks Workspace as well.

### BONUS: Deploy to Azure Functions

We can also deploy our Machine Learning model to an Azure Function. With [Functions](https://azure.microsoft.com/en-us/services/functions/) we can have an event-driven serverless compute platform that can also solve complex orchestration problems. Build and debug locally without additional setup, deploy and operate at scale in the cloud, and integrate services using triggers and bindings.

In this lab, inspired from this [doc](https://docs.microsoft.com/en-us/azure/azure-cache-for-redis/cache-ml) (some parts were copied from there) we will use our already trained model to be deployed to an Azure Function. For this, we can use `azureml-contrib-functions` together with Azure ML SDK.

**IMPORTANT**|
-------------|
**Take a look in this [notebook](/labs/lab%201/notebooks/deploy-model-to-azure-function.ipynb).**|

With this notebook, you will package the already trained model as a docker image an register it in the Azure Container Registry (in a  repository named as package). Now we need to use `az-cli` to create the Azure Function and associate it with the model image. To see more details about this process please also take a look in this [doc](https://docs.microsoft.com/en-us/azure/azure-cache-for-redis/cache-ml).

First, let's get the login credentials from Azure Container Registry:

`az acr credential show --name <myacr>`

This results (**username** and **password**) will be used soon.

For the Azure Function we will need a [*app service plan*](https://docs.microsoft.com/en-us/azure/app-service/overview-hosting-plans), so let`s create it:

`az appservice plan create --name myplanname --resource-group myresourcegroup --sku B1 --is-linux`

In this example, a Linux basic pricing tier (--sku B1) is used.
IMPORTANT: Images created by Azure Machine Learning use Linux, so you must use the --is-linux parameter.

For this process we need to connect an [Storage Account](https://docs.microsoft.com/en-us/azure/storage/common/storage-account-overview) to be used for the web job storage. You can create it with the following:

`az storage account create --name <webjobStorage> --location locationofstorageaccount --resource-group myresourcegroup --sku Standard_LRS`

Now, we will create the Azure Function. Replace `app-name` with the name you want to use. Replace `acrinstance` and `imagename` with the values from returned package.location earlier. Replace `webjobStorage` with the name of the storage account from the previous step:

`az functionapp create --resource-group myresourcegroup --plan myplanname --name <app-name> --deployment-container-image-name <acrinstance>.azurecr.io/package:<imagename> --storage-account <webjobStorage>`

To provide the function app with the credentials needed to access the container registry, use the following command. Replace `app-name` with the name of the function app. Replace `acrinstance` and `imagetag` with the values from the AZ CLI call in the previous step. Replace `username` and `password` with the ACR login information retrieved earlier:

`az functionapp config container set --name <app-name> --resource-group myresourcegroup --docker-custom-image-name <acrinstance>.azurecr.io/package:<imagetag> --docker-registry-server-url https://<acrinstance>.azurecr.io --docker-registry-server-user <username> --docker-registry-server-password <password>`

We did it! 😜

To test the Rest Endpoint we can use some tool like `Postman`, etc. or simply follow the steps:

1. Go to your Azure Function app in the Azure portal.
2. Under developer, select Code + Test.
3. On the right hand side, select the Input tab.
4. Click on the Run button to test the Azure Function HTTP trigger.
