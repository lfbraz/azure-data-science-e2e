# Lab 2: Build a MLOps Pipeline

In this lab you see how to create a MLOps Pipeline using **Azure DevOps**.

[Azure DevOps](https://docs.microsoft.com/en-us/azure/databricks/notebooks/azure-devops-services-version-control) is a collection of services that provide an end-to-end solution for the five core practices of DevOps: planning and tracking, development, build and test, delivery, and monitoring and operations. This article describes how to set Azure DevOps as your Git provider.

We can use these capabilities to create a MLOps Pipeline as well, so we can automate all the tasks required to put a machine learning in production.

To know more please take a look in this [link](https://docs.microsoft.com/en-us/azure/machine-learning/concept-model-management-and-deployment#:~:text=%20Azure%20Machine%20Learning%20provides%20the%20following%20MLOps,associated%20metadata%20required%20to%20use%20the...%20More%20/).

## Microsoft Learn & Technical Documentation

The following Azure services will be used in this lab. If you need further training resources or access to technical documentation please find in the table below links to Microsoft Learn and to each service's Technical Documentation.

Azure Service | Microsoft Learn | Technical Documentation|
--------------|-----------------|------------------------|
Azure Machine Learning| [Start the machine learning lifecycle with MLOps](https://docs.microsoft.com/en-us/learn/modules/start-ml-lifecycle-mlops/) | [MLOps: Model management, deployment, and monitoring with Azure Machine Learning](https://docs.microsoft.com/en-us/azure/machine-learning/concept-model-management-and-deployment)
Azure DevOps| [Get started with Azure DevOps](https://docs.microsoft.com/en-us/learn/modules/get-started-with-devops/) | [Azure DevOps Technical Documentation](https://docs.microsoft.com/en-us/azure/devops/?view=azure-devops)

## Lab Architecture

![1](/images/data-science-architecture-lab-2.png)

Step     | Description
-------- | -----
![1](/images/Black1.png) | Connect your Databricks Workspace with a github/Azure DevOps repository
![2](/images/Black2.png) | Create a simple Hello World pipeline with Azure DevOps
![3](/images/Black3.png) | Customize the pipeline to represent a MLOps process with *DevOps for Databricks extension*

### Connect your Databricks Workspace with a github/Azure DevOps repository

1. In your Databricks Workspace create some notebooks to package and deploy your models based on what we learned in the previous labs

**IMPORTANT**|
-------------|
**Review this notebook [notebook](/labs/lab%201/notebooks/deploy-model-churn-prediction.ipynb)** and don't forget to automate your model training as well|

Follow a suggestion's structure (in this case we used the *Shared* structured in the Databricks Workspace):

![1](/images/lab-2/1-shared-folder.png)

We created some methods in the notebook **utils** to help to automate all the process in a organized way, but feel free to customize according your requirements. All the scripts can be seen in this [folder](../../mlops/src/). Please use only as inspiration 🤗.

2. Now we will connect our repository to Databricks Workspace. Click on your Workspace icon and after *User settings*:

![1](/images/lab-2/2-connect-git.png)

Click on *Git integration* and select *Azure DevOps Services* (must be in the same tenant). For cases you would like to connect to other provides (eg. Github) you can user *Personal Tokens*.

3. Now, with the Git integration configured, we can come to back to our notebooks and we must sync the notebook with the repository. For this, click on *Revision history* button to be able to sync your notebook with Git. In this case, when we click on *Save Now* (with *Also commit to Git* option) the notebook will be commited to the repository (Databricks transform your notebook to a python's file, with .py extension).

![1](/images/lab-2/3-git-integration.png)

![2](/images/lab-2/4-revision-history.png)

![3](/images/lab-2/5-commit-to-git.png)

Don't forget to provide a description 😆. This description will be recorded in the repository and saved to posteriority 😜.

It's also important to choose a good folder structure in your repository (remember it will be used in the MLOps pipeline).

Don't forget to do the same process to all the other notebooks you would like to be part of this MLOps process.

### Create a simple Hello World pipeline with Azure DevOps

Now we have the notebooks integrated in a git repository. Next step we will create the MLOps Pipeline in the Azure DevOps.

1. First of all, go to [Azure DevOps](https://dev.azure.com/) and be sure to have an organization created and configured. If you have questions about this process, please take a look in [this doc](https://docs.microsoft.com/en-us/azure/devops/organizations/accounts/create-organization).
2. In your Azure DevOps organization, create a new project (or use an existing one) that can be used to host our MLOps Pipeline.

![1](/images/lab-2/6-create-new-azure-devops-project-b.png)

![2](/images/lab-2/6-create-new-azure-devops-project.png)

3. Now, go to your project and click on *Pipelines*

![1](/images/lab-2/9-click-on-pipelines.png)

4. Click on *New pipeline* and select *Where is your code?* (in our case Azure Repos Git)

![1](/images/lab-2/10-create-new-pipeline.png)

![2](/images/lab-2/11-where-is-your-code.png)

This option refers to where your pipeline's code you be hosted (and not your notebooks). So you will be able to control your own pipeline in a git repository (in this case we will use the same repository for both, the pipeline and the training/deployment code as well).

Don't forget to select the *Repository* you created in Azure Repos. If you have question about how to do it please take a look in this [doc](https://docs.microsoft.com/en-us/azure/devops/repos/get-started/what-is-repos?view=azure-devops).

5. Now select *Starter pipeline* (if you have an exiting .yml file with your pipeline you can select *Existing Azure Pipelines YAML file*).

![1](/images/lab-2/12-configure-your-pipeline.png)

6. Finally, click on *Save and run*, it will commit your pipeline file in the repository and run a *Hello, world!* echo in the pipeline.

![1](/images/lab-2/13-save-and-run.png)

We have a simple pipeline created, now let's add more tasks to this pipeline.

### Customize the pipeline to create a MLOps process with *DevOps for Databricks extension*

1. In this MLOps lab we will use the [DevOps for Azure Databricks](https://marketplace.visualstudio.com/items?itemName=riserrad.azdo-databricks&targetId=09d19ee8-b94a-4f99-a763-11cc0fe1a111&utm_source=vstsproduct&utm_medium=ExtHubManageList) extension. It uses a abstraction of [databricks-cli](https://docs.databricks.com/dev-tools/cli/index.html) a command-line interface that provides an easy-to-use interface to the Databricks plataform. To install this extension go to Azure DevOps *MarketPlace* and search for *DevOps for Azure Databricks*

![1](/images/lab-2/7-search-devops-for-databricks-market-place.png)

![2](/images/lab-2/8-install-devops-for-databricks.png)

2. Now, let's customize the previous created pipeline. Click on *Pipelines* and *Edit* your pipeline.

![1](/images/lab-2/14-edit-pipeline.png)

3. In the *Tasks* menu type Databricks to list all the Tasks we can use in the MLOps pipeline.

![1](/images/lab-2/15-tasks-databricks.png)

In a simple MLOps pipeline we could create a Pipeline that simply call the notebooks, to train, deploy, etc. the models. 

For each task, it will be required to put some informations about your Azure Databricks Workspace (Workspace URL, Cluster ID, etc.). Be sure to use *variables* in the pipeline, so you can keep them secret (**Don't commit sensitive data to your repository !!!**).

Take a look in this [simple MLOps pipeline](/mlops/mlops-pipeline.yml).

We have a lot of useful features in the Azure DevOps, as trigger from a branch (to trigger when someone commit a code), the separation in *stages, jobs, etc.*, among others.

In the end, your pipeline will be shown like this:

![1](/images/lab-2/16-final-pipeline.png)
