# Data Science with Azure Machine Learning and Azure Databricks

In this workshop we will show you how to use Azure Databricks and Azure Machine Learning for the development and training of artificial intelligence models making them available in an integration and continuous delivery process (CI/CD), demonstrating how to build an automated MLOps process consuming the generated artifacts and making them available in a simple and dynamic way for consumption from other applications.

In the exercises you will learn how to build machine learning models for Churn prediction with Spark development and using a fake database for model training. The exercises also include a scenario for automating the deployment process of the model generated using this Github repository and an Azure DevOps project.

The idea of ​​the Workshop is to build an end-to-end experience for developing, training, deploying and monitoring machine learning models, thus facilitating a complete Data Science experience to be reproduced in your own Azure environment.

**IMPORTANT:**

* The reference architecture proposed in this workshop aims to explain just enough of the role of each of the Azure Data Services included in the overall data science architecture. This workshop does not replace the need of in-depth training on each Azure service covered.

* The services covered in this course are only a subset of a much larger family of Azure services. Similar outcomes can be achieved by leveraging other services and/or features not covered by this workshop. Specific business requirements may require the use of different services or features not included in this workshop.

* Some concepts presented in this course can be quite complex and you may need to seek more information from different sources to compliment your understanding of the Azure services covered.

![](/images/ml-lifecycle.png)

## Document Structure

This document contains detailed step-by-step instructions on how to implement a Data Science platform architecture using Azure Services. It’s recommended you carefully read the detailed description contained in this document for a successful experience with all Azure services.

You will see the label **IMPORTANT** whenever a there is a critical step to the lab. Please pay close attention to the instructions given.

For each lab it will be required you create some resources (Azure Machine Learning Workspace, Azure Databricks Workspace, etc.) in your Azure Subscription. For do so it will be expected your create the resource manually (it won`t be provided templates or script to create the services automatically). To help you it will be add in each lab the documentations that could be useful to do that.

## Prerequisites

The following prerequisites must be completed before you start these labs:

* You must be connected to the internet;

* Use either Edge or Chrome when executing the labs. Internet Explorer may have issues when rendering the UI for specific Azure services.

* You must have a Pay-As-You-Go Azure account with administrator - or contributor-level access to your subscription. If you don't have an account, you can sign up for an account following the instructions here: https://azure.microsoft.com/en-au/pricing/purchase-options/pay-as-you-go/.

    <br>**IMPORTANT**: Azure free subscriptions have quota restrictions that prevent the workshop resources from being create successfully. Please use a Pay-As-You-Go subscription instead.

    <br>**IMPORTANT**: When you create the lab resources in your own subscription you are responsible for the charges related to the use of the services provisioned. For more information about the list of services and tips on how to save money when executing these labs, please visit the [Azure Cost Management Documentation](https://docs.microsoft.com/en-us/azure/cost-management-billing/cost-management-billing-overview#:~:text=%20Understand%20Azure%20Cost%20Management%20%201%20Plan,the%20Azure%20Cost%20Management%20%20Billing...%20More%20).

* Please create the resources in a separated Resource Group.

## Lab Guide

Throughout a series of X labs you will progressively implement a data science architecture using datasets developed specially for this Lab 🤩.

You will start by developing a *churn prediction model* using Azure Databricks. To do this, you must import a dataset with customer information from Empresa Contoso already classified with information from customers who previously canceled the service or not.

By the end of the workshop you will have implemented the lab architecture referenced below:

![](/images/data-science-architecture.png)

## [Lab 0: Train a churn prediction model using Azure Databricks](labs/lab%200/Lab0.md)

In this lab you will use Azure Databricks to develop a simple Machine Learning model to predict customer`s churn. The customer is the most important asset for Contoso, and they need to develop actions that can mitigate this problem. So in this lab you will be able to help them to provide a mean to act in this way.

You will train a machine learning model with `sklearn` framework. After the training you will track the model using [`mlflow`](https://docs.microsoft.com/pt-br/azure/databricks/applications/mlflow/) package as well. MLflow is an open source platform for managing the end-to-end machine learning lifecycle. It has the following primary components: tracking, models, projects, model registry and model serving. In this lab we will demonstrate how to use the tracking capabilities to be able to deploy the model after this Lab.

The estimated time to complete this lab is: **60 minutes**

## [Lab 1: Deploy the machine learning model](labs/lab%201/Lab1.md)

In this lab you will use Azure Databricks to deploy the machine learning model trained in the previous lab. For this, we will use the integration capability of Azure Databricks and Azure Machine Learning.

The estimated time to complete this lab is: **45 minutes**