# Lab 3: Monitor and collect data from ML web service endpoints

In this lab you see how to monitor the machine learning endpoints we deployed in the previous labs.

Azure Machine Learning integrates with [Application Insights](https://docs.microsoft.com/en-us/azure/azure-monitor/app/app-insights-overview) that is a feature of [Azure Monitor](https://docs.microsoft.com/en-us/azure/azure-monitor/overview). So, with this, you can monitor your application (in our case the API 😁) and collect useful insights and logs of endpoint's usage.

To know more please take a look in this [link](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-enable-app-insights/).

## Microsoft Learn & Technical Documentation

The following Azure services will be used in this lab. If you need further training resources or access to technical documentation please find in the table below links to Microsoft Learn and to each service's Technical Documentation.

Azure Service | Microsoft Learn | Technical Documentation|
--------------|-----------------|------------------------|
Azure Machine Learning| [Monitor models with Azure Machine Learning
](https://docs.microsoft.com/en-us/learn/paths/build-ai-solutions-with-azure-ml-service/) | [Monitor and collect data from ML web service endpoints](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-enable-app-insights/)
Azure Monitor| [Analyze your Azure infrastructure by using Azure Monitor logs](https://docs.microsoft.com/en-us/learn/modules/analyze-infrastructure-with-azure-monitor-logs/) | [Azure Monitor Technical Documentation](https://docs.microsoft.com/pt-br/azure/azure-monitor/overview)

## Lab Architecture

![1](/images/data-science-architecture-lab-3.png)

Step     | Description
-------- | -----
![1](/images/Black1.png) | Enable Application Insights in your endpoint
![2](/images/Black2.png) | Display custom logs

### Enable App Insights and collect custom log in your endpoint

**IMPORTANT**|
-------------|
**Take a look in this [notebook](/labs/lab%203/notebooks/monitor-with-appinsights.ipynb).**|

You can also import this notebook to you own workspace. Just righ-click on the blank space below your username and choose *Import -> File* and put the path of the file.
