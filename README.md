# Clustering Project Summary

## Project Goals

> - Document code, process (data acquistion, preparation, exploratory data analysis and statistical testing, modeling, and model evaluation), findings, and key takeaways.
> - Create modules that make our process repeateable and our report (notebook) easier to read and follow.
> - Ask exploratory questions of the data that will help us understand more about the attributes and drivers of error in the Zestimates. 
> - Make recommendations to a data science team about how to improve predictions.
> - Refine our work into a report, in the form of a jupyter notebook, that I will walk through in a 5 minute presentation to our data science team about our goals, the work we did, what we found, our methodologies, and our conclusions.
> - Be prepared to answer panel questions about my code, process, findings and key takeaways, and model.

## Project Description

### Business Goals

> - Utilize clustering to create new features for exploration and model prediction
> - Find the key drivers of error in the Zestimates.
> - Deliver a report that the data science team can read through and replicate, understand what steps were taken, why and what the outcome was.
> - Make recommendations on what works or doesn't work in predicting error in the Zestimates.

### Deliverables

> - **Readme (.md)**
> - **A Python Module or Modules that automate the data acquisition and preparation process
> - **Final Report (.ipynb)**
> - 5 min Recorded Presentation

## Data Dictionary

|Target|Definition
|:-------|:----------|
|logerror|The total tax assessed value of the property|

|Feature|Definition|
|:-------|:----------|
|bedrooms                 |Number of bedrooms in home|
|bathrooms                |Number of bathrooms in home including fractional bathrooms|
|lat                      |Latitude of the middle of the parcel multiplied by 10e6|
|lon                      |Longitude of the middle of the parcel multiplied by 10e6|
|house_age                |How old the property is in years| 
|taxrate                  |The total property tax assessed for that assessment year divided by the total tax assessed value of the home|
|fips                     |Federal Information Processing Standard code|
|acres                    |Acreage of the home|
|tot_sqft                 |Calculated total finished living area of the home|
|land_dollar_per_sqft     |The assessed value of the land area of the home divided by the lot size in sqft|
|structure_dollar_per_sqft|The assessed value of the built structure on the house divided by the tot_sqft|


## Initial Hypotheses
> We believe that by utilizing clustering methodology we will be able to discover new things about our dataset in order to discover why models are predicting as accurately as they should.  We believe that the clusters we create will add new dimensions to our future models thus increasing its accuracy. 

## Executive Summary - Key Findings and Recommendations
> 1. Utilizing three different cluster groups (area_cluster, size_cluster, price cluster) we were not able to increase the accuracy of our models in any meaningful way.

> 2. Many of the newly created features via feature engineering and clustering did nothing to add any value to the zillow dataset in terms of model creation, but it did help us futher our exploration and ask new questions we didn't know to ask beforehand. 

> 3. Our recommendations are that we maybe delve deeper into new clusters if given more time possibly brining in other features such as airconditioningtype and maybe looking into more time sensitive data such as transaction dates, etc.

## Project Plan

### Planning Phase

> - Created a README.md file.
> - Imported all tools and files needed to properly conduct the project.

### Acquire Phase

> - Utilized wrangle.py to pull zillow data from a Codeup database.

### Prepare Phase

> - Utilized wranlge.py to clean up zillow dataset.
> - Split the overall dataset into my train, validate, and test datasets to be used later.
> - Utilized modeling.py to scale the datasets in order to properly cluster the dataset for exploration.

### Explore Phase

> - Asked questions of the data and utilized visualizations and hypothesis testing to answer said questions.
> - Used clustering to further our exploration and created visualizations that enabled us to see the data in a new perspective.

### Model Phase

> - Set up the baseline prediction for future models to base their success on.
> - Trained multiple models for each type of Regression technique (Ordinary Least Squares, LASSO + LARS, and Polynomial Regression Model)
> - Validated all models to narrow down my selection to the best performing model.
> - Chose the MVP of all created models and used the test data set to ensure the best model worke entirely to expectations.

### Deliver Phase

> - Prepped our final notebook with a clean presentation to best present our findings and process to the data science team.

## How To Reproduce My Project

> 1. Read this README.md.
> 2. Download the wrangle.py, formating.py, tests.py, viz.py, modeling.py, and final_report.ipynb files into your directory along with your own env file that contains your user, password, and host variables,
> 3. Run our final_report.ipynb notebook
