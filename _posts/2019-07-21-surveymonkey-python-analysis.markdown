---
layout: post
title: "Analyze Survey Monkey data using Python"
date: 2019-07-21T18:22:25+02:00
tags: [Survey Monkey, Python, visualization, multi-answer, single-answer, matrix table, checkboxes, multi-choice question, by group]
categories: [User Experience, Data Science]
---

## Introduction 

Survey Monkey is an online survey software that helps you to create and run online surveys. It is also possible to visualize the survey results in Survey Monkey (for a single survey).

![alt_text](/assets/1_1.png "surveymonkey_visualization")


However, if we want to compare the same question across several different surveys, it is difficult to do so directly in Survey Monkey. Luckily, we can solve the problem by using Python.

This post will show you how to analyze survey data directly downloaded from Survey Monkey from multiple surveys in Python. To be more specific, I will first explain show how to import Survey Monkey data into Python and automatically generate a codebook, and then share my code for visualizing the survey results for three types of survey questions: 1) checkboxes (multi-answer question), 2) multiple choice (single-answer question), and 3) matrix table.

For all the codes below, I used python 3.7 and imported the following packages:


```python
#pandas==0.24.2
import pandas as pd
#numpy==1.16.2
import numpy as np
#seaborn==0.9.0
import seaborn as sns
#matplotlib==3.0.3
import matplotlib.pyplot as plt
#set plot style
sns.set_palette("Set2")
```



## Import Survey Monkey data into Python


### Step 1: Download from Survey Monkey

I first created an example survey on Survey Monkey including the aforementioned three types of questions. I also manually created some fake responses in order to demonstrate how the raw data look like after exporting from Survey Monkey.


![alt_text](/assets/1_2.png "example_survey"){:width="60%"}


You can export data from Survey Monkey easily by going to the “Analyze results” tab in a Survey and then click ”Save as”...”All response data”. It is recommended to download the "Actual answer text" so that you do not need to prepare the codebook manually.



![alt_text](/assets/1_3.png "download_data")


Please export all the survey data that you want to merge later in Python from Survey Monkey, rename the csv files inside the zip file, and move them into the working directory. 


### Step 2: Import Survey Monkey data into Python

We can use the pandas package to import the data as dataframe. Please note the data downloaded directly from Survey Monkey actually includes two rows of headers. They can be used as a codebook to better understand the meaning of each column.


![alt_text](/assets/1_4.png "read_surveymonkey_data_with_header"){:width="80%"}


As we can see from the picture above, the original column names are very complicated. In order to easily refer to each column later in the data analysis process, I decided to replace the header in a dataframe using numerical values with a prefix of “Q”. 

In the following screenshot, you can see that I imported two datasets from two Surveys that share the same question and merged them together into one dataframe.


![alt_text](/assets/1_5.png "read_surveymonkey_data_without_header"){:width="80%"}



## Data visualization by group

In order to demonstrate the use of the following functions, I created more fake data with the python library of [Faker](https://github.com/joke2k/faker). You can see the complete code for all my analysis in this [notebook](https://github.com/RuoyunLin/code_snippets/blob/master/survey_monkey_python_analysis/CodeExamples_ForSurveyMonkeyData.ipynb).


### How to visualize the multi-answer question by group? 


**Example code:**
```python
# Please customize the manual coding here
def score_to_numeric(x):
    if x=='Strongly agree':
        return 5
    if x=='Agree':
        return 4
    if x=='Undecided':
        return 3
    if x=='Disagree':
        return 2
    if x=='Strongly disagree':
        return 1

def numerical_describe(data):
    res = []
    cols = data.columns
    for col in cols:
        x = data[col].astype(np.float)
        res.append([col, "%.2f" % x.mean(), "%.2f" % x.std(), x.count()])
    return pd.DataFrame(columns=['variable','mean','std','n'], 
                        data=res).set_index('variable').sort_values(by=['mean'],ascending=False)

def gen_sub_table(data, group_name, col_range):
    
    data_sub = data[data['group']==group_name].iloc[:,col_range].dropna(how='all')
    
    for var in data_sub.columns:
        data_sub[var] = data_sub[var].apply(score_to_numeric)

    return data_sub

def gen_table(data, group_name, col_range):
    data_sub = gen_sub_table(data, group_name, col_range)
       
    data_describe = numerical_describe(data_sub)
    
    index = []
    
    for var in data_describe.index:
        i=int(var[1:])
        index.append(codebook.iloc[i,1])
    
    data_describe['item'] = index
    
    return data_describe
```


**Input:**


*   Data: Name of the dataframe
*   Column_list: A list of columns for the multi-answer questions

**Output:**

![alt_text](/assets/1_6.png "output_checkboxes"){:width="80%"}



### How to visualize the single-answer question by group?

**Example code:**
```python
def gen_chart_radiobutton(data, question_name, index):
    print("Number of answers in each group: ")
    print(data[[question_name,'group']].groupby('group').count())
    i_counts = (data.groupby(['group'])[question_name]
                         .value_counts(normalize=True)
                         .rename('percentage(%)')
                         .mul(100)
                         .reset_index().round(2)
               )
    
    listOfGroup = list(df.group.unique())
    listOfGroup.sort()
    
    fig, ax = plt.subplots(figsize=(10,8))
    fig = sns.barplot(x="percentage(%)", y=question_name, order=index, hue="group", hue_order = listOfGroup, data=i_counts)
    
    
    plt.title(codebook.iloc[int(question_name[1:]),0])
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title='group')
    ax.set(xlim = (0,100))
    ax.set_ylabel('') 
    return plt.show()
```


**Input:**



*   Data: Name of the dataframe
*   Question_name: Column name of the question
*   Index: a list of the categorical labels in an order that makes the most sense for understanding the chat

**Output:**

![alt_text](/assets/1_7.png "output_multi_choice"){:width="80%"}



### How to visualize the matrix table by group?

**Example code:**
```python
def gen_chart_checkbox(data, column_list):
    listOfGroup = list(data.group.unique())
    listOfGroup.sort()
    table_sum = prepare_data_summary(data, column_list, listOfGroup)
    fig, ax = plt.subplots(figsize=(10,8))
    ax = sns.barplot(x = "percentage(%)", y = 'options', hue = 'group', hue_order = listOfGroup, data = table_sum)
    ax.set(xlim = (0,100))
    ax.set_ylabel('') 
    plt.title(codebook.iloc[column_list[0],0],fontsize=15)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title='group')
    return plt.show()

def prepare_data_summary(data, column_list, groups):
    first_group, *other_groups = groups
    table_sum = prepare_sub_data(data, codebook, 'group', first_group, column_list)
    for group in other_groups:
        table_sum = table_sum.append(prepare_sub_data(data, codebook, 'group', group, column_list))
    return table_sum

def prepare_sub_data(data, codebook, group, group_name, column_list):
    col_name = codebook.options[list(column_list)].reset_index(drop = True)
    data_sub = data[data[group]==group_name].iloc[:,column_list].dropna(how = 'all') 
    s_count = data_sub.notnull().sum()
    s_per = pd.Series(s_count.to_numpy()/len(data_sub)*100, name="percentage(%)")
    print(group_name + " Answered: " + str(len(data_sub)))
    table_sum = pd.concat([s_per, col_name], axis=1)
    table_sum[group] = group_name
    return table_sum
```

```python
#Generate a chart to compare the importance of missing features across two groups
def compare_importance(data, groups, col_range):
    group_name_to_describe_data = {}
    for i,group_name in enumerate(groups):
        group_name_to_describe_data[group_name]='data_describe_%s'%i
    for group_name in groups:
        group_name_to_describe_data[group_name] = gen_table(data, group_name, col_range)
    #Visualize the mean value with the 95% confidence interval
    plt.figure(num=None, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
    
    for group_name in groups:
        plt.errorbar(group_name_to_describe_data[group_name]['mean'].astype(float), group_name_to_describe_data[group_name]['item'], xerr=1.96*(group_name_to_describe_data[group_name]['std'].astype(float)/(group_name_to_describe_data[group_name]['n']**.5)), fmt='o',elinewidth = .5, capsize = 4, marker='o', ms=4, label=group_name)

    plt.legend(loc='lower right')
    plt.title('Compare the mean values across groups (scale 1-5)')
    plt.show()
```

**Input:**



*   Data: Name of the dataframe
*   Groups: Group names of surveys that you are interested in comparing
*   Col_range:  A list of columns for the matrix question

Note: Here please customize the function of score_to_numeric above in order to convert the text labels in the raw data into meaningful numbers that you can interpret

**Output:**



![alt_text](/assets/1_8.png "output_compare_matrix"){:width="80%"}
