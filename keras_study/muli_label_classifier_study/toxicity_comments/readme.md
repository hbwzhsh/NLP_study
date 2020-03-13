# Toxicity Review Prediction

## Introduction

The project is divided into four parts

1. Text preprocessing on comment_text field
2. Data analysis
3. Model training
4. Model prediction

## Text preprocessing on comment_text field

Project path：data_analysis.ipynb

For preprocessing text for the comment_text field, we only need to do the following five parts：

1. Case conversion
2. Handling of punctuation
3. Participle
4. Remove stop words
5. Stemming

## Data analysis

Project path：data_analysis.ipynb

We analyze from both the label and the text:

1. From the analysis results of the label, it can be seen that the task is a multi-label task, and it belongs to the problem of data imbalance.；
2. From the analysis results of the text, it can be seen that the text length of the task is different, but sentences less than 100% account for about 95% of all, so we set max_len to 100；

## Model training

The model used in this project is text-cnn，text-rnn，text-rcnn

## Model prediction

To improve accuracy, we use the following steps：
1. Use text-cnn, text-rnn, text-rcnn to predict test data；
2. Using model fusion, weighted average of the prediction results of the three models 