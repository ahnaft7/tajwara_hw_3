"""
Ahnaf Tajwar
Class: CS 677
Date: 3/30/23
Homework Problem # 1
Description of Problem (just a 1-2 line summary!): This problem is to 
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt


# read in text file
df = pd.read_csv('data_banknote_authentication.txt', names=['f1', 'f2', 'f3', 'f4', 'class'])

# Map class value to color
color_map = {0: 'green', 1: 'red'}

# Creating 'color' column based on 'class' value
df['color'] = df['class'].map(color_map)

print(df)

# Compute mean and standard deviation for each 'f' column for all class values
summary_class = df.groupby("class").agg({'f1': ['mean', 'std'],
                                        'f2': ['mean', 'std'],
                                        'f3': ['mean', 'std'],
                                        'f4': ['mean', 'std']}).round(2)

# Rename columns to match the desired format
summary_class.columns = [f"{col}_{stat}" for col, stat in summary_class.columns]

summary_all = df.groupby(df.index // len(df)).agg({'f1': ['mean', 'std'],
                                                   'f2': ['mean', 'std'],
                                                   'f3': ['mean', 'std'],
                                                   'f4': ['mean', 'std']}).round(2)

summary_all.columns = [f"{col}_{stat}" for col, stat in summary_all.columns]

# Add a row labeled "all" with the aggregated statistics
summary_all.index = ['all']

summary_combined = pd.concat([summary_class, summary_all])

print(summary_combined)

X = df[['f1', 'f2', 'f3', 'f4']]
Y = df[['class']]

X_train, X_test, Y_train, Y_test=train_test_split(X,Y, test_size=0.5, random_state=1)

# Pairplot for training data only
train_df = pd.concat([X_train, Y_train], axis=1)

all_bills_pairplot = sns.pairplot(train_df, hue="class")

# Save the plot to a file
all_bills_pairplot.savefig('all_bills_pairplot.pdf')

# Display the plot
plt.show()

X_train_df = pd.concat([X_train, Y_train], axis=1)

X_train_df_class_0 = X_train_df[X_train_df['class'] == 0].drop(columns='class')

# Pairplot for training data with class value 0
good_bills_pairplot = sns.pairplot(X_train_df_class_0)
good_bills_pairplot.savefig('good_bills_pairplot.pdf')
plt.show()

X_train_df_class_1 = X_train_df[X_train_df['class'] == 1].drop(columns='class')

# Pairplot for training data with class value 1
fake_bills_pairplot = sns.pairplot(X_train_df_class_1)
fake_bills_pairplot.savefig('fake_bills_pairplot.pdf')
plt.show()
