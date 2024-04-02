"""
Ahnaf Tajwar
Class: CS 677
Date: 3/30/23
Homework Problem # 1-6
Description of Problem (just a 1-2 line summary!): These problems are to compute some statistics from data in txt file.
    It also is to create test and train data for the dataset and apply a simple classifier, kNN, and logistic regression.
    The accuracy results of each is then compared along with the effect of dropping different features.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


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

# Splitting train and test data
X_train, X_test, Y_train, Y_test=train_test_split(X,Y, test_size=0.5, random_state=1)

# Pairplot for training data only
train_df = pd.concat([X_train, Y_train], axis=1)

all_bills_pairplot = sns.pairplot(train_df, hue="class")
all_bills_pairplot.savefig('all_bills_pairplot.pdf')
# plt.show()

X_train_df = pd.concat([X_train, Y_train], axis=1)

X_train_df_class_0 = X_train_df[X_train_df['class'] == 0].drop(columns='class')

# Pairplot for training data with class value 0
good_bills_pairplot = sns.pairplot(X_train_df_class_0)
good_bills_pairplot.savefig('good_bills_pairplot.pdf')
# plt.show()

X_train_df_class_1 = X_train_df[X_train_df['class'] == 1].drop(columns='class')

# Pairplot for training data with class value 1
fake_bills_pairplot = sns.pairplot(X_train_df_class_1)
fake_bills_pairplot.savefig('fake_bills_pairplot.pdf')
# plt.show()

test_df = pd.concat([X_test, Y_test], axis=1)

print(test_df)

# Simple Classifier
print("\n-----------------Simple Classifier------------------\n")

test_df['simple_classifier'] = np.where((test_df['f1'] > 1) & (test_df['f2'] > 2) & (test_df['f3'] < 2), 0, 1)
print(test_df)

# Metrics
tp = ((test_df['simple_classifier'] == 0) & (test_df['class'] == 0)).sum()
print('True Positives: ', tp)

fp = ((test_df['simple_classifier'] == 1) & (test_df['class'] == 0)).sum()
print('False Positives: ', fp)

tn = ((test_df['simple_classifier'] == 1) & (test_df['class'] == 1)).sum()
print('True Negatives: ', tn)

fn = ((test_df['simple_classifier'] == 0) & (test_df['class'] == 1)).sum()
print('False Negatives: ', fn)

tpr = tp / (tp + fn)
print('True Positive Rate: ', tpr) 

tnr = tn / (tn + fp)
print('True Negative Rate: ', tnr) 

accuracy = (tp + tn) / (tp + tn + fp + fn)
print('Accuracy: ', accuracy) 

metrics_df = pd.DataFrame({
    'Metric': ['TP', 'FP', 'TN', 'FN', 'accuracy','TPR', 'TNR'],
    'Value': [tp, fp, tn, fn, accuracy, tpr, tnr]
})
metrics_df = metrics_df.set_index('Metric').transpose()
print(metrics_df)

# kNN Classifier
print("\n---------------------kNN----------------------\n")

X = df[['f1', 'f2', 'f3', 'f4']].values
Y = df[['class']].values

X_train, X_test, Y_train, Y_test=train_test_split(X,Y, test_size=0.5, random_state=1)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

error_rate = []
accuracy_scores = {}
k_values = [3, 5, 7, 9, 11]
for k in k_values:
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X_train, Y_train.ravel())
    pred_k = knn_classifier.predict(X_test)
    error_rate.append(np.mean(pred_k != Y_test))
    # Compute the accuracy of the classifier on the test data
    accuracy = accuracy_score(Y_test, pred_k)
    
    # Store the accuracy score for this value of k
    accuracy_scores[k] = accuracy

print("Error Rate :", error_rate)

for k, accuracy in accuracy_scores.items():
    print(f"Accuracy for k={k}: {accuracy:.4f}")

# Plot for k vs accuracy
plt.figure()
plt.plot(k_values, list(accuracy_scores.values()), marker='o', linestyle='-')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. k for k-NN Classifier')
plt.grid(True)
plt.xticks(k_values)
plt.savefig('k_accuracy_plot.pdf')
# plt.show()

# Find the optimal value of k (highest accuracy)
optimal_k = max(accuracy_scores, key=accuracy_scores.get)
print(f'Optimal value of k: {optimal_k}, Accuracy: {accuracy_scores[optimal_k]:.4f}')

# Use the optimal value of k = 7
optimal_k = 7

# Initialize kNN classifier with optimal k
knn_classifier = KNeighborsClassifier(n_neighbors=optimal_k)

# Train the classifier on the training data
knn_classifier.fit(X_train, Y_train.ravel())  # Use .values.ravel() to convert Y_train to a 1D array

# Predict the classes for the test data
Y_pred = knn_classifier.predict(X_test)

Y_test_values = Y_test.ravel()

# True Positive
tp = ((Y_pred == 0) & (Y_test_values == 0)).sum()

# False Positive
fp = ((Y_pred == 1) & (Y_test_values == 0)).sum()

# True Negative
tn = ((Y_pred == 1) & (Y_test_values == 1)).sum()

# False Negative
fn = ((Y_pred == 0) & (Y_test_values == 1)).sum()

# Accuracy
accuracy = (tp + tn) / (tp + tn + fp + fn)

# True Positive Rate
tpr = tp / (tp + fn)

# True Negative Rate
tnr = tn / (tn + fp)

# Create a dataframe to summarize the performance measures
performance_measures = pd.DataFrame({
    'Metric': ['TP', 'FP', 'TN', 'FN', 'accuracy', 'TPR', 'TNR'],
    'Value': [tp, fp, tn, fn, accuracy, tpr, tnr]
})
performance_measures = performance_measures.set_index('Metric').transpose()
print(performance_measures)

# Simple classifier UID
f1 = 8
f2 = 2
f3 = 6
f4 = 0

if f1 > 1 and f2 > 2 and f3 < 2:
    simple_predicted = 0
else:
    simple_predicted = 1

print("With UID 8260, predicted label is :", simple_predicted)

X_uid = scaler.transform([[8, 2, 6, 0]])

# Initialize k-NN classifier with k = 7
knn_classifier = KNeighborsClassifier(n_neighbors=7)

# Train the classifier on the training data
knn_classifier.fit(X_train, Y_train.ravel())  # Use .values.ravel() to convert Y_train to a 1D array

# Predict the class label for the standardized features
predicted_class = knn_classifier.predict(X_uid)

# Display the predicted class label
print("Predicted class label using kNN:", predicted_class[0])

# f1 drop--------------------------------------------------------
print("Dropping f1...")
X = df[['f2', 'f3', 'f4']].values
Y = df[['class']].values

X_train, X_test, Y_train, Y_test=train_test_split(X,Y, test_size=0.5, random_state=1)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

accuracy_scores = {}
k = 7

knn_classifier = KNeighborsClassifier(n_neighbors=7)
knn_classifier.fit(X_train, Y_train.ravel())
pred_k = knn_classifier.predict(X_test)

# Compute the accuracy of the classifier on the test data
accuracy = accuracy_score(Y_test, pred_k)

# Store the accuracy score for this value of k
accuracy_scores[k] = accuracy

for k, accuracy in accuracy_scores.items():
    print(f"Accuracy for k={k} for f2, f3, f4: {accuracy:.4f}")

# f2 drop--------------------------------------------------------
print("Dropping f2...")
X = df[['f1', 'f3', 'f4']].values
Y = df[['class']].values

X_train, X_test, Y_train, Y_test=train_test_split(X,Y, test_size=0.5, random_state=1)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

accuracy_scores = {}
k = 7

knn_classifier = KNeighborsClassifier(n_neighbors=7)
knn_classifier.fit(X_train, Y_train.ravel())
pred_k = knn_classifier.predict(X_test)

# Compute the accuracy of the classifier on the test data
accuracy = accuracy_score(Y_test, pred_k)

# Store the accuracy score for this value of k
accuracy_scores[k] = accuracy

for k, accuracy in accuracy_scores.items():
    print(f"Accuracy for k={k} for f1, f3, f4: {accuracy:.4f}")

# f3 drop--------------------------------------------------------
print("Dropping f3...")
X = df[['f1', 'f2', 'f4']].values
Y = df[['class']].values

X_train, X_test, Y_train, Y_test=train_test_split(X,Y, test_size=0.5, random_state=1)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

accuracy_scores = {}
k = 7

knn_classifier = KNeighborsClassifier(n_neighbors=7)
knn_classifier.fit(X_train, Y_train.ravel())
pred_k = knn_classifier.predict(X_test)

# Compute the accuracy of the classifier on the test data
accuracy = accuracy_score(Y_test, pred_k)

# Store the accuracy score for this value of k
accuracy_scores[k] = accuracy

for k, accuracy in accuracy_scores.items():
    print(f"Accuracy for k={k} for f1, f2, f4: {accuracy:.4f}")

# f4 drop--------------------------------------------------------
print("Dropping f4...")
X = df[['f1', 'f2', 'f3']].values
Y = df[['class']].values

X_train, X_test, Y_train, Y_test=train_test_split(X,Y, test_size=0.5, random_state=1)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

accuracy_scores = {}
k = 7

knn_classifier = KNeighborsClassifier(n_neighbors=7)
knn_classifier.fit(X_train, Y_train.ravel())
pred_k = knn_classifier.predict(X_test)

# Compute the accuracy of the classifier on the test data
accuracy = accuracy_score(Y_test, pred_k)

# Store the accuracy score for this value of k
accuracy_scores[k] = accuracy

for k, accuracy in accuracy_scores.items():
    print(f"Accuracy for k={k} for f1, f2, f3: {accuracy:.4f}")

# Logistic Regression
print("\n--------------Logistic Regression---------------\n")

features = ['f1', 'f2', 'f3', 'f4']
class_labels = [0, 1]

X = df[features].values

le = LabelEncoder()
Y = le.fit_transform(df['class'].values)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=1)
log_reg_classifier = LogisticRegression()
log_reg_classifier.fit(X_train, Y_train)

prediction = log_reg_classifier.predict(X_test)
accuracy = np.mean(prediction == Y_test)

print("Accuracy for logistic regression: ", accuracy)

# Compute TP, FP, TN, FN
tp = ((prediction == 0) & (Y_test == 0)).sum()
fp = ((prediction == 1) & (Y_test == 0)).sum()
tn = ((prediction == 1) & (Y_test == 1)).sum()
fn = ((prediction == 0) & (Y_test == 1)).sum()

# Compute accuracy
accuracy = (tp + tn) / (tp + tn + fp + fn)

# Compute TPR and TNR
tpr = tp / (tp + fn)
tnr = tn / (tn + fp)

performance_measures = pd.DataFrame({
    'Metric': ['TP', 'FP', 'TN', 'FN', 'Accuracy', 'TPR', 'TNR'],
    'Value': [tp, fp, tn, fn, accuracy, tpr, tnr]
})

# Set index to Metric
performance_measures = performance_measures.set_index('Metric').transpose()

print("Performance Measures for Logistic Regression:")
print(performance_measures)

# Prediction on BUID
X_uid = [[8, 2, 6, 0]]

prediction = log_reg_classifier.predict(X_uid)

print("Predicted class label using logistic regression:", prediction[0])

# f1 drop--------------------------------------------------------
print("Dropping f1...")
features = ['f2', 'f3', 'f4']
class_labels = [0, 1]

accuracy_scores = {}

X = df[features].values

le = LabelEncoder()
Y = le.fit_transform(df['class'].values)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=1)
log_reg_classifier = LogisticRegression()
log_reg_classifier.fit(X_train, Y_train)

prediction = log_reg_classifier.predict(X_test)
accuracy = np.mean(prediction == Y_test)

print("Accuracy after dropping f1 for logistic regression: ", accuracy)

# f2 drop--------------------------------------------------------
print("Dropping f2...")
features = ['f1', 'f3', 'f4']
class_labels = [0, 1]

accuracy_scores = {}

X = df[features].values

le = LabelEncoder()
Y = le.fit_transform(df['class'].values)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=1)
log_reg_classifier = LogisticRegression()
log_reg_classifier.fit(X_train, Y_train)

prediction = log_reg_classifier.predict(X_test)
accuracy = np.mean(prediction == Y_test)

print("Accuracy after dropping f2 for logistic regression: ", accuracy)

# f3 drop--------------------------------------------------------
print("Dropping f3...")
features = ['f1', 'f2', 'f4']
class_labels = [0, 1]

accuracy_scores = {}

X = df[features].values

le = LabelEncoder()
Y = le.fit_transform(df['class'].values)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=1)
log_reg_classifier = LogisticRegression()
log_reg_classifier.fit(X_train, Y_train)

prediction = log_reg_classifier.predict(X_test)
accuracy = np.mean(prediction == Y_test)

print("Accuracy after dropping f3 for logistic regression: ", accuracy)

# f4 drop--------------------------------------------------------
print("Dropping f4...")
features = ['f1', 'f2', 'f3']
class_labels = [0, 1]

accuracy_scores = {}

X = df[features].values

le = LabelEncoder()
Y = le.fit_transform(df['class'].values)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=1)
log_reg_classifier = LogisticRegression()
log_reg_classifier.fit(X_train, Y_train)

prediction = log_reg_classifier.predict(X_test)
accuracy = np.mean(prediction == Y_test)

print("Accuracy after dropping f4 for logistic regression: ", accuracy)