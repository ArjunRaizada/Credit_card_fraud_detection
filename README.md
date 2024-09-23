# K-Nearest Neighbors Classification

## Overview

**K-Nearest Neighbors (KNN)** is a supervised learning algorithm used for classification tasks. It predicts the class of a given data point by considering the classes of the 'K' nearest data points and selecting the class that appears most frequently among them.

### Visualization of the K-Nearest Neighbors Algorithm

![K-Nearest Neighbors](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/images/KNN_Diagram.png)

In the example above, if we consider a k value of 3, the prediction is Class B. If we consider a k value of 6, the prediction changes to Class A.

## Table of Contents

1. [About the Dataset](#about-the-dataset)
2. [Data Visualization and Analysis](#data-visualization-and-analysis)
3. [Classification](#classification)

---

## About the Dataset

Imagine a telecommunications provider that has segmented its customer base into four groups based on service usage patterns. The objective is to predict group membership using demographic data (region, age, marital status) to customize offers for prospective customers.

The target field, **custcat**, has four possible values:
- 1: Basic Service
- 2: E-Service
- 3: Plus Service
- 4: Total Service

We aim to build a KNN classifier to predict the class of unknown cases.

**Did you know?** IBM offers a unique opportunity for businesses with 10 TB of IBM Cloud Object Storage: [Sign up now for free](http://cocl.us/ML0101EN-IBM-Offer-CC)

## Data Visualization and Analysis

### Class Distribution

```python
df['custcat'].value_counts()
```

This results in:

| Class             | Count |
|-------------------|-------|
| 1: Basic Service   | 281   |
| 2: E-Service      | 266   |
| 3: Plus Service   | 236   |
| 4: Total Service  | 217   |


### Feature Set

The feature set is defined as follows:

```python
X = df[['region', 'tenure', 'age', 'marital', 'address', 'income', 'ed', 'employ', 'retire', 'gender', 'reside']].values
y = df['custcat'].values
```

### Normalize Data

Data standardization is performed to give the data zero mean and unit variance, which is crucial for distance-based algorithms like KNN.

```python
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
```

### Train/Test Split

To evaluate out-of-sample accuracy, the dataset is split into training and testing sets.

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
```

### Classification

#### K-Nearest Neighbors (KNN)

**Import Library**

```python
from sklearn.neighbors import KNeighborsClassifier
```

#### Training the Model

Set `k = 4` and train the model:

```python
k = 4
neigh = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
```


#### Making Predictions

```python
yhat = neigh.predict(X_test)
```

#### Accuracy Evaluation

```python
from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))
```


#### Exploring Different K Values

To find the optimal k, we test various values and plot the accuracy:

```python
Ks = 10
mean_acc = np.zeros((Ks-1))

for n in range(1, Ks):
    neigh = KNeighborsClassifier(n_neighbors=n).fit(X_train, y_train)
    yhat = neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

plt.plot(range(1, Ks), mean_acc, 'g')
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Accuracy')
plt.tight_layout()
plt.show()
```


#### Determining the Best Accuracy

The best accuracy is determined as:

```python
print("The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax() + 1)
```

## Author 
Arjun Raizada
