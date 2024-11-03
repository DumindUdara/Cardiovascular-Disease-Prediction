# Cardiovascular Disease Prediction

### Project Introduction
This project focuses on analyzing cardiovascular health data using machine learning techniques to classify and predict cardiovascular conditions. By leveraging a range of data preprocessing, model selection, and evaluation steps, this project aims to identify significant patterns and factors that contribute to cardiovascular risk.

### Workflow Summary
1. **Data Preprocessing**: The dataset is cleaned and transformed, including encoding categorical variables and standardizing numeric features, to prepare for modeling.
2. **Model Selection and Training**: Several machine learning models—such as Logistic Regression, Random Forest, K-Nearest Neighbors (KNN), and Support Vector Classifier (SVC)—are trained to classify cardiovascular risk.
3. **Ensemble Modeling**: An ensemble approach using Voting Classifier combines multiple models to improve prediction accuracy.
4. **Evaluation**: Model performance is evaluated using accuracy, precision, recall, and other metrics to assess predictive capability.

This project uses machine learning to predict the likelihood of cardiovascular disease in patients based on health indicators. It leverages libraries such as **scikit-learn** for data preprocessing, feature engineering, and model building.

<br>

![GitHub Repo stars](https://img.shields.io/github/stars/yourusername/your-repo-name?style=social)
![GitHub last commit](https://img.shields.io/github/last-commit/yourusername/your-repo-name)
![GitHub issues](https://img.shields.io/github/issues/yourusername/your-repo-name)
![GitHub forks](https://img.shields.io/github/forks/yourusername/your-repo-name?style=social)
![Python](https://img.shields.io/badge/Python-3.8-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-0.24-orange)
![Data Source](https://img.shields.io/badge/dataset-Kaggle-blue)
![Status](https://img.shields.io/badge/status-active-brightgreen)
![Contributions](https://img.shields.io/badge/contributions-welcome-brightgreen)


## Dataset
- **Dataset Name**: Cardiovascular Disease Dataset
- **Source**: [Kaggle](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)
- **Description**: This dataset contains health information of patients with features such as age, gender, blood pressure, cholesterol levels, etc.



<br><hr>

### More Details 

Here’s an explanation of each algorithm used in my project:

### 1. **Logistic Regression**
   - **Description**: Logistic Regression is a statistical model commonly used for binary classification. It estimates the probability that a given input belongs to one of two classes (e.g., disease or no disease).
   - **Why it’s used**: This algorithm is a good starting point for classification tasks because it’s simple, interpretable, and works well with binary outcomes like cardiovascular disease predictions.
   - **How it works**: Logistic regression calculates the probability of a binary outcome using a logistic function (a type of sigmoid curve), transforming predictions into values between 0 and 1, which can be interpreted as probabilities.

### 2. **Random Forest Classifier**
   - **Description**: Random Forest is an ensemble learning algorithm that builds multiple decision trees and merges their predictions for a more accurate and stable output.
   - **Why it’s used**: Random Forest can handle high-dimensional datasets well and is less prone to overfitting due to its use of multiple trees. It’s also highly interpretable in terms of feature importance.
   - **How it works**: Each decision tree in the forest makes a prediction, and the final output is determined by the majority vote (for classification) of all the trees. Random Forests use a technique called "bagging," where each tree is trained on a random subset of the data and features, making it robust against noise and variance.

### 3. **K-Nearest Neighbors (KNN)**
   - **Description**: KNN is a non-parametric, instance-based learning algorithm that classifies data points based on the classes of their k nearest neighbors.
   - **Why it’s used**: KNN is useful for understanding patterns in smaller or simpler datasets and can be a good choice for initial model comparison. It’s also easy to implement and intuitive.
   - **How it works**: For each data point, the algorithm calculates the distances to all other points in the training set and identifies the ‘k’ closest points. The class that appears most frequently among these neighbors is assigned to the new point.

### 4. **Support Vector Machine (SVM)**
   - **Description**: SVM is a powerful classification algorithm that aims to find the optimal boundary (or hyperplane) that separates classes with the maximum margin.
   - **Why it’s used**: SVM is effective in high-dimensional spaces and is often used when clear class boundaries are desired. It works well for binary classification and can be effective with datasets with many features.
   - **How it works**: SVM identifies a hyperplane that best separates the data points of one class from those of another, maximizing the margin between the classes. In cases where data isn’t linearly separable, SVM can use kernel functions to project the data into a higher dimension.

### 5. **Voting Classifier (Ensemble Method)**
   - **Description**: The Voting Classifier is an ensemble algorithm that combines predictions from multiple classifiers (in this case, Logistic Regression, Random Forest, KNN, and SVM) to improve accuracy and generalization.
   - **Why it’s used**: By combining the strengths of several algorithms, ensemble methods like Voting Classifier can produce a more robust and accurate model, reducing the likelihood of poor performance due to a single weak algorithm.
   - **How it works**: The Voting Classifier aggregates the predictions from each of its individual classifiers. For a majority voting approach, the final prediction is the one that receives the most votes among the classifiers, which helps improve overall prediction accuracy.

### Final Model and Evaluation
Your model uses these algorithms to predict cardiovascular disease based on the dataset's health features. The final performance metrics indicate that the model achieves:
- **Precision**: 70% for class 0 (no disease) and 76% for class 1 (disease)
- **Recall**: 80% for class 0 and 64% for class 1
- **Accuracy**: 72%
  
<img width="465" alt="Screenshot 2024-11-03 at 23 56 27" src="https://github.com/user-attachments/assets/29314911-4e02-4d1a-bb1f-ba0dd33ee77e">

<br>
These metrics suggest a balanced model, with strengths in both classes but some areas for improvement, especially in recall for class 1 (disease).

By combining multiple classifiers, the Voting Classifier capitalizes on the strengths of each algorithm, making the overall model more accurate and robust for predicting cardiovascular disease.
