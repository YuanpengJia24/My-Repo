# Domestic Car Accident Severity Prediction Project

This project aims to predict the severity of car accidents in the United States based on various factors. Using statistical and machine learning approaches, the project explores the factors influencing accident severity and assesses the performance of various predictive models to aid in promoting safer driving behaviors.

## Table of Contents
1. [Introduction](#introduction)
2. [Project Overview](#project-overview)
3. [Setup](#setup)
4. [Data Cleaning and Preprocessing](#data-cleaning-and-preprocessing)
5. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
6. [Machine Learning Modeling](#machine-learning-modeling)
7. [Model Comparison and Conclusion](#model-comparison-and-conclusion)
8. [Challenges and Obstacles Faced](#challenges-and-obstacles-faced)
9. [Further Exploration](#further-exploration)

---

### Introduction
Welcome to our Car Accident Analysis project!

Our team—Yicheng Shen, Leshan Feng, and Yuanpeng Jia—has set out to analyze a comprehensive dataset of car accidents across the United States to identify key factors influencing accident occurrence and severity. Leveraging a dataset covering 49 states with around 2.8 million accident records, our goal is to develop a model that accurately predicts the likelihood and severity of car accidents.

---

### Project Overview
**Part 1: Project Design**
The primary objective is to analyze factors affecting car accident severity using supervised and unsupervised machine learning models. This project includes:
- Feature reduction through unsupervised learning.
- Supervised model comparisons to determine the most effective predictive model.
- Insights and recommendations for promoting safer driving habits.

### Setup
**Part 0: Before Running**
- **Prerequisites**: A Kaggle account is needed to access the dataset.
- **Data Access**: Upload `kaggle.json` to Colab to retrieve the [US Accidents dataset](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents).
- **Execution Note**: Ensure `kaggle.json` is loaded for full data visualization compatibility in Colab.

---

### Data Cleaning and Preprocessing
**Part 1**
- Loaded and cleaned the accident dataset, removing unnecessary or incomplete entries.
- Conducted preprocessing to format data for analysis, focusing on handling missing values and optimizing feature relevance.

---

### Exploratory Data Analysis (EDA)
**Part 2**
- Visualizations were created to explore trends and patterns across various accident factors, such as weather, time, location, and demographic aspects.
- EDA helped reveal key insights into high-risk factors that contribute to accident severity, guiding feature selection for modeling.

---

### Machine Learning Modeling
**Part 3**
- Utilized unsupervised learning techniques to reduce dimensionality, such as **Principal Component Analysis (PCA)**.
- Experimented with multiple supervised machine learning models to predict accident severity:
  - **Decision Trees**
  - **Logistic Regression**
  - **Support Vector Machines (SVM)**
  - **Random Forest Classifiers**
- Models were evaluated based on predictive accuracy and feature importance.

---

### Model Comparison and Conclusion
**Part 4**
- **Model Performance**: Each model's performance was compared to identify the most accurate predictor of accident severity.
- **Best Model**: The **Random Forest Classifier** achieved the highest accuracy and robustness, effectively capturing complex relationships among features.
- **Conclusion**: While Random Forest emerged as the top model, continued tuning and feature engineering could improve results further.

---

### Challenges and Obstacles Faced
**Part 5**
- Data quality posed significant challenges due to missing values and incomplete records.
- The high-dimensionality of features made it difficult to isolate impactful factors, underscoring the need for dimensionality reduction techniques like PCA.

---

### Further Exploration
**Part 6**
- Future steps include expanding the dataset to incorporate additional factors, such as real-time traffic data, for more granular predictions.
- Considering other machine learning models or ensemble methods could further improve predictive power.

---

### Requirements

- Python 3.x
- Libraries: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
- Kaggle API (for data access)

---

### Acknowledgments
This project uses the [US Accidents dataset](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents) provided by Sobhan Moosavi on Kaggle.
