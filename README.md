# DSA 210 - Student Academic Performance Analysis

## Project Overview

This repository contains the final project for **DSA 210 - Introduction to Data Science**. The project investigates factors related to student academic performance and applies the full data science pipeline on educational datasets. The main goal is to understand which student-related variables are associated with academic success and to build machine learning models that can predict student performance levels.

The project started with the **Higher Education Students Performance Evaluation** dataset, which contains survey-based academic and demographic information about university students. During the machine learning milestone, the original modeling approach was improved because the first version used only one machine learning model and mainly reported final accuracy. The final version now includes multiple models, stronger evaluation metrics, cross-validation, classification reports, confusion matrices, and an external dataset for comparative analysis.

## Motivation

Academic performance is affected by many factors such as study habits, attendance, previous academic success, family background, and learning behavior. Understanding these factors is useful because it can help educators identify patterns related to student success and possible academic risk. This project focuses on using data science methods to examine these relationships and to evaluate whether student performance can be predicted from available student information.

The project also has a practical purpose: applying the main stages of a data science workflow, including data collection, cleaning, exploratory data analysis, hypothesis testing, feature preparation, machine learning modeling, and result interpretation.

## Data Sources

### Main Dataset

The main dataset used in the project is:

```text
Higher Education Students Performance Evaluation Dataset
```

File used in the repository:

```text
data/student_performance_value.csv
```

This dataset contains **145 student records** and education-related variables such as weekly study hours, class attendance, note-taking behavior, last semester GPA, expected graduation GPA, course information, demographic information, and the final output grade.

The original target variable is:

```text
output_grade
```

with the following grade categories:

```text
Fail, DD, DC, CC, CB, BB, BA, AA
```

### External Dataset for Enrichment

Because the project guidelines expect public datasets to be enriched with an additional data source, an external educational dataset was added for a separate comparative analysis:

```text
UCI Student Performance Dataset
```

File used in the repository:

```text
data/external_student_performance_uci.csv
```

This external dataset contains **1044 student records** after combining the Mathematics and Portuguese course datasets. It includes demographic, social, family, school-related, and academic variables. The original final grade variable is:

```text
G3
```

The external dataset was not directly merged with the main dataset because the two datasets have different feature structures and were collected in different educational contexts. Instead, it was used as a **separate comparative dataset** to test whether the machine learning workflow performs better on a larger educational dataset.

## Why the Dataset Was Diversified

The original dataset is useful for exploratory analysis, but it is small for machine learning. It contains only **145 rows**, while the original target variable has **8 different grade classes**. This makes the prediction task difficult because some classes have very few examples. For example, exact grade prediction with labels such as `Fail`, `DD`, `DC`, `CB`, and `AA` creates a sparse classification problem.

To make the machine learning task more stable and meaningful, the project made two improvements:

1. The original 8-grade target was transformed into a broader 3-class performance target.
2. A larger external student performance dataset was added for comparative analysis.

The external dataset was added for enrichment and comparison, not as a replacement for the original dataset. The main dataset remains the central project dataset, while the external dataset supports a broader machine learning evaluation.

## Target Variable Preparation

### Main Dataset Target Mapping

The original grade labels were grouped into three broader academic performance levels:

```text
Fail, DD, DC  -> Low
CC, CB        -> Medium
BB, BA, AA    -> High
```

The new target variable is:

```text
performance_level
```

This transformation keeps the educational meaning of the output while making the classification problem more realistic for a 145-row dataset.

### External Dataset Target Mapping

For the external UCI dataset, the final numeric grade `G3` was converted into the same three-class target:

```text
G3 < 10          -> Low
10 <= G3 < 15    -> Medium
G3 >= 15         -> High
```

This allowed the main and external datasets to be evaluated using the same type of academic performance label.

## Project Pipeline

The project follows the main stages of a data science workflow:

1. Data collection and organization
2. Data cleaning and preprocessing
3. Exploratory Data Analysis (EDA)
4. Data visualization
5. Hypothesis testing
6. Feature preparation
7. Machine learning model training
8. Model comparison and evaluation
9. Interpretation of results
10. Final report preparation

## Exploratory Data Analysis

The EDA stage examines the structure and distribution of the main dataset. The analysis includes:

- Dataset shape and column inspection
- Data type inspection
- Missing value checks
- Category distribution analysis
- Output grade distribution
- Weekly study hours distribution
- Attendance distribution
- Last semester GPA distribution
- Visual comparisons between selected variables and academic output

The main EDA script is:

```text
eda.py
```

The notebook version is:

```text
EDA_and_Hypothesis.ipynb
```

Important EDA figures are stored under:

```text
figures - EDA/
```

## Hypothesis Testing

The hypothesis testing stage investigates whether selected variables are statistically related to academic performance.

The tested relationships include:

- Attendance to classes vs output grade
- Weekly study hours vs output grade
- Last semester GPA vs output grade
- Differences between study-hour groups

The main hypothesis testing script is:

```text
hypothesis.py
```

The milestone report for EDA and hypothesis testing is:

```text
eda-hypothesis-report.pdf
```

The strongest statistical result from this stage was that previous academic performance, represented by last semester GPA, had a meaningful relationship with the current output grade. Attendance and weekly study hours did not show strong statistically significant relationships in the original small dataset.

## Machine Learning Methods

The machine learning milestone was redesigned to include a proper comparative analysis. Instead of using only one model, the final version evaluates multiple machine learning methods and reports several evaluation metrics.

The main ML script is:

```text
Milestone2_ML.py
```

The notebook version is:

```text
Milestone2_ML.ipynb
```

The following machine learning models were used:

1. Dummy Baseline
2. Logistic Regression
3. Linear Support Vector Machine
4. K-Nearest Neighbors
5. Decision Tree
6. Random Forest
7. Gradient Boosting

The Dummy Baseline is included to show whether real models perform better than a simple majority-class prediction strategy.

## ML Experiment Settings

Three machine learning experiments were performed:

### 1. Original Higher Education Dataset - 3-Class Performance

This experiment uses the main 145-row dataset with the transformed `Low`, `Medium`, and `High` target variable.

### 2. External UCI Dataset - With Prior Grades

This experiment uses the external dataset and includes `G1` and `G2` as input features. These variables represent first-period and second-period grades, so they are strong predictors of the final grade `G3`.

### 3. External UCI Dataset - Early Prediction Without Prior Grades

This experiment removes `G1` and `G2` from the input features. This creates a harder and more realistic early-prediction setting, because the model must predict final performance without using previous course grade information.

## Train-Test Split and Evaluation

For each experiment, the data was split as follows:

```text
80% training data
20% test data
```

The split was stratified by the target variable to preserve the class distribution across train and test sets.

The training set was also evaluated using:

```text
3-fold Stratified Cross-Validation
```

The following metrics were reported:

- Cross-validation accuracy
- Cross-validation macro F1
- Test accuracy
- Balanced accuracy
- Macro precision
- Macro recall
- Macro F1
- Weighted F1
- Classification report
- Confusion matrix

These metrics provide a more complete evaluation than accuracy alone, especially because the class distributions are not perfectly balanced.

## Main Machine Learning Results

| Experiment | Best Model | Test Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|
| Original Higher Education Dataset - 3-Class Performance | Gradient Boosting | 0.724 | 0.682 | 0.729 |
| External UCI Dataset - With Prior Grades | Random Forest | 0.885 | 0.876 | 0.885 |
| External UCI Dataset - Without Prior Grades | Linear SVM | 0.579 | 0.533 | 0.582 |

## Interpretation of Results

The original exact 8-class grade prediction task was too difficult for the small 145-row dataset. After converting the target into three broader performance levels, the original dataset produced a much more stable result. The best model on the original dataset was Gradient Boosting, with approximately **0.724 test accuracy** and **0.682 macro F1**.

The strongest result came from the external UCI dataset when prior grades `G1` and `G2` were included. Random Forest achieved approximately **0.885 test accuracy** and **0.876 macro F1**. This result is strong, but it should be interpreted carefully because prior grades are naturally close to the final grade target.

When `G1` and `G2` were removed, the external dataset became a more realistic early-prediction problem. In this setting, Linear SVM achieved approximately **0.579 test accuracy** and **0.533 macro F1**. The lower performance shows that predicting academic performance early, before grade information is available, is a harder task.

Overall, the results show that academic performance prediction becomes more reliable when the target variable is not too fragmented and when informative academic history features are available.

## Key Changes Made in the Final Version

Several changes were made to improve the project after the earlier milestone version:

- The machine learning section was expanded from a single-model approach to a comparative ML analysis.
- Seven models were evaluated, including a baseline model.
- The original 8-class grade target was transformed into a 3-class performance target.
- An external UCI Student Performance dataset was added as a separate enrichment and comparative analysis.
- Evaluation was expanded beyond accuracy to include macro F1, weighted F1, balanced accuracy, precision, recall, classification reports, and confusion matrices.
- Cross-validation was added for more reliable model comparison.
- The ML code was provided both as a Python script and as a Jupyter notebook.
- Result CSV files and model comparison figures were generated automatically.
- The final report was prepared in PDF and LaTeX formats.

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── DSA210_Project_Proposal.pdf
├── Milestone1.md
├── eda.py
├── hypothesis.py
├── EDA_and_Hypothesis.ipynb
├── eda-hypothesis-report.pdf
├── Milestone2_ML.py
├── Milestone2_ML.ipynb
├── Final_Report.pdf
├── Final_Report.tex
├── data/
│   ├── student_performance_value.csv
│   └── external_student_performance_uci.csv
├── figures - EDA/
│   ├── output_grade_distribution.png
│   ├── study_hours_vs_grade.png
│   ├── attendance_vs_grade.png
│   └── last_semester_gpa_vs_grade.png
├── figures_ml/
│   ├── original_3class_model_comparison_accuracy.png
│   ├── original_3class_model_comparison_macro_f1.png
│   ├── original_3class_confusion_matrix_best_model.png
│   ├── external_uci_with_prior_grades_model_comparison_accuracy.png
│   ├── external_uci_with_prior_grades_model_comparison_macro_f1.png
│   ├── external_uci_with_prior_grades_confusion_matrix_best_model.png
│   ├── external_uci_without_prior_grades_model_comparison_accuracy.png
│   ├── external_uci_without_prior_grades_model_comparison_macro_f1.png
│   └── external_uci_without_prior_grades_confusion_matrix_best_model.png
└── results_ml/
    ├── all_model_comparisons.csv
    ├── original_3class_model_comparison.csv
    ├── external_uci_with_prior_grades_model_comparison.csv
    ├── external_uci_without_prior_grades_model_comparison.csv
    ├── original_3class_classification_report_best_model.csv
    ├── external_uci_with_prior_grades_classification_report_best_model.csv
    ├── external_uci_without_prior_grades_classification_report_best_model.csv
    ├── original_3class_confusion_matrix_best_model.csv
    ├── external_uci_with_prior_grades_confusion_matrix_best_model.csv
    └── external_uci_without_prior_grades_confusion_matrix_best_model.csv
```

## How to Reproduce the Analysis

### 1. Clone the Repository

```bash
git clone https://github.com/ecemrana/DSA210-DataScience-Project.git
cd DSA210-DataScience-Project
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run EDA

```bash
python eda.py
```

### 4. Run Hypothesis Tests

```bash
python hypothesis.py
```

### 5. Run Machine Learning Analysis

```bash
python Milestone2_ML.py
```

The ML script will generate result tables under:

```text
results_ml/
```

and visual outputs under:

```text
figures_ml/
```

The notebook version can also be opened and run:

```text
Milestone2_ML.ipynb
```

## Final Report

The final report is included in two formats:

```text
Final_Report.pdf
Final_Report.tex
```

The PDF version is the main report for submission. The `.tex` file is included for transparency and future editing.

## Limitations

The original dataset is small and contains only 145 observations. This limits the reliability of exact grade prediction, especially when using the original 8-class grade target. The external dataset improves the size of the comparative analysis, but it comes from a different educational context and should not be treated as identical to the original dataset.

Another limitation is that the strongest external result depends on previous grade variables `G1` and `G2`. These features are very informative, but they make the task easier because they are closely related to the final grade `G3`. The early-prediction version without `G1` and `G2` is more realistic but naturally produces lower performance.

## Future Work

Possible future improvements include:

- Collecting more student records using the same survey structure as the original dataset
- Testing additional models such as XGBoost or LightGBM
- Applying hyperparameter tuning with grid search or randomized search
- Using SHAP or permutation importance for stronger model interpretability
- Building an early-warning system for students at academic risk
- Comparing results across different universities, departments, and course types
- Improving feature engineering for behavioral and academic variables

## Author

Rana Ecem Kılıç

## Course

DSA 210 - Introduction to Data Science  
2025-2026 Spring Term
