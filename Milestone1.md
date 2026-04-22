# Milestone 1

## Project Title
Factors Related to University Students' Academic Performance

## Dataset
This project uses the **Higher Education Students Performance Evaluation** dataset.
The file used in this milestone is:

- `data/student_performance_value.csv`

The dataset has **145 rows** and **32 columns**.
The target variable is:

- `output_grade`

## Files in This Milestone
- `eda.py`: basic exploratory data analysis and figure generation
- `hypothesis.py`: hypothesis testing for selected variables
- `report.pdf`: short written milestone report in PDF format
- `requirements.txt`: Python dependencies

## How to Run
Run these commands from the project root folder:

```bash
python eda.py
python hypothesis.py
```

## What `eda.py` Does
- loads the dataset
- prints dataset shape, columns, data types, and missing values
- prints value counts for important variables
- saves EDA figures into the `figures/` folder

Generated figures:
- `figures/output_grade_distribution.png`
- `figures/study_hours_vs_grade.png`
- `figures/attendance_vs_grade.png`
- `figures/last_semester_gpa_vs_grade.png`

## What `hypothesis.py` Does
The script tests several milestone hypotheses:

1. class attendance and output grade
2. weekly study hours and output grade
3. last semester GPA and output grade
4. grade-score differences across study-hour groups

Used methods:
- chi-square test of independence
- Spearman correlation
- Kruskal-Wallis test

## Initial Summary
From the current milestone results, the clearest statistically significant relationship is between **last semester GPA** and **output grade**. The attendance and study-hour hypotheses do not reach the 0.05 significance level in the current tests.
