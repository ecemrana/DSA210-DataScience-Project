from pathlib import Path
import pandas as pd
from scipy.stats import chi2_contingency, spearmanr, kruskal


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / 'data' / 'student_performance_value.csv'

GRADE_MAP = {
    'Fail': 0,
    'DD': 1,
    'DC': 2,
    'CC': 3,
    'CB': 4,
    'BB': 5,
    'BA': 6,
    'AA': 7,
}

STUDY_MAP = {
    'none': 0,
    '<5 hours': 1,
    '6-10 hours': 2,
    '11-20 hours': 3,
    '+20 hours': 4,
}

GPA_MAP = {
    '<2.00': 0,
    '2.00-2.49': 1,
    '2.50-2.99': 2,
    '3.00-3.49': 3,
    'above 3.49': 4,
}



def load_data() -> pd.DataFrame:
    """Load the main dataset."""
    df = pd.read_csv(DATA_PATH)
    df['grade_score'] = df['output_grade'].map(GRADE_MAP)
    df['study_score'] = df['weekly_study_hours'].map(STUDY_MAP)
    df['last_semester_gpa_score'] = df[
        'cumulative_grade_point_average_in_the_last_semester'
    ].map(GPA_MAP)
    return df



def run_chi_square_test(df: pd.DataFrame, feature: str, label: str) -> None:
    """Run a chi-square independence test for a categorical feature."""
    table = pd.crosstab(df[feature], df['output_grade'])
    chi2, p_value, dof, expected = chi2_contingency(table)

    print(f'\n=== {label} ===')
    print('Contingency table:')
    print(table)
    print(f'Chi-square statistic: {chi2:.4f}')
    print(f'Degrees of freedom: {dof}')
    print(f'P-value: {p_value:.6f}')

    if p_value < 0.05:
        print('Result: Reject H0. There is a statistically significant relationship.')
    else:
        print('Result: Fail to reject H0. There is not enough evidence for a significant relationship.')



def run_spearman_test(df: pd.DataFrame) -> None:
    """Run Spearman correlation between last semester GPA and output grade."""
    rho, p_value = spearmanr(df['last_semester_gpa_score'], df['grade_score'])

    print('\n=== H3: Last Semester GPA and Output Grade ===')
    print(f'Spearman correlation: {rho:.4f}')
    print(f'P-value: {p_value:.6f}')

    if p_value < 0.05:
        print('Result: Reject H0. The relationship is statistically significant.')
    else:
        print('Result: Fail to reject H0. The relationship is not statistically significant.')



def run_kruskal_test(df: pd.DataFrame) -> None:
    """Run Kruskal-Wallis test for grade scores across study-hour groups."""
    groups = []
    ordered_groups = ['none', '<5 hours', '6-10 hours', '11-20 hours', '+20 hours']
    for group_name in ordered_groups:
        group_scores = df.loc[df['weekly_study_hours'] == group_name, 'grade_score']
        if len(group_scores) > 0:
            groups.append(group_scores)

    statistic, p_value = kruskal(*groups)

    print('\n=== H4: Grade Score Differences Across Study Hour Groups ===')
    print(f'Kruskal-Wallis statistic: {statistic:.4f}')
    print(f'P-value: {p_value:.6f}')

    if p_value < 0.05:
        print('Result: Reject H0. At least one study-hour group is different.')
    else:
        print('Result: Fail to reject H0. There is not enough evidence for a group difference.')



def main() -> None:
    """Run all hypothesis tests for milestone 1."""
    df = load_data()

    print('MILESTONE 1 HYPOTHESIS TESTS')
    print('Significance level: 0.05')

    print('\nH1')
    print('H0: attendance_to_classes and output_grade are independent.')
    print('H1: attendance_to_classes and output_grade are related.')
    run_chi_square_test(df, 'attendance_to_classes', 'H1: Attendance to Classes vs Output Grade')

    print('\nH2')
    print('H0: weekly_study_hours and output_grade are independent.')
    print('H1: weekly_study_hours and output_grade are related.')
    run_chi_square_test(df, 'weekly_study_hours', 'H2: Weekly Study Hours vs Output Grade')

    print('\nH3')
    print('H0: last semester GPA and output grade are not associated.')
    print('H1: last semester GPA and output grade are associated.')
    run_spearman_test(df)

    print('\nH4')
    print('H0: grade_score distributions are the same across study-hour groups.')
    print('H1: at least one study-hour group has a different grade_score distribution.')
    run_kruskal_test(df)


if __name__ == '__main__':
    main()
