from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / 'data' / 'student_performance_value.csv'
FIGURE_DIR = BASE_DIR / 'figures'


GRADE_ORDER = ['Fail', 'DD', 'DC', 'CC', 'CB', 'BB', 'BA', 'AA']
STUDY_ORDER = ['none', '<5 hours', '6-10 hours', '11-20 hours', '+20 hours']
GPA_ORDER = ['<2.00', '2.00-2.49', '2.50-2.99', '3.00-3.49', 'above 3.49']


def ensure_output_dir() -> None:
    """Create the figure directory if it does not exist."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)



def load_data() -> pd.DataFrame:
    """Load the main dataset."""
    return pd.read_csv(DATA_PATH)



def print_basic_overview(df: pd.DataFrame) -> None:
    """Print a quick summary of the dataset."""
    print('\n=== DATASET OVERVIEW ===')
    print(f'Rows, columns: {df.shape}')
    print('\nColumn names:')
    for col in df.columns:
        print(f'- {col}')

    print('\n=== DATA TYPES ===')
    print(df.dtypes)

    print('\n=== MISSING VALUES ===')
    print(df.isna().sum())

    print('\n=== FIRST FIVE ROWS ===')
    print(df.head())



def print_categorical_summaries(df: pd.DataFrame) -> None:
    """Print frequency tables for a few important variables."""
    important_columns = [
        'output_grade',
        'weekly_study_hours',
        'attendance_to_classes',
        'cumulative_grade_point_average_in_the_last_semester',
        'sex',
        'scholarship_type',
    ]

    print('\n=== CATEGORY COUNTS ===')
    for column in important_columns:
        print(f'\n{column}')
        print(df[column].value_counts(dropna=False))



def plot_output_grade_distribution(df: pd.DataFrame) -> None:
    """Plot the final grade distribution."""
    grade_counts = (
        df['output_grade']
        .value_counts()
        .reindex(GRADE_ORDER)
        .fillna(0)
    )

    plt.figure(figsize=(8, 5))
    grade_counts.plot(kind='bar')
    plt.title('Output Grade Distribution')
    plt.xlabel('Output Grade')
    plt.ylabel('Number of Students')
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / 'output_grade_distribution.png', dpi=300)
    plt.close()



def plot_study_hours_vs_grade(df: pd.DataFrame) -> None:
    """Plot study hours against output grade."""
    table = pd.crosstab(df['weekly_study_hours'], df['output_grade'])
    table = table.reindex(index=STUDY_ORDER, columns=GRADE_ORDER, fill_value=0)

    plt.figure(figsize=(10, 6))
    table.plot(kind='bar', stacked=True)
    plt.title('Weekly Study Hours vs Output Grade')
    plt.xlabel('Weekly Study Hours')
    plt.ylabel('Number of Students')
    plt.legend(title='Output Grade', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / 'study_hours_vs_grade.png', dpi=300)
    plt.close()



def plot_attendance_vs_grade(df: pd.DataFrame) -> None:
    """Plot class attendance against output grade."""
    table = pd.crosstab(df['attendance_to_classes'], df['output_grade'])
    table = table.reindex(columns=GRADE_ORDER, fill_value=0)

    plt.figure(figsize=(8, 5))
    table.plot(kind='bar', stacked=True)
    plt.title('Attendance to Classes vs Output Grade')
    plt.xlabel('Attendance to Classes')
    plt.ylabel('Number of Students')
    plt.legend(title='Output Grade', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / 'attendance_vs_grade.png', dpi=300)
    plt.close()



def plot_last_semester_gpa_vs_grade(df: pd.DataFrame) -> None:
    """Plot previous GPA group against output grade."""
    table = pd.crosstab(
        df['cumulative_grade_point_average_in_the_last_semester'],
        df['output_grade']
    )
    table = table.reindex(index=GPA_ORDER, columns=GRADE_ORDER, fill_value=0)

    plt.figure(figsize=(10, 6))
    table.plot(kind='bar', stacked=True)
    plt.title('Last Semester GPA vs Output Grade')
    plt.xlabel('Last Semester GPA')
    plt.ylabel('Number of Students')
    plt.legend(title='Output Grade', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / 'last_semester_gpa_vs_grade.png', dpi=300)
    plt.close()



def main() -> None:
    """Run the full EDA workflow and save plots."""
    ensure_output_dir()
    df = load_data()

    print_basic_overview(df)
    print_categorical_summaries(df)

    plot_output_grade_distribution(df)
    plot_study_hours_vs_grade(df)
    plot_attendance_vs_grade(df)
    plot_last_semester_gpa_vs_grade(df)

    print('\nEDA is complete. Figures were saved into the figures/ folder.')


if __name__ == '__main__':
    main()
