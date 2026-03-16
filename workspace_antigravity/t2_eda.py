import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("Loading dataset...")
    df = pd.read_csv('adult_income.csv')

    # 1. Handle non-standard missing values ('?') by dropping them
    # Strip whitespace just in case there are spaces before/after '?'
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df.replace('?', np.nan, inplace=True)
    initial_shape = df.shape
    df.dropna(inplace=True)
    print(f"Dropped {initial_shape[0] - df.shape[0]} rows containing missing values '?'.")
    print(f"Data shape after dropping missing values: {df.shape}")

    # Separate features by type
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if 'income' in categorical_cols:
        categorical_cols.remove('income')

    print("Generating distribution plots for continuous features...")
    # 2. Generate distribution plots for all continuous features
    for col in numerical_cols:
        plt.figure(figsize=(8, 5))
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f'Distribution Plot for {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(f'dist_{col}.png')
        plt.close()

    print("Generating frequency plots for categorical features...")
    # Generate frequency plots for all categorical features
    for col in categorical_cols:
        plt.figure(figsize=(10, 6))
        sns.countplot(y=col, data=df, order=df[col].value_counts().index)
        plt.title(f'Frequency Plot for {col}')
        plt.xlabel('Count')
        plt.ylabel(col)
        plt.tight_layout()
        plt.savefig(f'freq_{col}.png')
        plt.close()

    # Target variable frequency
    plt.figure(figsize=(6, 4))
    sns.countplot(x='income', data=df)
    plt.title('Class Balance of Target Variable: income')
    plt.xlabel('Income')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('freq_income.png')
    plt.close()

    print("Generating correlation heatmap...")
    # 3. Produce a correlation heatmap for numerical features
    plt.figure(figsize=(10, 8))
    corr = df[numerical_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title('Correlation Heatmap of Numerical Features')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    plt.close()

    print("Generating grouped categorical comparisons...")
    # 4. Grouped categorical comparisons reporting proportions
    for col in categorical_cols:
        crosstab = pd.crosstab(df[col], df['income'], normalize='index')
        crosstab.plot(kind='bar', stacked=True, figsize=(10, 6))
        plt.title(f'Proportion of Income Classes by {col}')
        plt.xlabel(col)
        plt.ylabel('Proportion')
        plt.legend(title='Income')
        plt.tight_layout()
        plt.savefig(f'grouped_prop_{col}_vs_income.png')
        plt.close()

    # 5. Narrative Summary
    income_counts = df['income'].value_counts(normalize=True)
    majority_class = income_counts.index[0]
    majority_prop = income_counts.iloc[0] * 100
    minority_class = income_counts.index[1]
    minority_prop = income_counts.iloc[1] * 100

    print("\n" + "="*60)
    print("EXPLORATORY DATA ANALYSIS SUMMARY")
    print("="*60)

    print("\n1. Key Patterns Relevant to Modeling:")
    print("   a. Zero-Inflated Skewed Features: 'capital-gain' and 'capital-loss' have \n"
          "      the vast majority of their values at exactly 0. These will need careful \n"
          "      treatment (e.g., binning, non-linear models, or transformations) as \n"
          "      they are heavily right-skewed.")
    print("   b. Multicollinearity: 'education' and 'education-num' contain the same \n"
          "      information in categorical vs. numerical format. One should be dropped \n"
          "      prior to modeling to avoid perfect collinearity.")
    print("   c. Predictive Categoricals: Certain categories in features like 'marital-status' \n"
          "      (e.g., Married-civ-spouse) and 'occupation' show strong correlations \n"
          "      with the >50K income bracket relative to other categories.")

    print("\n2. Class Balance of the Target Variable:")
    print(f"   The 'income' target variable exhibits significant class imbalance. \n"
          f"   The majority class '{majority_class}' accounts for {majority_prop:.1f}% of the \n"
          f"   dataset (after dropping rows with missing values), while the minority \n"
          f"   class '{minority_class}' makes up only {minority_prop:.1f}%.")

    print("\n3. Expected Impact on Model Training/Evaluation:")
    print("   Because of the class imbalance (~75% vs ~25%), raw accuracy will be a \n"
          "   misleading evaluation metric (a naive model predicting <=50K always \n"
          "   achieves 75% accuracy). We must evaluate model performance using metrics \n"
          "   like F1-Score, Precision, Recall, or PR-AUC. Furthermore, we may need \n"
          "   to apply class weights or resampling techniques (like SMOTE) during \n"
          "   training to prevent the model from becoming biased towards the majority class.")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
