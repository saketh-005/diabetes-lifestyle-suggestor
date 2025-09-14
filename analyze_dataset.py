import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('dataset/synthetic_amd_federated_dataset.csv')

# Show basic info
def basic_info(df):
    print('Shape:', df.shape)
    print('Columns:', df.columns.tolist())
    print('Missing values per column:')
    print(df.isnull().sum())
    print('\nSample rows:')
    print(df.head())

# Target distribution
def target_distribution(df, target='Diabetes'):
    print(f'\n{target} value counts:')
    print(df[target].value_counts())
    sns.countplot(x=target, data=df)
    plt.title(f'{target} Distribution')
    plt.show()

# Correlation heatmap
def correlation_heatmap(df):
    plt.figure(figsize=(12,8))
    numeric_df = df.select_dtypes(include=['number'])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm')
    plt.title('Feature Correlation Heatmap (Numeric Features Only)')
    plt.show()

# Main analysis
if __name__ == '__main__':
    basic_info(df)
    target_distribution(df)
    correlation_heatmap(df)
