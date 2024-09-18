import pandas as pd


file_path = '/home/prashant/Documents/multimodal/scripts/yooo.csv'  # Replace with the path to your CSV file
data = pd.read_csv(file_path)


def calculate_statistics(group):
    stats = {}

    for col in group.columns:
        if col not in ['Unnamed: 0', 'filename', 'Frame']:
            stats[f'{col}_Mean'] = group[col].mean()
            stats[f'{col}_Median'] = group[col].median()
            stats[f'{col}_StdDev'] = group[col].std()
            stats[f'{col}_Range'] = group[col].max() - group[col].min()
            stats[f'{col}_IQR'] = group[col].quantile(0.75) - group[col].quantile(0.25)
            stats[f'{col}_IDR'] = group[col].quantile(0.90) - group[col].quantile(0.10)
            stats[f'{col}_MAD'] = (group[col] - group[col].mean()).abs().mean()
    return pd.Series(stats)

result = data.groupby('filename').apply(calculate_statistics)

output_path = 'filename_based_statistics.csv'
result.to_csv(output_path)

print(f"Statistics saved to {output_path}")

