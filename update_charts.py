import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv('results/benchmark_results_v2.csv')
RESULTS_DIR = 'results'
configs = ['N=3_raw', 'N=3_lowercase', 'N=1_cleaned', 'N=2_cleaned', 'N=3_cleaned', 'N=4_cleaned']

# Plotting: Score Distribution Boxplots for Positive Matches
plt.figure(figsize=(10, 6))
pos_df = df[df['label'] == 1]
sns.boxplot(data=pos_df[configs])
plt.title("Distribution of True Plagiarism Match Scores")
plt.ylabel("Similarity Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'score_distribution.png'))
plt.close()

# Plotting: Mean Score by Obfuscation Level
plt.figure(figsize=(10, 6))
if not pos_df.empty:
    obf_means = pos_df.groupby('obfuscation')[configs].mean()
    obf_means.plot(kind='bar', figsize=(10, 6))
    plt.title("Average Overlap Score by Obfuscation Level")
    plt.ylabel("Average Similarity Score")
    plt.xticks(rotation=0)
    plt.legend(title="Config")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'obfuscation_impact.png'))
    plt.close()

print("Graphs updated.")
