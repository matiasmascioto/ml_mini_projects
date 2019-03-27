"""
Data exploration and cleaning
"""


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# -- Load data --

# First look in a subset
train_chunks_df = pd.read_csv("data/raw/train.csv", iterator=True)

train_chunk_df = train_chunks_df.get_chunk(5000)
print(train_chunk_df.head())
print("-"*10)
print(train_chunk_df.info())
print("-"*10)
print(train_chunk_df.describe())
print("-"*10)

# Load full datasets
train_df = pd.read_csv("data/raw/train.csv")
test_df = pd.read_csv("data/raw/test.csv")

print(train_df.info())
print("-"*10)
print(test_df.info())
print("-"*10)
print(train_df.describe())
print("-"*10)
print(test_df.describe())


# -- Missing Values --
print("Train set has missing values: {}".format(train_df.isnull().values.any()))
print("Test set has missing values: {}".format(test_df.isnull().values.any()))


# -- Target distribution --
sns_plot = sns.countplot(train_df["target"])
sns_plot.figure.savefig("plots/target_distribution.png")

print("There are {}% target values with 1".format(100 * train_df["target"].value_counts()[1]/train_df.shape[0]))

# -- Feature distribution --

def plot_feature_distribution(df1, df2, label1, label2, features, file_path):
    i = 0
    sns.set_style("whitegrid")
    plt.figure()
    fig, ax = plt.subplots(10,10,figsize=(18,22))

    for feature in features:
        i += 1
        plt.subplot(10,10,i)
        sns.kdeplot(df1[feature], bw=0.5,label=label1)
        sns.kdeplot(df2[feature], bw=0.5,label=label2)
        plt.xlabel(feature, fontsize=9)
        locs, labels = plt.xticks()
        plt.tick_params(axis="x", which="major", labelsize=6, pad=-6)
        plt.tick_params(axis="y", which="major", labelsize=6)
    # plt.show()
    fig.savefig(file_path)


t0 = train_df.loc[train_df["target"] == 0]
t1 = train_df.loc[train_df["target"] == 1]
features = train_df.columns.values[2:102]
plot_feature_distribution(t0, t1, "0", "1", features, "plots/feature_distribution_1.png")
features = train_df.columns.values[102:202]
plot_feature_distribution(t0, t1, "0", "1", features, "plots/feature_distribution_2.png")