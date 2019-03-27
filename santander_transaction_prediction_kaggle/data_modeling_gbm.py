"""
Data Modeling - Gradient Boost Machine

Algorithms:
    - Bayesian Optimization
    - Gradient Boost Model + Stratified K-Fold Cross Validation
"""

from bayes_opt import BayesianOptimization
import lightgbm as lgb
import numpy as np
import pandas as pd
from time import time
import warnings
import yaml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# Configuration
config = yaml.load(open("data_modeling_gbm.yml", encoding="utf-8"))
warnings.filterwarnings("ignore")


def light_gradient_boost(**kwargs):
    """ Light Gradient Boost Machine Model. This function parameters are optimized by BayesianOptimization """
    for key_, value_ in kwargs.items():
        if key_ in config["int_params_to_optimize"]:
            config["lgb"]["param"][key_] = int(value_)
        elif key_ in config["float_params_to_optimize"]:
            config["lgb"]["param"][key_] = float(value_)

    # Load data into LightGBM datasets
    train_data_ = lgb.Dataset(train_df.iloc[train_indexes_bay][features], label=y_train.iloc[train_indexes_bay])
    validation_data_ = lgb.Dataset(train_df.iloc[val_indexes_bay][features], label=y_train.iloc[val_indexes_bay])

    # Model training
    gbm_ = lgb.train(params=config["lgb"]["param"],
                     train_set=train_data_,
                     num_boost_round=config["lgb"]["num_boost_round"],
                     valid_sets=[validation_data_],
                     verbose_eval=-1,
                     early_stopping_rounds=config["lgb"]["early_stopping_rounds"])

    predictions_ = gbm_.predict(train_df.iloc[val_indexes_bay][features], num_iteration=gbm_.best_iteration)

    return roc_auc_score(y_train.iloc[val_indexes_bay], predictions_)


# Load datasets
train_df = pd.read_csv(**config["data"]["raw"]["train"])
test_df = pd.read_csv(**config["data"]["raw"]["test"])

if config["subset_testing"]:
    print("Testing script with subsets")
    train_df = train_df.iloc[0:500]
    test_df = test_df.iloc[0:500]

features = [c for c in train_df.columns if c not in [config["id"], config["target"]]]
y_train = train_df[config["target"]]

# -- Bayesian Optimization --
train_indexes_bay, val_indexes_bay = \
    list(StratifiedKFold(n_splits=2, shuffle=True, random_state=1).split(train_df.values, y_train.values))[0]

# Bounded region of parameter space
bounds_lgb = config["lgb"]["params_to_optimize"]

optimizer = BayesianOptimization(f=light_gradient_boost, pbounds=bounds_lgb, verbose=2, random_state=1)

print("-"*50)
with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    optimizer.maximize(**config["optimizer"])
print("-" * 50)
optimizer.probe(params=config["lgb"]["probe"], lazy=True)
optimizer.maximize(init_points=0, n_iter=0)
print(optimizer.max)
print("-"*50)

# Save optimized parameters
for key, value in optimizer.max["params"].items():
    if key in config["int_params_to_optimize"]:
        config["lgb"]["param"][key] = int(value)
    if key in config["float_params_to_optimize"]:
        config["lgb"]["param"][key] = float(value)

# -- LGBM --

# The folds are made by preserving the percentage of samples for each class.
folds = StratifiedKFold(**config["stratified_k_fold"])

out_of_fold = np.zeros(len(train_df))
predictions = np.zeros(len(test_df))

feature_importance_df = pd.DataFrame()

for fold_i, (train_indexes, val_indexes) in enumerate(folds.split(train_df.values, y_train.values)):
    print("Fold {}".format(fold_i))

    # Load data into LightGBM datasets
    train_data = lgb.Dataset(train_df.iloc[train_indexes][features], label=y_train.iloc[train_indexes])
    validation_data = lgb.Dataset(train_df.iloc[val_indexes][features], label=y_train.iloc[val_indexes])

    # Model training
    gbm = lgb.train(config["lgb"]["param"],
                    train_data,
                    config["lgb"]["num_boost_round"],
                    valid_sets=[train_data, validation_data],
                    verbose_eval=config["lgb"]["verbose_eval"],
                    early_stopping_rounds=config["lgb"]["early_stopping_rounds"])

    # Predictions on out of fold subset. Limit the number of iterations until the best performance in training.
    out_of_fold[val_indexes] = gbm.predict(train_df.iloc[val_indexes][features], num_iteration=gbm.best_iteration)

    # Predictions on test set.
    predictions += gbm.predict(test_df[features], num_iteration=gbm.best_iteration) / folds.n_splits

# Model evaluation (AUC score)
print("CV score: {:<8.5f}".format(roc_auc_score(y_train, out_of_fold)))

# Submission DataFrame
sub_df = pd.DataFrame({config["id"]: test_df[config["id"]].values})
sub_df[config["target"]] = predictions

timestamp_ = int(time())

# Export submission data
config["data"]["submissions"]["path_or_buf"] = config["data"]["submissions"]["path_or_buf"].format(timestamp_)
sub_df.to_csv(**config["data"]["submissions"])
