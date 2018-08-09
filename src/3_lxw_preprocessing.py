#!/usr/bin/env python3
# coding: utf-8
# File: 3_lxw_preprocessing.py
# Author: lxw
# Date: 3/27/18 3:50 PM

"""
# Spacy -> LTP
# textblob -> 情感分析(snownlp)

Reference:
1. [(How to get 81%) GRU-ATT + LGBM + TF-IDF + EDA](https://www.kaggle.com/hoonkeng/how-to-get-81-gru-att-lgbm-tf-idf-eda/notebook)
"""

import pandas as pd
from matplotlib import pyplot as plt


def subject_categories_subcategories(df):
    """
    Process field "project_subject_categories" & "project_subject_subcategories"
    """
    cat_df = df[["project_subject_categories", "project_subject_subcategories", "project_is_approved"]]
    new_cat, new_sub, new_approve = [], [], []

    for i, row in cat_df.iterrows():
        cats = row["project_subject_categories"].str.split(",")
        for cat in cats:
            # cat: <list of str>
            for item in cat:
                new_cat.append(item.strip())

        subs = row["project_subject_subcategories"].str.split(",")
        for sub in subs:
            # sub: <list of str>
            for item in sub:
                new_sub.append(item.strip())

        new_approve.append(row["project_is_approved"])    # type(row["project_is_approved"]): int

    new_cat = pd.DataFrame({"project_subject_categories": new_cat})
    print(new_cat, "\n", "--" * 20)
    new_sub = pd.DataFrame({"project_subject_subcategories": new_sub})
    print(new_sub, "\n", "--" * 20)
    new_approve = pd.DataFrame({"project_is_approved": new_approve})
    print(new_approve, "\n", "--" * 20)

    cat_total_df = pd.concat([new_cat, new_sub, new_approve], axis=1).reset_index()
    cat_approval_df = cat_total_df.groupby(["project_subject_categories", "project_subject_subcategories"])[
        "project_is_approved"].agg("sum")
    cat_all_df = cat_total_df.groupby(["project_subject_categories", "project_subject_subcategories"]).count()[
        "project_is_approved"]
    cat_heat = (cat_approval_df / cat_all_df).unstack()

    plt.figure(figsize=(18, 5))    # You can Arrange The Size As Per Requirement
    # ax = sns.heatmap(cat_heat, cmap="viridis_r")
    plt.title("Aprroved rate for categories and sub-categories combination")
    plt.show()


def main():
    train_df = pd.read_csv("../input/train.csv", sep=",")
    test_df = pd.read_csv("../input/test.csv", sep=",")
    data_list = [train_df, test_df]
    data_df = pd.concat(data_list, axis=1)
    # print(data_df.describe())

    subject_categories_subcategories(data_df)


if __name__ == "__main__":
    main()
