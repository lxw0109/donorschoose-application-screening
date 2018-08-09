#!/usr/bin/env python3
# coding: utf-8
# File: 1_kernal_1.py
# Author: lxw
# Date: 3/19/18 10:51 PM
"""
Reference:
[(How to get 81%) GRU-ATT + LGBM + TF-IDF + EDA](https://www.kaggle.com/hoonkeng/how-to-get-81-gru-att-lgbm-tf-idf-eda/notebook)
"""

def main():
    # 1.1 Importing the libraries
    import pandas as pd
    import matplotlib.pyplot as plt
    import plotly.offline as py

    # 1.2 Data Preparation
    df = pd.read_csv("../input/train.csv")
    """
    print(df.head(3))
    print(df.sort_values("project_submitted_datetime").head(3))
    """

    # 1.2.1 Migrate the features from resource.csv to train.csv
    resources_df = pd.read_csv("../input/resources.csv")
    # print(resources_df.head(3))
    resources_df["amount"] = resources_df["quantity"] * resources_df["price"]
    amount_df = resources_df.groupby("id")["amount"].agg("sum").sort_values(ascending=False).reset_index()

    resource_amount_map = {}
    for i, row in amount_df.iterrows():
        resource_amount_map[row["id"]] = row["amount"]

    df.insert(4, "total_amount", df["id"].map(resource_amount_map))
    print(df.tail(3))



if __name__ == "__main__":
    main()
