#!/usr/bin/env python3

from catboost.datasets import titanic

train, test = titanic()

train.to_csv("/home/artem/Projects/mlops-course/data/raw/train.csv", columns=train.columns)

test.to_csv("/home/artem/Projects/mlops-course/data/raw/test.csv", columns=test.columns)

