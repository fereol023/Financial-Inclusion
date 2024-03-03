import time
import os
import pandas as pd
from enum import Enum
from scipy.stats import shapiro, pearsonr
from itertools import product
from pprint import pprint


def line(fonction):
    def wrapper(*args, **kwargs):
        l_ = '=' * 200
        print(l_)
        start = time.time()
        res = fonction(*args, **kwargs)
        print(f"Durée : {round(time.time() - start, 5)} secs")
        return res
    return wrapper


class MyPaths(Enum):
    estimatorFittedStandardEncoder: str = './model/artifacts/fitted_standardScaler.joblib'
    estimatorFittedLabelEncoder: str = './model/artifacts/fitted_labelEncoder.joblib'


class Explorer:

    def __init__(self, csvfilepath: str, target: str):
        self.target = target
        self.path = csvfilepath
        self.shapiro_df = None
        if os.path.exists(csvfilepath):
            self.df = pd.read_csv(csvfilepath)
        else:
            raise Exception(f'{csvfilepath} doesnt exist.')

    @line
    def global_description(self):
        print(self.df.dtypes)
        print(self.df.describe())

    @line
    def global_check_null(self):
        for col in self.df.columns:
            print(f"Null counts {col} : {self.df[col].isnull().sum()}")

    @line
    def distro_top_category(self):
        # nbre de modalités
        for col in self.df.select_dtypes(include='object').columns:
            print(f"{col} has {self.df[col].nunique()} values")
            print(100 * self.df[col].value_counts() / self.df[col].count())
        # graphes

    @line
    def normality(self):
        # tests de normalité
        shapiro_res = {}
        for col in self.df.select_dtypes(exclude='object').columns:
            shapiro_res.update({col: shapiro(self.df[col])[1]})
            self.shapiro_df = pd.DataFrame.from_dict(shapiro_res, orient='index', columns=['pvalue'])
        print(self.shapiro_df)
        # graphes

    @line
    def correlation(self):
        # tests de correlation
        corr_res = []
        for col1, col2 in list(product(self.df.columns, self.df.columns)):
            try:
                corr, xx = pearsonr(self.df[col1], self.df[col2])
                corr_res.append((col1, col2, corr, xx))
            except Exception as e:
                pass
        corr_res = [x for x in corr_res if x[0] == self.target and x[3] <= .05]
        pprint(corr_res)
