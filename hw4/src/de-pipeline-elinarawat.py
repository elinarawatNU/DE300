from airflow import DAG
from airflow.operators.python_operator import PythonOperator, BranchPythonOperator
#from airflow.providers.sftp.operators.sftp import SFTPOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.operators.dummy_operator import DummyOperator
#from airflow.contrib.hooks.ssh_hook import SSHHook
from airflow.utils.dates import days_ago
from airflow.hooks.base_hook import BaseHook
import boto3
import pandas as pd
from io import StringIO
import tomli
import pathlib
from sqlalchemy import create_engine

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

plt.rcParams.update(**{'figure.dpi':150})
plt.style.use('ggplot') # can skip this - plots are more visually appealing with this style

from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, FloatType, LongType, StringType, DoubleType
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import col, when , isnan, count, udf, pow, log1p
from pyspark.ml import Pipeline, Transformer
from pyspark.ml.feature import StringIndexer, VectorAssembler, Imputer
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression, LinearSVC, DecisionTreeClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator, Evaluator
import pyspark.sql.functions as F
from pyspark.ml.clustering import KMeans
from itertools import combinations
import os

import requests
from bs4 import BeautifulSoup

import scrapy
from scrapy import Selector
from typing import List
import re

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix


# read the parameters from toml
CONFIG_FILE = "s3://de300spring2024-airflow//config.toml"  # Example S3 path


TABLE_NAMES = {
    "original_data": "heart_disease",
    "clean_data": "heart_disease_clean_data",
    "train_data": "heart_disease_train_data",
    "test_data": "heart_disease_test_data",
    "normalization_data": "normalization_values",
    "max_fe": "max_fe_features",
    "product_fe": "product_fe_features"
}

ENCODED_SUFFIX = "_encoded"

# Define the default args dictionary for DAG
default_args = {
    'owner': 'elinarawat',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'retries': 1,
}


download_path = "/s3_data/heart_disease.csv"
data_path = "../data/heart_disease.csv"
def read_config_from_s3() -> dict:
    # Parse the bucket name and file key from the S3 path
    bucket_name = "de300spring2024"
    key = "/elina_rawat/heart_disease.csv"

    # Create a boto3 S3 client
    s3_client = boto3.client('s3')
    s3_client.download_file(bucket_name, key, download_path)
    return download_path

hw1_clean_data = "../data/hw1_clean_impute.csv"
def hw1_clean_impute(**kwargs):
    # Cleaning step 1
    columns = ['age', 'sex', 'painloc', 'painexer', 'cp', 'trestbps', 'smoke', 'fbs', 'prop', 'nitr', 'pro', 'diuretic', 'thaldur', 'thalach', 'exang', 'oldpeak', 'slope', 'target']
    df = pd.read_csv(data_path)
    df = df[columns]

    # Cleaning step 2
    def trestbpsClean(x):
        if x < 100:
            x = np.nan
        return x
    def oldpeakClean(x):
        if x < 0 or x > 4:
            x = np.nan
        return x
    def greaterthanoneClean(x):
        if x > 1:
            x = np.nan
        return x
    df['trestbps'] = df['trestbps'].apply(trestbpsClean)
    df['oldpeak'] = df['oldpeak'].apply(oldpeakClean)
    for col in ['fbs', 'prop', 'nitr', 'pro', 'diuretic']:
        df[col] = df[col].apply(greaterthanoneClean)

    for col in ['painloc', 'painexer', 'fbs', 'prop', 'nitr', 'pro', 'diuretic', 'exang', 'slope']:
        mode_object = df[col].mode()
        df[col].fillna(mode_object[0], inplace=True)

    for col in ['trestbps', 'thaldur', 'thalach', 'oldpeak']:
        df[col].fillna(df[col].median(), inplace=True)
        
    df["age"] = pd.to_numeric(df["age"], errors='coerce')

    # Cleaning step 3
    df['abs_smoke'] = np.NaN
    df['cdc_smoke'] = np.NaN

    df.to_csv(hw1_clean_data, index=False)

hw3_clean_data = "../data/hw3_clean_impute.csv"
def hw3_clean_impute(**kwargs):
    spark = SparkSession.builder \
        .appName("Heart Disease") \
        .getOrCreate()
    df = spark.read \
        .format("csv") \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .load(data_path)
    print(df)

    columns = ['age', 'sex', 'painloc', 'painexer', 'cp', 'trestbps', 'smoke', 'fbs', 'prop', 'nitr', 'pro', 'diuretic', 'thaldur', 'thalach', 'exang', 'oldpeak', 'slope', 'target']
    df = df.select(columns)
    df = df.dropna(subset=["target"])
    df = df.withColumn("trestbps", when(F.col("trestbps") < 100, 100).otherwise(F.col("trestbps")))
    df = df.withColumn("oldpeak", when(F.col("oldpeak") < 0, 0).when(F.col("oldpeak") > 4, 4).otherwise(F.col("oldpeak")))
    for col_name in ["painloc", "painexer", 'fbs', 'prop', 'nitr', 'pro', 'diuretic']:
        df = df.withColumn(col_name, when(F.col(col_name) < 0, 0).when(F.col(col_name) > 1, 1).otherwise(F.col(col_name)))
    df = df.withColumn("age", df["age"].cast(IntegerType()))
    print(df)

    df.write \
        .format("csv") \
        .option("header", "true") \
        .mode("overwrite") \
        .save(hw3_clean_data)
    
hw1_fe_data = "../data/hw1_fe_data.csv"
def hw1_fe(**kwargs):
    df = pd.read_csv(hw1_clean_data)
    binary_cols = []
    for col in df.columns:
        if df[col].isin([0, 1, np.nan]).all():
            binary_cols.append(col)

    # Categorical data
    category_cols = ["cp", "slope"]

    # Numberical data
    number_cols = []
    for col in df.columns:
        if col not in binary_cols and col not in category_cols:
            number_cols.append(col)


    kmeans = KMeans(n_clusters=5, random_state=0)
    kmeans_cluster = kmeans.fit(df[number_cols])
    df['cluster'] = kmeans_cluster.labels_

    to_calc = df[number_cols].values - kmeans_cluster.cluster_centers_[df['cluster']]
    df['distance_to_center'] = np.linalg.norm(to_calc, axis=1)

    outlier_threshold = df['distance_to_center'].quantile(0.95)
    df = df[df['distance_to_center'] < outlier_threshold]

    df = df.drop(['cluster', 'distance_to_center'], axis=1)

    def calculate_skewness(series):
        skewness = (((series - series.mean()) / series.std())**3).mean()
        return skewness

    transformation_df = df.copy(deep=True)

    for col in number_cols:
        skewness = calculate_skewness(df[col])
        # Positive skewness (atypical) - log transformation
        if skewness > 1:
            transformation_df[col] = np.log1p(df[col])
        # Negative skewness (atypical) - square transformation
        elif skewness < -1:
            transformation_df[col] = np.square(df[col])

    # Since the categorical and binary data have already been transformed into numbers, no further transformations are necessary or make sense

    transformation_df.to_csv(hw1_fe_data, index=False)
    
hw3_fe_data = "../data/hw3_fe_data.csv"
def hw3_fe(**kwargs):
    spark = SparkSession.builder \
        .appName("Heart Disease") \
        .getOrCreate()
    df = spark.read \
        .format("csv") \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .load(hw3_clean_data)
    
    # to handle skewed data
    df = df.withColumn("log_age", log1p(F.col("age")))
    # to standardize data
    df = df.withColumn("scaled_chol", F.col("chol")/10)
    # cholesterol categories
    df = df.withColumn("chol_category", when(F.col("chol") < 200, "low").when((F.col("chol") >= 200) & (F.col("chol") < 240), "medium").otherwise("high"))

    df.write \
        .format("csv") \
        .option("header", "true") \
        .mode("overwrite") \
        .save(hw3_fe_data)



hw_scrape_smoke = "../data/hw_scrape_smoke.csv"
def scrape_data(**kwargs):
    # Referenced Robert's lab python file
    def parse_row(row:Selector) -> List[str]:
        '''
        Parses a html row into a list of individual elements
        '''
        cells = row.xpath('.//th | .//td')
        row_data = []
        
        for cell in cells:
            cell_text = cell.xpath('normalize-space(.)').get()
            cell_text = re.sub(r'<.*?>', ' ', cell_text)  # Remove remaining HTML tags
            # if there are br tags, there will be some binary characters
            cell_text = cell_text.replace('\xa0', '')  # Remove \xa0 characters
            row_data.append(cell_text)
        
        
        return row_data

    # Referenced Robert's lab python file
    def parse_table_as_df(table_sel:Selector,header:bool=True) -> pd.DataFrame:
        '''
        Parses a html table and returns it as a Pandas DataFrame
        '''
        # extract rows
        rows = table_sel.xpath('./tbody//tr')
        
        # parse header and the remaining rows
        columns = None
        start_row = 0
        if header:
            columns = parse_row(rows[0])
            start_row += 1
            
        table_data = [parse_row(row) for row in rows[start_row:]]
        
        # return data frame
        return pd.DataFrame(table_data,columns=columns)


    def age_to_smoke_percent(abs_data, age):
        if pd.isna(age):
            return np.nan
        for i, row in abs_data.iterrows():
            age_group = row["Age Range"]

            if len(age_group) == 2:
                if int(age) >= int(age_group[0]) and int(age) <= int(age_group[1]):
                    return row['2022 (%)']
            elif len(age_group) == 1:
                if int(age) >= int(age_group[0]):
                    return row['2022 (%)']
        return np.nan

    def cdc_age_sex_smoke(cdc_age_data, age, cdc_sex_data, sex):
        if pd.isna(age) or pd.isna(sex):
            return np.nan
        age_rate = np.nan
        for index, row in cdc_age_data.iterrows():
            age_range = row['Age Range']
            if len(age_range) == 2:
                if int(age) >= int(age_range[0]) and int(age) <= int(age_range[1]):
                    age_rate = float(row['2022 (%)'])
            elif len(age_range) == 1:
                if int(age) >= int(age_range[0]):
                    age_rate = float(row['2022 (%)'])

        sex_rate = np.nan
        if sex == 0.0:
            sex_rate = float(cdc_sex_data[cdc_sex_data['Sex'] == 'Female']['2022 (%)'].values[0])
        else:
            female_rate = float(cdc_sex_data[cdc_sex_data['Sex'] == 'Female']['2022 (%)'].values[0])
            male_rate = float(cdc_sex_data[cdc_sex_data['Sex'] == 'Male']['2022 (%)'].values[0])
            sex_rate = age_rate * (male_rate / female_rate)

        return sex_rate

    def smoke_transform(x):
        if pd.isna(x):
            x = np.nan
        else:
            x = float(x)
            if x >= 12:
                x = 1
            else:
                x = 0
        return x

    def impute_smoke_transform(row):
        if pd.isna(row["smoke"]):
            if row["abs_transformed"] == 1 or row["cdc_transformed"] == 1:
                return 1
            else:
                return 0
        return row["smoke"]

    # Source 2
    url = "https://www.cdc.gov/tobacco/data_statistics/fact_sheets/adult_data/cig_smoking/index.htm"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # CDC DATA
    html_sections = str(soup).split('<h4 class="card-header h4 bg-white rounded-0">')
    html_sections = html_sections[1:]
    cdc_data = {}
    for section in html_sections:
        sup_index = section.index("<sup>")
        to_append = section[3:sup_index]
        soup_section = BeautifulSoup(section, 'html.parser')
        data_key = soup_section.find_all('li')
        data_list = []
        for item in data_key:
            if item is not None:
                data_text = item.get_text(strip=True)
                data_list.append(data_text)

        if to_append == "Sex" or to_append == "Age":
            cdc_data[to_append] = data_list

    df_sex_list = [['Male'], ['Female']]

    male_number = cdc_data["Sex"][0][cdc_data["Sex"][0].index("(") + 1:cdc_data["Sex"][0].index("%")]
    female_number = cdc_data["Sex"][1][cdc_data["Sex"][1].index("(") + 1:cdc_data["Sex"][1].index("%")]

    df_sex_list[0].append(male_number)
    df_sex_list[1].append(female_number)

    df_age_list = []
    for age in cdc_data["Age"]:
        age_range = age.index("aged")
        years_range = age.index("years")
        final_ages = age[age_range + 5:years_range - 1]
        
        bracket_index = age.index("(")
        percentage_index = age.index("%")
        final_percent_age = age[bracket_index + 1:percentage_index]
        df_age_list.append([final_ages, final_percent_age])

    cdc_sex_data = pd.DataFrame(df_sex_list)
    cdc_sex_data = cdc_sex_data.set_axis(['Sex', '2022 (%)'], axis=1)

    for row in df_age_list:
        if "–" in row[0]:
            dash_index = row[0].index("–")
            row[0] = [row[0][:dash_index], row[0][dash_index+1:]]
        else:
            row[0] = [row[0]]

    cdc_age_data = pd.DataFrame(df_age_list)
    cdc_age_data = cdc_age_data.set_axis(['Age Range', '2022 (%)'], axis=1)

    # Source 1
    url = "https://www.abs.gov.au/statistics/health/health-conditions-and-risks/smoking-and-vaping/latest-release"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    url_selector_table = Selector(text = response.content).xpath("//table")[1]

    html_body = url_selector_table.xpath(".//tbody//tr")
    html_header = parse_row(url_selector_table.xpath(".//thead//tr")[0])

    row_list = []
    for row in html_body:
        row_list.append(parse_row(row))

    html_header[0] = "Age Range"

    for row in row_list:
        if "–" in row[0]:
            dash_index = row[0].index("–")
            row[0] = [row[0][:dash_index], row[0][dash_index+1:]]
        else:
            years_index = row[0].index("years")
            row[0] = [row[0][:years_index-1]]

    abs_data_df = pd.DataFrame(row_list, columns = html_header)
    abs_data = abs_data_df[["Age Range", "2022 (%)"]]


    df = pd.read_csv(hw1_clean_data)

    # Create new columns, and impute missing values into Smoke

    df['abs_smoke'] = df['age'].apply(lambda x: age_to_smoke_percent(abs_data, x))
    df['cdc_smoke'] = df.apply(lambda x: cdc_age_sex_smoke(cdc_age_data, x['age'], cdc_sex_data, x['sex']), axis=1)

    df['abs_smoke'] = pd.to_numeric(df['abs_smoke'])

    # Transformation: If >= 12%, then classify as smoker, otherwise classify as non-smoker
    df["abs_transformed"] = df["abs_smoke"].apply(lambda x: smoke_transform(x))
    df["cdc_transformed"] = df["cdc_smoke"].apply(lambda x: smoke_transform(x))

    df["smoke"] = df.apply(impute_smoke_transform, axis=1)

    # More cleaning
    for col in df.columns:
        for row in df[col]:
            if not pd.isna(row):
                row = float(row)

    # FROM HW1
    # Binary data (1 or 0)
    binary_cols = []
    for col in df.columns:
        if df[col].isin([0, 1, np.nan]).all():
            binary_cols.append(col)

    # Categorical data
    category_cols = ["cp", "slope"]

    # Numberical data
    number_cols = []
    for col in df.columns:
        if col not in binary_cols and col not in category_cols:
            number_cols.append(col)
            
    # Delete negative values
    for col in number_cols:
        if (df[col] < 0).any():
                df = df[df[col] >= 0]

    # Binary and categorical cleaning: impute using the mode
    for col in binary_cols:
        mode_object = df[col].mode()
        df[col].fillna(mode_object[0], inplace=True)
        
    for col in category_cols:
        mode_object = df[col].mode()
        df[col].fillna(mode_object[0], inplace=True)
        
    # Numerical cleaning: impute using mean
    for col in number_cols:
        df[col].fillna(df[col].mean(), inplace=True)
    
    df.to_csv(hw_scrape_smoke, index=False)
    
hw_merge = "../data/hw_merge.csv"
def merge_hws(**kwargs):
    df_hw1 = pd.read_csv(hw1_fe_data)
    spark = SparkSession.builder \
        .appName("Heart Disease") \
        .getOrCreate()
    df_hw3 = spark.read \
        .format("csv") \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .load(hw3_fe_data)
    
    def drop_duplicate_columns(df1, df2):
        cols_to_drop = []
        for col in df2.columns:
            if col in df1.columns and col != "id":
                cols_to_drop.append(col)
        return df2.drop(*cols_to_drop)
    
    df_hw3 = drop_duplicate_columns(df_hw1, df_hw3)
    df_scraped = pd.read_csv(hw_scrape_smoke)
    df_scraped = drop_duplicate_columns(df_hw1, df_scraped)

    merge_df1 = df_hw1.join(df_hw3, on="id", how="inner")
    merge_df = merge_df1.join(df_scraped, on="id", how="inner")
    merge_df.to_csv(hw_merge, index=False)


def hw1_train(**kwargs):
    def evaluate_train_model(X, y, model):
        roc_auc = cross_val_score(model, X, y, scoring = "roc_auc", cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=100))
        f1 = cross_val_score(model, X, y, scoring = "f1", cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=100))
        accuracy = cross_val_score(model, X, y, scoring = "accuracy", cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=100))
        to_return_dict = {"accuracy": np.mean(accuracy), "f1": np.mean(f1), "roc_auc": np.mean(roc_auc)}
        return to_return_dict
    
    df = pd.read_csv(hw1_fe_data)
    columns = ['age', 'sex', 'painloc', 'painexer', 'cp', 'trestbps', 'smoke', 'fbs', 'prop', 'nitr', 'pro', 'diuretic', 'thaldur', 'thalach', 'exang', 'oldpeak', 'slope', 'target']
    df = df[columns]

    y = df['target']
    X = df.drop(columns="target")

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 100, test_size = 0.1, stratify = y)

    random_model = RandomForestClassifier(random_state = 100)
    logistic_model = LogisticRegression(random_state = 100, max_iter = 1000)
    svc_model = SVC(random_state = 100, probability = True)

    model_dict = {"Random Forest": random_model, "Logistic Regression": logistic_model, "SVC": svc_model}

    pipeline = Pipeline([('scaler', StandardScaler()), ('imputer', SimpleImputer(strategy='mean'))])

    X_train = pipeline.fit_transform(X_train)
    X_test = pipeline.transform(X_test)

    model_results = {}

    for model in model_dict:
        model_dict[model].fit(X_train, y_train)
        model_results[model] = evaluate_train_model(X_train, y_train, model_dict[model])

    print("### HW1 RESULTS ###")

    print("Best Model: ", max(model_results, key = lambda x: model_results[x]['roc_auc']))
    best_model = model_dict[max(model_results, key = lambda x: model_results[x]['roc_auc'])]
    best_model.fit(X_train, y_train)
    y_prediction = best_model.predict(X_test)
    y_prediction_probability = best_model.predict_proba(X_test)
    y_prediction_probability = y_prediction_probability[:, 1]

    roc_test = roc_auc_score(y_test, y_prediction_probability)
    print("ROC AUC Score Test: ", roc_test)
    f1_test = f1_score(y_test, y_prediction)
    print("F1 Score Test: ", f1_test)
    accu_test = accuracy_score(y_test, y_prediction)
    print("Accuracy Score Test: ", accu_test)
    confusion_matrix_result = confusion_matrix(y_test, y_prediction)
    print("Confusion Matrix: ", confusion_matrix_result)

    kwargs['ti'].xcom_push(key='hw1_roc_auc', value=roc_test)
    kwargs['ti'].xcom_push(key='hw1_f1', value=f1_test)
    kwargs['ti'].xcom_push(key='hw1_accuracy', value=accu_test)
    kwargs['ti'].xcom_push(key='hw1_best_model', value=max(model_results, key = lambda x: model_results[x]['roc_auc']))

def merge_train(**kwargs):
    def evaluate_train_model(X, y, model):
        roc_auc = cross_val_score(model, X, y, scoring = "roc_auc", cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=100))
        f1 = cross_val_score(model, X, y, scoring = "f1", cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=100))
        accuracy = cross_val_score(model, X, y, scoring = "accuracy", cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=100))
        to_return_dict = {"accuracy": np.mean(accuracy), "f1": np.mean(f1), "roc_auc": np.mean(roc_auc)}
        return to_return_dict
    
    df = pd.read_csv(hw_merge)
    columns = ['age', 'sex', 'painloc', 'painexer', 'cp', 'trestbps', 'smoke', 'fbs', 'prop', 'nitr', 'pro', 'diuretic', 'thaldur', 'thalach', 'exang', 'oldpeak', 'slope', 'target']
    df = df[columns]

    y = df['target']
    X = df.drop(columns="target")

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 100, test_size = 0.1, stratify = y)

    random_model = RandomForestClassifier(random_state = 100)
    logistic_model = LogisticRegression(random_state = 100, max_iter = 1000)
    svc_model = SVC(random_state = 100, probability = True)

    model_dict = {"Random Forest": random_model, "Logistic Regression": logistic_model, "SVC": svc_model}

    pipeline = Pipeline([('scaler', StandardScaler()), ('imputer', SimpleImputer(strategy='mean'))])

    X_train = pipeline.fit_transform(X_train)
    X_test = pipeline.transform(X_test)

    model_results = {}

    for model in model_dict:
        model_dict[model].fit(X_train, y_train)
        model_results[model] = evaluate_train_model(X_train, y_train, model_dict[model])

    print("### MERGED RESULTS ###")

    print("Best Model: ", max(model_results, key = lambda x: model_results[x]['roc_auc']))
    best_model = model_dict[max(model_results, key = lambda x: model_results[x]['roc_auc'])]
    best_model.fit(X_train, y_train)
    y_prediction = best_model.predict(X_test)
    y_prediction_probability = best_model.predict_proba(X_test)
    y_prediction_probability = y_prediction_probability[:, 1]

    roc_test = roc_auc_score(y_test, y_prediction_probability)
    print("ROC AUC Score Test: ", roc_test)
    f1_test = f1_score(y_test, y_prediction)
    print("F1 Score Test: ", f1_test)
    accu_test = accuracy_score(y_test, y_prediction)
    print("Accuracy Score Test: ", accu_test)
    confusion_matrix_result = confusion_matrix(y_test, y_prediction)
    print("Confusion Matrix: ", confusion_matrix_result)

    kwargs['ti'].xcom_push(key='merge_roc_auc', value=roc_test)
    kwargs['ti'].xcom_push(key='merge_f1', value=f1_test)
    kwargs['ti'].xcom_push(key='merge_accuracy', value=accu_test)
    kwargs['ti'].xcom_push(key='merge_best_model', value=max(model_results, key = lambda x: model_results[x]['roc_auc']))


def hw3_train(**kwargs):
    spark = SparkSession.builder \
        .appName("Heart Disease") \
        .getOrCreate()
    df = spark.read \
        .format("csv") \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .load(hw3_fe_data)
    
    numeric_features = [f.name for f in df.schema.fields if isinstance(f.dataType, (DoubleType, FloatType, IntegerType, LongType))]
   
    numeric_features.remove("target")

    imputed_columns_numeric = [f"Imputed{v}" for v in numeric_features]
    imputer_numeric = Imputer(inputCols=numeric_features, outputCols=imputed_columns_numeric, strategy="mean")

    assembler = VectorAssembler(
        inputCols=imputed_columns_numeric, 
        outputCol="features"
    )

    rf = RandomForestClassifier(labelCol='target', featuresCol='features')
    lr = LogisticRegression(labelCol='target', featuresCol='features', maxIter=1000)
    svc = LinearSVC(labelCol='target', featuresCol='features')
    dt = DecisionTreeClassifier(labelCol='target', featuresCol='features')
    class_dict = {"RandFor": rf, "LogReg": lr, "SVC": svc, "DecTree": dt}

    PG_rf = ParamGridBuilder().addGrid(rf.numTrees, [20, 50]).build()
    PG_lr = ParamGridBuilder().addGrid(lr.regParam, [0.01, 0.1]).build()
    PG_svc = ParamGridBuilder().addGrid(svc.regParam, [0.01, 0.1]).build()
    PG_dt = ParamGridBuilder().addGrid(dt.maxDepth, [4, 6, 8, 10, 12]).addGrid(dt.minInstancesPerNode, [1, 2, 4, 6]).build()

    PG_dict = {"RandFor": PG_rf, "LogReg": PG_lr, "SVC": PG_svc, "DecTree": PG_dt}

    evaluator_roc_auc = BinaryClassificationEvaluator(labelCol='target', rawPredictionCol='rawPrediction', metricName='areaUnderROC')
    evaluator_f1 = MulticlassClassificationEvaluator(labelCol='target', predictionCol='prediction', metricName='f1')
    evaluator_accuracy = MulticlassClassificationEvaluator(labelCol='target', predictionCol='prediction', metricName='accuracy')

    train, test = df.randomSplit([0.9, 0.1], seed=42)

    best_roc_auc = float("-inf")
    best_f1 = float("-inf")
    best_accuracy = float("-inf")

    full_scores_dict = {}
    scores_dict = {}

    for ML_model, model_class in class_dict.items():
        stages = [imputer_numeric, assembler, model_class]
        pipeline = Pipeline(stages=stages)


        crossval = CrossValidator(estimator=pipeline, estimatorParamMaps=PG_dict[ML_model], evaluator = BinaryClassificationEvaluator(labelCol='target'), numFolds=5, seed=42)
        cvModel = crossval.fit(train)
        predictions = cvModel.transform(test)

        roc_auc = evaluator_roc_auc.evaluate(predictions)
        f1 = evaluator_f1.evaluate(predictions)
        accuracy = evaluator_accuracy.evaluate(predictions)

        print(f"{ML_model} ROC AUC: {roc_auc}, F1: {f1}, Accuracy: {accuracy}")

        score = roc_auc + f1 + accuracy

        full_scores_dict[ML_model] = [roc_auc, f1, accuracy]
        scores_dict[ML_model] = score

    print("### HW3 RESULTS ###")

    best_model = max(scores_dict, key=lambda x: scores_dict[x])

    for model in full_scores_dict:
        if full_scores_dict[model][0] > best_roc_auc:
            best_roc_auc = full_scores_dict[model][0]
        if full_scores_dict[model][1] > best_f1:
            best_f1 = full_scores_dict[model][1]
        if full_scores_dict[model][2] > best_accuracy:
            best_accuracy = full_scores_dict[model][2]

    print(f"Best Model: {best_model}, Best ROC_AUC: {best_roc_auc}, Best F1: {best_f1}, Best Accuracy: {best_accuracy}")
    
    final_dict = {"RandFor": "RandomForest", "LogReg": "LogisticRegression", "SVC": "SVC", "DecTree": "DecisionTree"}
    print(f"Final Selected Model: {final_dict[best_model]}")

    kwargs['ti'].xcom_push(key='hw3_roc_auc', value=full_scores_dict[best_model][0])
    kwargs['ti'].xcom_push(key='hw3_f1', value=full_scores_dict[best_model][1])
    kwargs['ti'].xcom_push(key='hw3_accuracy', value=full_scores_dict[best_model][2])
    kwargs['ti'].xcom_push(key='hw3_best_model', value=best_model)


def final_train_result(**kwargs):
    task_ids = ['hw1_train', 'hw3_train', "merge_train"]
    roc_auc_keys = ["hw1_roc_auc", "hw3_roc_auc", "merge_roc_auc"]
    f1_keys = ["hw1_f1", "hw3_f1", "merge_f1"]
    accu_keys = ["hw1_accuracy", "hw3_accuracy", "merge_accuracy"]
    best_model_keys = ["hw1_best_model", "hw3_best_model", "merge_best_model"]
    
    roc_auc_dict = {"HW1": kwargs['ti'].xcom_pull(task_ids=task_ids[0])[roc_auc_keys[0]], "HW3": kwargs['ti'].xcom_pull(task_ids=task_ids[1])[roc_auc_keys[1]], "MERGE": kwargs['ti'].xcom_pull(task_ids=task_ids[2])[roc_auc_keys[2]]}
    f1_dict = {"HW1": kwargs['ti'].xcom_pull(task_ids=task_ids[0])[f1_keys[0]], "HW3": kwargs['ti'].xcom_pull(task_ids=task_ids[1])[f1_keys[1]], "MERGE": kwargs['ti'].xcom_pull(task_ids=task_ids[2])[f1_keys[2]]}
    accu_dict = {"HW1": kwargs['ti'].xcom_pull(task_ids=task_ids[0])[accu_keys[0]], "HW3": kwargs['ti'].xcom_pull(task_ids=task_ids[1])[accu_keys[1]], "MERGE": kwargs['ti'].xcom_pull(task_ids=task_ids[2])[accu_keys[2]]}
    best_model_dict = {"HW1": kwargs['ti'].xcom_pull(task_ids=task_ids[0])[best_model_keys[0]], "HW3": kwargs['ti'].xcom_pull(task_ids=task_ids[1])[best_model_keys[1]], "MERGE": kwargs['ti'].xcom_pull(task_ids=task_ids[2])[best_model_keys[2]]}

    current_score = float("-inf")
    current_best = None
    for i in ["HW1", "HW3", "MERGE"]:
        total_score = roc_auc_dict[i] + f1_dict[i] + accu_dict[i]
        if total_score >= current_score:
            current_score = total_score
            current_best = i
    
    best_model = best_model_dict[current_best]
    print(f"Best Model: {best_model}")

dag = DAG('elinarawat_hw4', default_args=default_args, description="HW4 Heart Disease", schedule_interval="@daily")

read_config_from_s3 = PythonOperator(
    task_id='read_config_from_s3',
    python_callable=read_config_from_s3,
    provide_context=True,
    dag=dag
)

hw1_clean_impute = PythonOperator(
    task_id='hw1_clean_impute',
    python_callable=hw1_clean_impute,
    provide_context=True,
    dag=dag
)

hw3_clean_impute = PythonOperator(
    task_id='hw3_clean_impute',
    python_callable=hw3_clean_impute,
    provide_context=True,
    dag=dag
)

hw1_fe = PythonOperator(
    task_id='hw1_fe',
    python_callable=hw1_fe,
    provide_context=True,
    dag=dag
)

hw3_fe = PythonOperator(
    task_id='hw3_fe',
    python_callable=hw3_fe,
    provide_context=True,
    dag=dag
)

scrape_data = PythonOperator(
    task_id='scrape_data',
    python_callable=scrape_data,
    provide_context=True,
    dag=dag
)

merge_hws = PythonOperator(
    task_id='merge_hws',
    python_callable=merge_hws,
    provide_context=True,
    dag=dag
)

hw1_train = PythonOperator(
    task_id='hw1_train',
    python_callable=hw1_train,
    provide_context=True,
    dag=dag
)

merge_train = PythonOperator(
    task_id='merge_train',
    python_callable=merge_train,
    provide_context=True,
    dag=dag
)

hw3_train = PythonOperator(
    task_id='hw3_train',
    python_callable=hw3_train,
    provide_context=True,
    dag=dag
)

final_train_result = PythonOperator(
    task_id='final_train_result',
    python_callable=final_train_result,
    provide_context=True,
    dag=dag
)

hw1_clean_impute >> hw1_fe
hw3_clean_impute >> [hw3_fe, scrape_data]
[hw1_fe, hw3_fe, scrape_data] >> merge_hws
hw1_fe >> hw1_train
hw3_fe >> hw3_train
merge_hws >> merge_train
[hw1_train, hw3_train, merge_train] >> final_train_result
