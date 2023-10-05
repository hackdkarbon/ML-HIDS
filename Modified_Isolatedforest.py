#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 1 07:25:16 2023
@author: rajeshsingh
"""
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import make_scorer, f1_score, classification_report
from sklearn.feature_extraction.text import CountVectorizer
import logging
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer
import ipaddress
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
#import mysql.connector
#import tensorflow as tf
#from tensorflow.keras.models import Model
#from tensorflow.keras.layers import Input, Dense

# logging.basicConfig(level=logging.INFO)
# Define the format and filename for logging
logging.basicConfig(filename='D:\Temporary\SNORT_ML_Project\Python\Final_codes\IsoForest_data_processor.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
error_logger = logging.getLogger('error_logger')
error_logger.addHandler(logging.FileHandler('D:\Temporary\SNORT_ML_Project\Python\Final_codes\Isoforesterror.log'))

class DataProcessor:
    def __init__(self, df):
        if df is None or df.empty:
            raise ValueError("DataFrame is empty or None")
        self.df = df
        self.identify_and_convert_columns()
        self.handle_text_columns()
        self.handle_missing_values()

    @staticmethod
    def is_valid_ip(ip_str):
        """Validates IP addresses using ipaddress module and regex."""
        try:
            ipaddress.ip_address(ip_str)
            return True
        except ValueError:
            # If conversion with ipaddress fails, falls back to regex
            return bool(re.match(r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$', str(ip_str)))
        
    def identify_and_convert_columns(self):
        try:
            self.categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
            self.continuous_cols = self.df.select_dtypes(include=['float64', 'int64']).columns.tolist()

            for col in self.categorical_cols:
                # Improved robust datetime check:
                is_datetime = False
                try:
                    is_datetime = all(pd.to_datetime(self.df[col], errors='coerce').notna())
                except Exception as e:
                    logging.warning(f"Unable to check datetime for column {col}: {str(e)}")
                
                if is_datetime:
                    self.df[col] = pd.to_datetime(self.df[col]).astype(int) / 10 ** 9
                    if col not in self.continuous_cols:
                        self.continuous_cols.append(col)
                    self.categorical_cols.remove(col)
                    
            logging.info("Columns identified and converted where necessary.")
        except Exception as e:
            logging.error(f"Error in identifying and converting columns: {str(e)}")
            raise

    def handle_missing_values(self):
        try:
            # Handle missing categorical values
            cat_imputer = SimpleImputer(strategy='most_frequent')
            self.df[self.categorical_cols] = cat_imputer.fit_transform(self.df[self.categorical_cols])
 
            # Handle missing continuous values
            cont_imputer = SimpleImputer(strategy='mean')
            self.df[self.continuous_cols] = cont_imputer.fit_transform(self.df[self.continuous_cols])
        except Exception as e:
            logging.error(f"Error in handling missing values: {str(e)}")
            raise
 
    def handle_text_columns(self):
        try:
            self.text_cols = []
            for col in self.df.columns:
                if self.df[col].dtype == 'object' and self.df[col].str.contains('\s').any():
                    self.text_cols.append(col)
        except Exception as e:
            logging.error(f"Error in identifying text columns: {str(e)}")
            raise
 
    def preprocess_data(self):
        try:
            # Extract text data
            def text_data(df):
                return df[self.text_cols] if self.text_cols else pd.DataFrame()
            
            # Extract non-text data
            def non_text_data(df):
                return df.drop(columns=self.text_cols) if self.text_cols else df
            
            non_text_transformer = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), self.continuous_cols),
                    ('cat', OneHotEncoder(), self.categorical_cols)
                ],
                remainder='passthrough'
            )

            preprocessor = FeatureUnion(
                transformer_list=[
                    ('text', Pipeline([
                        ('selector', FunctionTransformer(text_data, validate=False)),
                        ('vectorizer', CountVectorizer())
                    ])),
                    ('non_text', Pipeline([
                        ('selector', FunctionTransformer(non_text_data, validate=False)),
                        ('transformer', non_text_transformer)
                    ]))
                ]
            )

            return Pipeline([
                ('preprocessor', preprocessor),
                ('model', IsolationForest(contamination=0.01, random_state=42))
            ])
        except Exception as e:
            logging.error(f"Error in preprocessing data: {str(e)}")
            raise
            
    def perform_eda(self):
        """
        Perform exploratory data analysis on the dataset.
        """
        try:
            # Correlation Matrix Heatmap
            correlation_matrix = self.df.corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
            plt.title('Correlation Matrix of All Features')
            plt.show()

            # Assuming you have a target variable to pass as hue
            sns.pairplot(self.df, hue='target_variable_name')  # replace with the actual target variable name

            # Scatter Plot
            plt.scatter(self.df['feature_1'], self.df['feature_2'])
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.title('Scatter plot between Feature 1 and Feature 2')
            plt.show()

            # Boxplot
            sns.boxplot(data=self.df[['feature_1', 'feature_2']])
            plt.show()
            
            # Identifying outliers using Z-Score
            for column_name in self.df.select_dtypes(include=np.number).columns:
                z_scores = np.abs(zscore(self.df[column_name]))
                outliers_z = (z_scores > 3)
                # Log or print some info about the outliers
                num_outliers = np.sum(outliers_z)
                print(f"Feature: {column_name}, Outliers using Z-Score: {num_outliers}")
                
            # Identifying outliers using IQR
            # Loop through all numeric columns
            for column_name in self.df.select_dtypes(include=np.number).columns:
                # Identifying outliers using IQR
                Q1 = self.df[column_name].quantile(0.25)
                Q3 = self.df[column_name].quantile(0.75)
                IQR = Q3 - Q1
                outliers_iqr = ((self.df[column_name] < (Q1 - 1.5 * IQR)) | (self.df[column_name] > (Q3 + 1.5 * IQR)))

                # Log or print some info about the outliers
                num_outliers = np.sum(outliers_iqr)
                print(f"Feature: {column_name}, Outliers using IQR: {num_outliers}")
                
                # Visualization: Boxplot with outliers marked
                plt.figure(figsize=(10, 6))
                sns.boxplot(x=self.df[column_name])
                plt.scatter(np.where(outliers_iqr), self.df[column_name][outliers_iqr], marker='o', color='r', s=100)
                plt.title(f"Boxplot of {column_name} with Outliers Identified using IQR")
                plt.show()
                
            
            # Maybe visualize or log the outliers here.
            
        except Exception as e:
            logging.error(f"Error performing exploratory data analysis: {str(e)}")        


def main():
    sep_pattern = r'[;,\|]'
    file_path = 'D:/Temporary/SNORT_ML_Project/Python/Final_codes/mydata.csv'

    try:
        # Omitting error_bad_lines parameter
        df = pd.read_csv(file_path, sep=sep_pattern, engine='python', error_bad_lines=False, warn_bad_lines=True)
        df = df.applymap(lambda x: ' '.join(str(x).split()) if isinstance(x, str) else x)
        logging.info(f"Data loaded successfully from {file_path}.")
    except FileNotFoundError:
        logging.error(f"Data file not found at path: {file_path}.")
        return
    except pd.errors.EmptyDataError:
        logging.error("Data file is empty.")
        return
    except Exception as e:
        logging.error(f"Error in reading data file: {str(e)}")
        return
    
    try:
        processor = DataProcessor(df)
        processor.perform_eda() 
        pipeline = processor.preprocess_data()
        logging.info("Data preprocessed successfully.")
    except Exception as e:
        logging.error(f"Error in initializing DataProcessor or preprocessing data: {str(e)}")
        return

    df.to_csv('backup_df.csv', index=False)

    if 'Class' in df.columns:
        outlier_fraction = len(df[df['Class'] == 'anomaly']) / len(df)
        pipeline.named_steps['model'].set_params(contamination=outlier_fraction)

    X, y = (df.drop('Class', axis=1), df['Class']) if 'Class' in df.columns else (df, None)

    if y is not None:
        f1_scorer = make_scorer(f1_score, pos_label=-1)
        cv_score = cross_val_score(pipeline, X, y, cv=5, scoring=f1_scorer)
        print('Average F1 Score from Cross Validation:', np.mean(cv_score))

    X_train, X_val, y_train, y_val = (train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) 
                                      if y is not None 
                                      else train_test_split(X, test_size=0.2, random_state=42))

    pipeline.fit(X_train)

    if y_val is not None:
        scores = pipeline.decision_function(X_val)
        threshold = 0.1  # You may have to adjust the threshold value based on your specific use case
        predictions = [-1 if score < threshold else 1 for score in scores]
        print(classification_report(y_val, predictions))

    scores = pipeline.decision_function(X)
    threshold = 0.1  # Adjust the threshold as per your use case.
    predictions = [-1 if score < threshold else 1 for score in scores]
    anomalies_df = pd.DataFrame({'Index': df.index, 'Isolation_Forest_Anomalies': predictions})
    anomalies_df['Isolation_Forest_Anomalies'] = anomalies_df['Isolation_Forest_Anomalies'].map({1: 'normal', -1: 'anomaly'})

    df = df.merge(anomalies_df, left_index=True, right_on='Index').drop(columns=['Index'])
    
    for index, row in df.iterrows():
        if row['Isolation_Forest_Anomalies'] == 'anomaly':
            print(f"Anomaly detected at index {index}. Executing defined action...")

if __name__ == "__main__":
    main()
