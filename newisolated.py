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
from sklearn.feature_extraction.text import TfidfVectorizer
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

# logging.basicConfig(level=logging.INFO)
# Define the format and filename for logging
logging.basicConfig(filename='IsoForest_data_processor.log',
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
error_logger = logging.getLogger('error_logger')
error_logger.addHandler(logging.FileHandler('Isoforesterror.log'))


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

# =============================================================================
#   def identify_and_convert_columns(self):
#     try:
#        self.categorical_cols = self.df.select_dtypes(
#               include=['object']).columns.tolist()
#        self.continuous_cols = self.df.select_dtypes(
#               include=['float64', 'int64']).columns.tolist()
#     
#       for col in self.categorical_cols:
#         # Improved robust datetime check:
#         is_datetime = False
#         try:
#            for col in df.columns:
#              if df[col].dtype == 'object':
#                try:
#                   df[col] = pd.to_datetime(df[col])
#                 except ValueError:
#                   pass
#          is_datetime = True
#          except Exception as e:
#           print("An exception occurred: ", e)
#           #pd.to_datetime(self.df[col], errors='coerce').notna())
#           #df['date_column'] = pd.to_datetime(df['date_column']
#          except Exception as e:
#           logging.warning(
#               f"Unable to check datetime for column {col}: {str(e)}")
# 
#         if is_datetime:
#           self.df[col] = pd.to_datetime(self.df[col]).astype(int) / 10**9
#           if col not in self.continuous_cols:
#             self.continuous_cols.append(col)
#           self.categorical_cols.remove(col)
# 
#       logging.info("Columns identified and converted where necessary.")
# 
#       # Assertions to verify column names
#       assert all(col in self.df.columns for col in self.categorical_cols
#                  ), "Some categorical columns are not in DataFrame"
#       assert all(
#           col in self.df.columns for col in
#           self.continuous_cols), "Some continuous columns are not in DataFrame"
#     except Exception as e:
#       logging.error(f"Error in identifying and converting columns: {str(e)}")
#       raise
# =============================================================================
  def identify_and_convert_columns(self):
    try:
        self.categorical_cols = self.df.select_dtypes(
            include=['object']).columns.tolist()
        self.continuous_cols = self.df.select_dtypes(
            include=['float64', 'int64']).columns.tolist()

        for col in self.categorical_cols:
            # Improved robust datetime check:
            is_datetime = False
            try:
                for column in self.df.columns:  # <-- Changed `df` to `self.df` here
                    if self.df[column].dtype == 'object':  # <-- Changed `df` to `self.df` here
                        try:
                            self.df[column] = pd.to_datetime(self.df[column])  # <-- Changed `df` to `self.df` here
                            is_datetime = True
                        except ValueError:
                            pass
            except Exception as e:
                print("An exception occurred: ", e)
                # The following lines appear to be commented out and may be removed or uncommented as needed.
                #pd.to_datetime(self.df[col], errors='coerce').notna())
                #df['date_column'] = pd.to_datetime(df['date_column']

            if is_datetime:
                self.df[col] = pd.to_datetime(self.df[col]).astype(int) / 10**9
                if col not in self.continuous_cols:
                    self.continuous_cols.append(col)
                self.categorical_cols.remove(col)

        logging.info("Columns identified and converted where necessary.")

        # Assertions to verify column names
        assert all(col in self.df.columns for col in self.categorical_cols), "Some categorical columns are not in DataFrame"
        assert all(col in self.df.columns for col in self.continuous_cols), "Some continuous columns are not in DataFrame"
    except Exception as e:
        logging.error(f"Error in identifying and converting columns: {str(e)}")
        raise

  def handle_missing_values(self):
    try:
      print("Categorical Columns: ", self.categorical_cols)
      print("Continuous Columns: ", self.continuous_cols)
      if self.categorical_cols:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        self.df[self.categorical_cols] = cat_imputer.fit_transform(
            self.df[self.categorical_cols])

      if self.continuous_cols:
        # Identify columns with all missing values
        all_na_cols = self.df[self.continuous_cols].columns[self.df[
            self.continuous_cols].isna().all()].tolist()
        some_na_cols = [
            col for col in self.continuous_cols if col not in all_na_cols
        ]

        # For columns with some NA values - use mean strategy
        if some_na_cols:
          cont_imputer = SimpleImputer(strategy='mean')
          self.df[some_na_cols] = cont_imputer.fit_transform(
              self.df[some_na_cols])

        # For columns with all NA values - decide a strategy
        # Strategy 1: Remove them
        self.df = self.df.drop(columns=all_na_cols)

        # Or Strategy 2: Fill them with a constant (e.g., 0)
        self.df[all_na_cols] = self.df[all_na_cols].fillna(0)

        # Update continuous_cols list if you've dropped columns
        self.continuous_cols = [
            col for col in self.continuous_cols if col not in all_na_cols
        ]
    except Exception:
      logging.error("Error in handling missing values.", exc_info=True)

  def handle_text_columns(self, fraction=0.5):
    try:
      self.text_cols = []
      self.tfidf_feature_names = []
      for col in self.df.columns:
        if self.df[col].dtype == 'object' and self.df[col].str.contains(
            '\s').any():
          self.text_cols.append(col)

      for text_col in self.text_cols:
        # Count unique words and determine max_features dynamically
        unique_word_count = len(
            set(" ".join(self.df[text_col].astype('U')).split()))
        max_features = int(unique_word_count * fraction)

        vectorizer = TfidfVectorizer(max_features=max_features)
        tfidf_features = vectorizer.fit_transform(
            self.df[text_col].astype('U'))

        tfidf_col_names = [
            f"{text_col}_{i}" for i in range(tfidf_features.shape[1])
        ]
        tfidf_df = pd.DataFrame(tfidf_features.toarray(),
                                columns=tfidf_col_names)

        self.df = pd.concat([self.df, tfidf_df],
                            axis=1).drop(columns=[text_col])
        self.tfidf_feature_names.extend(tfidf_col_names)
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

      non_text_transformer = ColumnTransformer(transformers=[
          ('num', StandardScaler(), self.continuous_cols),
          ('cat', OneHotEncoder(), self.categorical_cols)
      ],
                                               remainder='passthrough')

      preprocessor = FeatureUnion(
          transformer_list=[(
              'text',
              Pipeline([('selector',
                         FunctionTransformer(text_data, validate=False)
                         ), ('vectorizer', CountVectorizer())])),
                            ('non_text',
                             Pipeline([('selector',
                                        FunctionTransformer(non_text_data,
                                                            validate=False)
                                        ), ('transformer',
                                            non_text_transformer)]))])

      return Pipeline([('preprocessor', preprocessor),
                       ('model',
                        IsolationForest(contamination=0.01, random_state=42))])
    except Exception as e:
      logging.error(f"Error in preprocessing data: {str(e)}")
      raise

  def perform_eda(self, target_variable_name=None):
    """
    Perform exploratory data analysis on the dataset.
    """
    try:
        # Correlation Matrix Heatmap
        correlation_matrix = self.df.corr(numeric_only=True)
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix of All Features')
        plt.show()

        # Pairplot (only if target variable is provided)
        if target_variable_name and target_variable_name in self.df.columns:
            sns.pairplot(self.df, hue=target_variable_name)
            plt.show()

        # Scatter Plot for first two numeric features (just an example)
        numeric_columns = self.df.select_dtypes(include=np.number).columns
        if len(numeric_columns) > 1:
            plt.scatter(self.df[numeric_columns[0]], self.df[numeric_columns[1]])
            plt.xlabel(numeric_columns[0])
            plt.ylabel(numeric_columns[1])
            plt.title(f'Scatter plot between {numeric_columns[0]} and {numeric_columns[1]}')
            plt.show()

        # Boxplot for all numeric features
        sns.boxplot(data=self.df[numeric_columns])
        plt.show()

        # Identifying outliers using Z-Score and IQR
        for column_name in numeric_columns:
            # Using Z-Score
            z_scores = np.abs(zscore(self.df[column_name]))
            outliers_z = (z_scores > 3)
            num_outliers = np.sum(outliers_z)
            print(f"Feature: {column_name}, Outliers using Z-Score: {num_outliers}")

            # Using IQR
            Q1 = self.df[column_name].quantile(0.25)
            Q3 = self.df[column_name].quantile(0.75)
            IQR = Q3 - Q1
            outliers_iqr = ((self.df[column_name] < (Q1 - 1.5 * IQR)) | (self.df[column_name] > (Q3 + 1.5 * IQR)))
            num_outliers = np.sum(outliers_iqr)
            print(f"Feature: {column_name}, Outliers using IQR: {num_outliers}")

            # Visualization: Boxplot with outliers marked using IQR
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=self.df[column_name])
            plt.scatter(np.where(outliers_iqr), self.df[column_name][outliers_iqr], marker='o', color='r', s=100)
            plt.title(f"Boxplot of {column_name} with Outliers Identified using IQR")
            plt.show()

    except Exception as e:
        logging.error(f"Error performing exploratory data analysis: {str(e)}")

      
  def false_positive_rate(y_true, y_pred):
    """Compute False Positive Rate."""
    # Determine unique labels
    unique_labels = set(y_true) | set(y_pred)
    if len(unique_labels) != 2:
        raise ValueError("Expected exactly two unique labels")

    # Determine which label is "positive" and which is "negative"
    sorted_labels = sorted(list(unique_labels))
    negative_label, positive_label = sorted_labels

    # Define true negatives (TN), false positives (FP), true positives (TP), and false negatives (FN)
    TN = sum((y_true == negative_label) & (y_pred == negative_label))
    FP = sum((y_true == negative_label) & (y_pred == positive_label))
    #TP = sum((y_true == positive_label) & (y_pred == positive_label))
    #FN = sum((y_true == positive_label) & (y_pred == negative_label))

    # Calculate FPR
    FPR = FP / (FP + TN)

    return FPR

def main():
  sep_pattern = r'[;,\|\s]'
  file_path = 'packets_2.csv'
  # Read the first line of the file to get the number of columns
  try:
    with open(file_path, 'r') as file:
      first_line = file.readline()
      num_columns = len(
          first_line.split(','))  # assuming comma-separated values
  except FileNotFoundError:
    print(f"Data file not found at path: {file_path}.")
    num_columns = 0  # or handle this appropriately

  # Generate dynamic column names
  column_names = [f'col_{i}' for i in range(num_columns)]

  # A function to warn about inconsistent line
  def warn_inconsistent_line(line: str, row_num: int):
    logging.warning(
        f"Inconsistent number of columns in line {row_num}: {line}")

  # Load the dataframe with handling of inconsistent rows
  try:
    df = pd.read_csv(
        file_path,
        sep=sep_pattern,
        quotechar='"',
        engine='python',
        header=None,
        names=column_names,
        on_bad_lines='warn',
        error_bad_lines=False
        #  warn_inconsistent_line=warn_inconsistent_line,
        # This will show a warning message for each bad line
    )

    df = df.applymap(lambda x: ' '.join(str(x).split())
                     if isinstance(x, str) else x)
    logging.info(f"Data loaded successfully from {file_path}.")
  except FileNotFoundError:
    logging.error(f"Data file not found at path: {file_path}.")
    return
  except pd.errors.EmptyDataError:
    logging.error("Data file is empty.")
    return
  except Exception as e:
    logging.error(f"Encountered an error: {str(e)}", exc_info=True)
    return

  try:
    processor = DataProcessor(df)
    processor.handle_missing_values()
    processor.perform_eda()
    pipeline = processor.preprocess_data()
    logging.info("Data preprocessed successfully.")
  except Exception as e:
    logging.error(
        f"Error in initializing DataProcessor or preprocessing data: {str(e)}")
    return

  df.to_csv('backup_df.csv', index=False)

  if 'Class' in df.columns:
    outlier_fraction = len(df[df['Class'] == 'anomaly']) / len(df)
    pipeline.named_steps['model'].set_params(contamination=outlier_fraction)
    
    # Convert anomaly and normal labels to 1 and -1 respectively for Isolation Forest
    df['Class'] = df['Class'].map({'anomaly': 1, 'normal': -1})

  X, y = (df.drop('Class', axis=1),
          df['Class']) if 'Class' in df.columns else (df, None)

  if y is not None:
    f1_scorer = make_scorer(f1_score, pos_label=-1)
    cv_score = cross_val_score(pipeline, X, y, cv=5, scoring=f1_scorer)
    print('Average F1 Score from Cross Validation:', np.mean(cv_score))

  if y is not None:
    X_train, X_val, y_train, y_val = train_test_split(X,
                                                y,
                                                test_size=0.2,
                                                random_state=42,
                                                stratify=y)
  else:
    X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)
    y_train, y_val = None, None

  # Training the pipeline
  pipeline.fit(X_train, y_train)
  
  # Making predictions on the test set
  y_pred = pipeline.predict(X_val)
  
  # Evaluating the model
  f1_scorer = make_scorer(f1_score, pos_label=1)
  f1 = f1_scorer(pipeline, X_val, y_val)
  print(f"F1 Score: {f1:.4f}")

  # Computing the False Positive Rate (FPR)
  fpr = DataProcessor.false_positive_rate(y_val, y_pred)
  print(f"False Positive Rate (FPR): {fpr:.4f}")
  
  # Additionally, printing a classification report for a detailed overview
  print("\nClassification Report:\n")
  print(classification_report(y_val, y_pred))

  if y_val is not None:
    scores = pipeline.decision_function(X_val)
    threshold = 0.1  # You may have to adjust the threshold value based on your specific use case
    predictions = [-1 if score < threshold else 1 for score in scores]
    print(classification_report(y_val, predictions))

  scores = pipeline.decision_function(X)
  threshold = 0.1  # Adjust the threshold as per your use case.
  predictions = [-1 if score < threshold else 1 for score in scores]
  anomalies_df = pd.DataFrame({
      'Index': df.index,
      'Isolation_Forest_Anomalies': predictions
  })
  anomalies_df['Isolation_Forest_Anomalies'] = anomalies_df[
      'Isolation_Forest_Anomalies'].map({
          1: 'normal',
          -1: 'anomaly'
      })

  df = df.merge(anomalies_df, left_index=True,
                right_on='Index').drop(columns=['Index'])

  for index, row in df.iterrows():
    if row['Isolation_Forest_Anomalies'] == 'anomaly':
      print(f"Anomaly detected at index {index}. Executing defined action...")


if __name__ == "__main__":
  main()
