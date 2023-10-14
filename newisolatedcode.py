# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import make_scorer, f1_score, classification_report
import logging
import ipaddress
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
import pickle

# Logging configuration
logging.basicConfig(filename='IsoForest_data_processor.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
error_logger = logging.getLogger('error_logger')
error_logger.addHandler(logging.FileHandler('Isoforesterror.log'))

class DataProcessor:
    def __init__(self, df):
        if df is None or df.empty:
            raise ValueError("DataFrame is empty or None")
        self.df = df
        self.categorical_cols = None
        self.continuous_cols = None
        self.pipeline = None
        self.identify_and_convert_columns()
        self.handle_missing_values()
        self.handle_text_columns()
        self.build_model()
    def identify_and_convert_columns(self):
        self.categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        self.continuous_cols = self.df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
    def handle_missing_values(self):
        if self.categorical_cols:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            self.df[self.categorical_cols] = cat_imputer.fit_transform(self.df[self.categorical_cols])

        if self.continuous_cols:
            all_na_cols = self.df[self.continuous_cols].columns[self.df[self.continuous_cols].isna().all()].tolist()
            some_na_cols = [col for col in self.continuous_cols if col not in all_na_cols]

            if some_na_cols:
                cont_imputer = SimpleImputer(strategy='mean')
                self.df[some_na_cols] = cont_imputer.fit_transform(self.df[some_na_cols])

            self.df = self.df.drop(columns=all_na_cols)
            self.df.fillna(0, inplace=True)
            self.continuous_cols = [col for col in self.continuous_cols if col not in all_na_cols]

    def handle_text_columns(self, fraction=0.5):
        self.text_cols = []
        self.tfidf_feature_names = []
        for col in self.df.columns:
            if self.df[col].dtype == 'object' and self.df[col].str.contains('\s').any():
                self.text_cols.append(col)

        for text_col in self.text_cols:
            unique_word_count = len(set(" ".join(self.df[text_col].astype('U')).split()))
            max_features = int(unique_word_count * fraction)

            vectorizer = TfidfVectorizer(max_features=max_features)
            tfidf_features = vectorizer.fit_transform(self.df[text_col].astype('U'))
            tfidf_col_names = [f"{text_col}_{i}" for i in range(tfidf_features.shape[1])]
            tfidf_df = pd.DataFrame(tfidf_features.toarray(), columns=tfidf_col_names)

            self.df = pd.concat([self.df, tfidf_df], axis=1).drop(columns=[text_col])
            self.tfidf_feature_names.extend(tfidf_col_names)

    def preprocess_data(self):
        def text_data(df):
            return df[self.text_cols] if self.text_cols else pd.DataFrame()
        
        def non_text_data(df):
            return df.drop(columns=self.text_cols) if self.text_cols else df
        
        non_text_transformer = ColumnTransformer(transformers=[
            ('num', StandardScaler(), self.continuous_cols),
            ('cat', OneHotEncoder(), self.categorical_cols)
        ], remainder='passthrough')

        preprocessor = FeatureUnion(transformer_list=[
    ('text', Pipeline([
        ('selector', FunctionTransformer(text_data, validate=False)),
        ('vectorizer', CountVectorizer())
    ])),
    ('non_text', Pipeline([
        ('selector', FunctionTransformer(non_text_data, validate=False)),
        ('transformer', non_text_transformer)
    ]))
])

        self.pipeline = Pipeline([('preprocessor', preprocessor), ('model', IsolationForest(contamination=0.01, random_state=42))])
        return 

def main():
    file_path = 'train.csv'
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Data loaded successfully from {file_path}.")
    except FileNotFoundError:
        logging.error(f"Data file not found at path: {file_path}.")
        return
    except pd.errors.EmptyDataError:
        logging.error("Data file is empty.")
        return
    processor = DataProcessor(df)
    
    

if __name__ == "__main__":
    main()
