import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
import pickle
import os

def main(file_path):
    os.makedirs('./mlpickle', exist_ok=True)
    csv_file = file_path
    print("#### Reading from training data")
    df = pd.read_csv(csv_file)
    df_numeric, df_categorical = separate_categorical_continuous_data_and_impute(df)
    new_df = pd.concat([df_numeric, df_categorical], axis=1)
    new_df = new_df[['ip_src', 'tcp_srcport', 'ip_dst', 'tcp_dstport', 'frame_time', 'frame_protocols', 'http_request_method', 'http_request_uri', 'http_host', 'http_user_agent', 'http_referer', 'http_response_code', 'http_content_type']]
    text_columns = df_categorical.columns.to_list()
    new_df1,labenc = handle_text_columns(new_df, text_columns)
    iso_forest = IsolationForest(n_jobs=10, random_state=42,n_estimators=200, max_features=13)
    iso_forest.fit(new_df1)
    with open('./mlpickle/le.pkl', 'wb') as f:
        pickle.dump(labenc, f)
        print("#### ecoding saved successfully")
    with open('./mlpickle/model.pkl', 'wb') as f:
        pickle.dump(iso_forest, f)
        print("#### trained model saved successfully")

def handle_text_columns(df, text_cols):
    le_objects=[]
    for text_col in text_cols:
        le=LabelEncoder()
        le.fit_transform(df[text_col].astype('U'))
        df[text_col]=le.transform(df[text_col])
        le_objects.append(le)
    return df,le_objects


def separate_categorical_continuous_data_and_impute(df):
    imputer_categorical = SimpleImputer(strategy="most_frequent")
    imputer_continuous = SimpleImputer(strategy="mean")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df.select_dtypes(exclude=['float64', 'int64']).columns
    df_numeric = df[numeric_cols]
    df_categorical = df[categorical_cols]
    df_categorical = imputer_categorical.fit_transform(df_categorical)
    df_numeric = imputer_continuous.fit_transform(df_numeric)
    with open('./mlpickle/cont_imp.pkl', 'wb') as f:
        pickle.dump(imputer_continuous, f)
        print("#### imputer continuous saved successfully")
    with open('./mlpickle/cat_imp.pkl', 'wb') as f:
        pickle.dump(imputer_categorical, f)
        print("#### imputer categorical saved successfully")
    df_categorical = pd.DataFrame(df_categorical, columns=categorical_cols)
    df_numeric = pd.DataFrame(df_numeric, columns=numeric_cols)

    return df_numeric, df_categorical



