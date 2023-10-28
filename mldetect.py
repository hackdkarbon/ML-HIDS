import base64
import socket
import pickle
import threading
import pandas as pd
import numpy as np
import asyncio
ana= []
non_ana  = []
model_file_path = "mlpickle"
try:
    with open(model_file_path + '/model.pkl', 'rb') as model_file:
        trained_model = pickle.load(model_file)
    with open(model_file_path + '/cat_imp.pkl', 'rb') as model_file:
        cat_imp = pickle.load(model_file)
    with open(model_file_path + '/cont_imp.pkl', 'rb') as model_file:
        cont_imp = pickle.load(model_file)
    with open(model_file_path + '/le.pkl', 'rb') as model_file:
        le = pickle.load(model_file)
except:
    print("#### Error in loading ML pickle")                
    
def process_data(data):

    elements = data.split(',')

    # Organize the elements into a structured format
    sample = [
        elements[0] if elements[0] else "0.0.0.0",
        float(elements[1]) if elements[1] else 0.0,  # Use a default float value
        elements[2] if elements[2] else "9.9.9.9",
        float(elements[3]) if elements[3] else 0.0,  # Use a default float value
        float(elements[4]) if elements[4] else 0.0,  # Use a default float value
        elements[5] if elements[5] else "1234567890",
        elements[6] if elements[6] else "GET",
        elements[7] if elements[7] else "http://localhost",
        elements[8] if elements[8] else "localhost",
        elements[9] if elements[9] else "default",
        elements[10] if elements[10] else "http://referrer99.com",
        elements[11] if elements[11] else "200",
        elements[12] if elements[12] else "text/html"
    ]

    # Define the column order
    column = ['ip_src', 'tcp_srcport', 'ip_dst', 'tcp_dstport', 'frame_time', 'frame_protocols', 'http_request_method', 'http_request_uri', 'http_host', 'http_user_agent', 'http_referer', 'http_response_code', 'http_content_type']

    df = pd.DataFrame([sample], columns=column)
    numeric_cols = [ 'tcp_srcport', 'tcp_dstport', 'frame_time', 'http_response_code']
    categorical_cols = ['ip_src', 'ip_dst', 'frame_protocols', 'http_request_method', 'http_request_uri', 'http_host', 'http_user_agent', 'http_referer', 'http_content_type']

    df_numeric = df[numeric_cols]
    df_categorical = df[categorical_cols]
    df_numeric = cont_imp.transform(df_numeric)
    df_categorical = cat_imp.transform(df_categorical)
    df_categorical = pd.DataFrame(df_categorical, columns=categorical_cols)
    df_numeric = pd.DataFrame(df_numeric, columns=numeric_cols)

    new_df = pd.concat([df_numeric, df_categorical], axis=1)
    new_df = new_df[column]
    for i in range(len(categorical_cols)):
        #print(f"{i}{new_df[categorical_cols[i]]}")
        #print(le[t].transform(new_df[t]))
        new_df[categorical_cols[i]] = le[i].transform(new_df[categorical_cols[i]].astype('U'))
        #print(new_df[categorical_cols[i]])
        #print(le[t].inverse_transform(new_df[t]))    pred = trained_model.predict(new_df)
    pred = trained_model.predict(new_df)
    if(pred == 1):
        print("#### ML: known traffic data")
        non_ana.append(sample)
        with open('anomaly-data/ml_good_data.csv', 'a') as output:
            output.write(str(sample)+'\n')
            output.write(str(new_df) + '\n')
    else:
        print(f"#################### ML: Anomaly Detected ###############")
        ana.append(sample)
        with open('anomaly-data/ml_anomaly_data.csv', 'a') as output:
            output.write(str(sample)+'\n')
            output.write(str(new_df) + '\n')
