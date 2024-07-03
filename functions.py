
import pandas as pd
import json
import os
import numpy as np
import requests
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from joblib import dump, load


def everything(df_list,lvl1,lvl2,shift,chunk,threshold,target_shift,testsize):
    df_new_list = num_to_cik(df_list)
    merged_df = Convertor_Multiple(df_new_list)
    na_maxed = na_max(merged_df,threshold)
    df_num = frame_num_drop(na_maxed)
    divided = df_asset_target(df_num)
    merged = process_in_chunks(divided,lvl1,lvl2,shift,chunk)
    df_targeted=df_target_dropna(merged,lvl1,lvl2,target_shift)
    df_shuffled = df_targeted.sample(frac=1, random_state=42)
    train_set, test_set = train_test_split(df_shuffled, test_size=testsize, random_state=42)
    df = train_set
    df2 = test_set
    rf_model, train_accuracy, train_roc_auc = train_random_forest(df, 'target')
    save_model(rf_model, 'random_forest_classifier.joblib')
    rf_model_loaded = load_model('random_forest_classifier.joblib')
    unseen_accuracy, unseen_roc_auc = evaluate_model(rf_model_loaded, df2, 'target')
    print(f"Training Accuracy: {train_accuracy}")
    print(f"Training ROC AUC: {train_roc_auc}")
    print(f"Unseen Data Accuracy: {unseen_accuracy}")
    print(f"Unseen Data ROC AUC: {unseen_roc_auc}")
    print(df_info(df))

    return train_accuracy, train_roc_auc, unseen_accuracy, unseen_roc_auc

def na_max(df,threshold):

    missing_values = df.isnull().sum()

    # Calculate the percentage of missing values in each column
    missing_percentage = missing_values / len(df)

    # Get the columns where the missing percentage is greater than the threshold
    columns_to_drop = missing_percentage[missing_percentage > threshold].index

    # Drop the columns from the DataFrame
    df = df.drop(columns=columns_to_drop)
    return df


def Convertor_Single(path):
    with open(path, 'r') as json_file:
        data = json.load(json_file)

    columns = set()
    values = {}
    cik = data['cik']
    entity_name = data['entityName']

    for key, value in data['facts']['us-gaap'].items():
        for unit_key in value['units'].keys():
            for entry in value['units'][unit_key]:
                if 'frame' in entry.keys():
                    frame = entry['frame']
                    if frame.endswith('I'):
                        frame = frame[:-1]
                    row_key = (frame, cik)
                    columns.add(key)
                    values.setdefault(row_key, {}).update({
                        'CIK': cik,
                        'EntityName': entity_name,
                        'Form': entry['form'],
                        'FP': entry['fp'],
                        'End': entry['end'],
                        'Filed': entry['filed'],
                        'Frame': frame,
                        key: entry['val']
                    })

    df = pd.DataFrame.from_dict(values, orient='index')
    return df

import pandas as pd
import json

# def Convertor_Multiple(file_paths):
#     all_dfs = []
    
#     for idx, path in enumerate(file_paths, start=1):
#         with open(path, 'r') as json_file:
#             data = json.load(json_file)

#         columns = set()
#         values = {}
#         cik = data['cik']
#         entity_name = data['entityName']

#         for key, value in data['facts']['us-gaap'].items():
#             for unit_key in value['units'].keys():
#                 for entry in value['units'][unit_key]:
#                     if 'frame' in entry.keys():
#                         frame = entry['frame']
#                         if frame.endswith('I'):
#                             frame = frame[:-1]
#                         row_key = (frame, cik)
#                         columns.add(key)
#                         values.setdefault(row_key, {}).update({
#                             'CIK': cik,
#                             'EntityName': entity_name,
#                             'Form': entry['form'],
#                             'FP': entry['fp'],
#                             'End': entry['end'],
#                             'Filed': entry['filed'],
#                             'Frame': frame,
#                             key: entry['val']
#                         })

#         df = pd.DataFrame.from_dict(values, orient='index')
#         all_dfs.append(df)

#     merged_df = pd.concat(all_dfs)
#     return merged_df

# import json
# import pandas as pd

def Convertor_Multiple(file_paths):
    all_dfs = []
    
    for idx, path in enumerate(file_paths, start=1):
        with open(path, 'r') as json_file:
            data = json.load(json_file)

        columns = set()
        values = {}
        
        try:
            cik = data['cik']
            entity_name = data['entityName']
            us_gaap_facts = data['facts']['us-gaap']
        except KeyError as e:
            print(f"Key {e} missing in file {path}. Skipping this file.")
            continue

        for key, value in us_gaap_facts.items():
            for unit_key in value['units'].keys():
                for entry in value['units'][unit_key]:
                    try:
                        if 'frame' in entry:
                            frame = entry['frame']
                            if frame.endswith('I'):
                                frame = frame[:-1]
                            row_key = (frame, cik)
                            columns.add(key)
                            values.setdefault(row_key, {}).update({
                                'CIK': cik,
                                'EntityName': entity_name,
                                'Form': entry['form'],
                                'FP': entry['fp'],
                                'End': entry['end'],
                                'Filed': entry['filed'],
                                'Frame': frame,
                                key: entry['val']
                            })
                    except KeyError as e:
                        print(f"Key {e} missing in entry for {key}. Skipping this entry.")

        df = pd.DataFrame.from_dict(values, orient='index')
        all_dfs.append(df)

    merged_df = pd.concat(all_dfs, ignore_index=True)
    return merged_df


def df_info(df):
    # Calculate the percentage of missing values
    missing_percentage = ((df.isnull().sum().sum()) / (df.shape[0] * df.shape[1])).round(2)
    
    # Get the shape of the DataFrame
    shape = df.shape
    
    return missing_percentage, shape

def random_forest_classifier(df, target_column, test_size, random_state=42):
    # Step 1: Prepare your data
    # Separate features (X) and target variable (y)
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Step 2: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Step 3: Model training
    rf_classifier = RandomForestClassifier()
    rf_classifier.fit(X_train, y_train)

    # Step 4: Model evaluation
    # Accuracy
    y_pred = rf_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # AUC/ROC
    y_pred_proba = rf_classifier.predict_proba(X_test)[:, 1]
    auc_roc = roc_auc_score(y_test, y_pred_proba)

    return accuracy, auc_roc

def frame_num_drop(df):
    
    # Define quarters list
    years_quarters = [
        "CY2012Q1", "CY2012Q2", "CY2012Q3", "CY2012Q4",
        "CY2013Q1", "CY2013Q2", "CY2013Q3", "CY2013Q4",
        "CY2014Q1", "CY2014Q2", "CY2014Q3", "CY2014Q4",
        "CY2015Q1", "CY2015Q2", "CY2015Q3", "CY2015Q4",
        "CY2016Q1", "CY2016Q2", "CY2016Q3", "CY2016Q4",
        "CY2017Q1", "CY2017Q2", "CY2017Q3", "CY2017Q4",
        "CY2018Q1", "CY2018Q2", "CY2018Q3", "CY2018Q4",
        "CY2019Q1", "CY2019Q2", "CY2019Q3", "CY2019Q4",
        "CY2020Q1", "CY2020Q2", "CY2020Q3", "CY2020Q4",
        "CY2021Q1", "CY2021Q2", "CY2021Q3", "CY2021Q4",
        "CY2022Q1", "CY2022Q2", "CY2022Q3", "CY2022Q4",
        "CY2023Q1", "CY2023Q2", "CY2023Q3", "CY2023Q4"
    ]

    # Filter dataframe based on quarters list
    df = df[df['Frame'].isin(years_quarters)]
    df = df.sort_values(by='Frame', ascending=True)
    df = pd.pivot_table(df, index=['CIK', 'Frame'])
    df = df.reset_index()
    df = df.dropna(subset=['EarningsPerShareBasic'])
    df = df.fillna(0)
    
    # Calculate division result
    # df_num = df.drop([#'CIK', 
    #                   'EntityName', 'Form', 'FP', 'End', 'Filed'
    #                 #, 'Frame'
    #                 ], axis=1)
    return df

def df_reshape(df, column_name, num_pivot=10):
    warnings.filterwarnings("ignore")
    df = df.dropna(axis=1, how='any')
    division_result = (df.shift(-1) / df) - 1
    df_difference = division_result.replace([np.inf, -np.inf, np.nan], 0).round(2)
    df_difference = df_difference[~(df_difference == 0).all(axis=1)]
    df = df_difference
    df = df.dropna()

    # Reshape dataframe
    df_new = pd.DataFrame()
    for col in df.columns:
        for i in range(0, num_pivot):
            df_new[f"{col}_{i}"] = df[col].shift(-i)
    
    target_column = f"{column_name}_{num_pivot-1}"
    df_new['target'] = df_new[target_column].shift(-1)
    df_new = df_new.dropna()

    # Create target variable
    def map_to_binary(value):
        if value < 0:
            return 0
        else:
            return 1
    df_new['target'] = df_new['target'].apply(lambda x: map_to_binary(x))
    
    return df_new

def df_pivot_multiple(df, level_1, level_2):
    pivoted = pd.pivot_table(df, index=[level_1, level_2])
    shift_columns = pivoted.columns
    shift_periods = [-1, -2]
    new_columns_order = []

    for col in shift_columns:
        new_columns_order.append(col)
        for period in shift_periods:
            new_columns_order.append(f'{col}_{abs(period)}')

    pivoted = pivoted.reindex(columns=new_columns_order)

    for col in shift_columns:
        for period in shift_periods:
            pivoted[f'{col}_{abs(period)}'] = pivoted.groupby(level=level_1)[col].shift(periods=period)
    
    return pivoted


def pivot_diff_lag(df, level_1, level_2, shift):
    shift_periods = [-i for i in range(1, shift + 1)]
    pivoted = pd.pivot_table(df, index=[level_1, level_2])
    df_diff = pivoted.groupby(level_1).pct_change().round(2)
    shift_columns = df_diff.columns
    for col in shift_columns:
        for period in shift_periods:
            df_diff[f'{col}_{abs(period)}'] = df_diff.groupby(level=level_1)[col].shift(periods=period)
    desired_order = sorted(list(df_diff.columns))
    df_diff = df_diff[desired_order]
    return df_diff

# Example usage:
# df_diff = create_shifted_pivot_table(df, 'Company', 'Year', 2)

def num_to_cik(num_list):
    newlist = []
    for i in num_list:
        if len(str(i))==4:
            newlist.append('companyfacts/CIK000000' + str(i)+'.json')
        if len(str(i))==5:
            newlist.append('companyfacts/CIK00000' + str(i)+'.json')
        if len(str(i))==6:
            newlist.append('companyfacts/CIK0000' + str(i)+'.json')
        if len(str(i))==7:
            newlist.append('companyfacts/CIK000' + str(i)+'.json')
    return newlist

# def pivot_lag(df, level_1, level_2, shift):
#     shift_periods = [-i for i in range(1, shift + 1)]
#     pivoted = pd.pivot_table(df, index=[level_1, level_2])
#     shift_columns = pivoted.columns
#     for col in shift_columns:
#         for period in shift_periods:
#             pivoted[f'{col}_{abs(period)}'] = pivoted.groupby(level=level_1)[col].shift(periods=period)
#     desired_order = sorted(list(pivoted.columns))
#     pivoted = pivoted[desired_order]
#     return pivoted

def pivot_lag(df, level_1, level_2, shift):
    shift_periods = [-i for i in range(1, shift + 1)]
    pivoted = pd.pivot_table(df, index=[level_1, level_2])
    new_columns = {}

    for col in pivoted.columns:
        for period in shift_periods:
            new_col_name = f'{col}_{abs(period)}'
            new_columns[new_col_name] = pivoted.groupby(level=level_1)[col].shift(periods=period)

    # Use pd.concat to join all new columns at once
    pivoted = pd.concat([pivoted, pd.DataFrame(new_columns, index=pivoted.index)], axis=1)
    desired_order = sorted(list(pivoted.columns))
    pivoted = pivoted[desired_order]
    return pivoted


# def df_target_dropna(df,lvl1,lvl2,target_name):
#     df = pd.pivot_table(df, index=[lvl1,lvl2])
#     difference = df[df.columns[-2]] - df[df.columns[-1]]
#     df['real_target'] = (difference <= 0).astype(int)

#    # pivoted['real_target']=pivoted.groupby(level='CIK')[target_name].shift(-1)
#     df = df.reset_index()
#     df=df.dropna()
#     return df

def df_target_dropna(df,lvl1,lvl2,last):
    df = pd.pivot_table(df, index=[lvl1,lvl2])
    df['target_lag'] = df.groupby(level=lvl1)[last].shift(-1)
    df['target'] = 0
    df.loc[df[last] < df[df.columns[-2]], 'target'] = 1
    df.loc[df[last] > df[df.columns[-2]], 'target'] = 0
    df = df.dropna()
    df = df.drop('target_lag',axis=1)
    df = df.reset_index()
    df = df.drop(lvl1,axis=1)
    df = df.drop(lvl2,axis=1)
    return df

# def df_asset_target(df_num):
#     df_num_1 = df_num[['CIK','Frame']]
#     df_num_2 = df_num[df_num.columns[2:]]
#     target =df_num['EarningsPerShareBasic']
#     df_num_2 = df_num_2.div(df_num['Assets'], axis=0)
#     df_num_2 = df_num_2.drop('Assets',axis=1)
#     df_num = pd.concat([df_num_1,df_num_2],axis=1)
#     df_num['target']=target
#     pivoted = pd.pivot_table(df_num, index=['CIK', 'Frame'])
#     pivoted = pivoted.groupby(level='CIK')['target'].shift(-1)
#     pivoted = pivoted.reset_index()
#     return pivoted

# def df_asset_target(df_num):
#     df_num_1 = df_num[['CIK','Frame']]
#     df_num_2 = df_num[df_num.columns[2:]]
#     target =df_num['EarningsPerShareBasic']
#     df_num_2 = df_num_2.div(df_num['Assets'], axis=0)
#     df_num_2 = df_num_2.drop('Assets',axis=1)
#     df_num = pd.concat([df_num_1,df_num_2],axis=1)
#     df_num['target_2']=target
#     pivoted = pd.pivot_table(df_num, index=['CIK', 'Frame'])
#     pivoted['target'] = pivoted.groupby(level='CIK')['target_2'].shift(-1)
#     pivoted = pivoted.drop('target_2',axis=1)
#     pivoted = pivoted.reset_index()
#     return pivoted

# def df_asset_target(df_num):
#     df_num_1 = df_num[['CIK','Frame']]
#     df_num_2 = df_num[df_num.columns[2:]]
#     target =df_num['EarningsPerShareBasic']
#     df_num_2 = df_num_2.div(df_num['Assets'], axis=0)
#     df_num_2 = df_num_2.drop('Assets',axis=1)
#     df_num = pd.concat([df_num_1,df_num_2],axis=1)
#     df_num['target']=target
#     return df_num

def df_asset_target(df_num):
    df_num_1 = df_num[['CIK','Frame']]
    df_num_2 = df_num[df_num.columns[2:]]
    df_num_2 = df_num_2.div(df_num['Assets'], axis=0)
    df_num_2 = df_num_2.drop('Assets',axis=1)
    df_num = pd.concat([df_num_1,df_num_2],axis=1)
    return df_num
    
def process_in_chunks(df, key_column_1, key_column_2, number_of_shifts, n):
    # Ensure the key columns are at the front
    cols = [key_column_1, key_column_2] + [col for col in df if col not in [key_column_1, key_column_2]]
    df = df[cols]
    
    # Calculate the number of columns per chunk, excluding the first two key columns
    columns_per_chunk = (len(df.columns) - 2) // n
    
    # List to hold each processed chunk
    processed_chunks = []
    
    for i in range(0, len(df.columns) - 2, columns_per_chunk):
        # Select columns for the chunk, always include the first two key columns
        chunk_columns = [key_column_1, key_column_2] + df.columns[2+i:2+i+columns_per_chunk].tolist()
        sub_df = df[chunk_columns]
        
        # Apply the df_lagged function to the chunk
        processed_chunk = pivot_lag(sub_df, key_column_1, key_column_2, number_of_shifts)
        
        # Append the processed chunk to the list
        processed_chunks.append(processed_chunk)
    
    # Concatenate all processed chunks
    merged_df = pd.concat(processed_chunks, axis=1)
    
    # Drop duplicate columns that may have been included in multiple chunks
    merged_df = merged_df.loc[:,~merged_df.columns.duplicated()]
    
    return merged_df

# Example usage:
# merged_df = process_in_chunks(large_df, 'key_col_1', 'key_col_2', 3, 5)


def merge_json_data(folder_path, columns):
    """
    This function reads JSON files from the specified folder, extracts the specified keys,
    and merges them into a single pandas DataFrame. Files missing any of these keys are skipped.
    It also prints out the count of files processed by thousands.

    Parameters:
    folder_path (str): The path to the folder containing the JSON files.
    columns (list): A list of column names (keys) to extract from the JSON files.

    Returns:
    pd.DataFrame: A DataFrame containing the merged data.
    """
    data_list = []  # List to hold the data
    file_count = 0  # Counter for the number of files processed

    # Loop through each file in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json'):
            file_count += 1
            file_path = os.path.join(folder_path, file_name)
            try:
                with open(file_path, 'r') as file:
                    # Load the content of the JSON file
                    json_data = json.load(file)
                    # Extract the desired data
                    data = {column: json_data[column] for column in columns}
                    data_list.append(data)
            except (KeyError, json.JSONDecodeError):
                continue  # Skip the file if an error occurs

    # Print the count of files processed by thousands
    print(f"Processed {file_count // 1000 * 1000} files.")

    # Create a DataFrame from the list of data
    return pd.DataFrame(data_list)

# Usage example:
#folder_path = 'companyfacts'  # Replace with your folder path
#columns = ['cik', 'entityName']  # Replace with your desired column names
#df = merge_json_data(folder_path, columns)


def retrieve_and_merge_data(company_ciks, keys_of_interest):
    headers = {'User-Agent': 'YourAppName/1.0 (your-email@example.com)'}
    company_data = []

    for cik in company_ciks:
        url = f'https://data.sec.gov/submissions/{cik}'
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            extracted_data = {key: data.get(key, None) for key in keys_of_interest}
            company_data.append(extracted_data)
        else:
            print(f'Failed to retrieve data for CIK: {cik}')

    return pd.DataFrame(company_data)

# Usage
# company_ciks = list(os.listdir('asd'))  # Replace with actual CIKs
# keys_of_interest = ['cik', 'entityType', 'sic', 'sicDescription', 'name', 'tickers', 'exchanges']
# df = retrieve_and_merge_data(company_ciks, keys_of_interest)

def train_random_forest(df, target_name):
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(target_name, axis=1), df[target_name], test_size=0.3, random_state=42
    )
    rf_classifier = RandomForestClassifier(random_state=42)
    rf_classifier.fit(X_train, y_train)
    y_pred = rf_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, rf_classifier.predict_proba(X_test)[:, 1])
    return rf_classifier, accuracy, roc_auc

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.metrics import accuracy_score, roc_auc_score
# from sklearn.feature_selection import SelectFromModel

# def train_random_forest(df, target_name):
#     # Splitting the dataset
#     X_train, X_test, y_train, y_test = train_test_split(
#         df.drop(target_name, axis=1), df[target_name], test_size=0.3, random_state=42
#     )
    
#     # Feature selection
#     rf_selector = RandomForestClassifier(n_estimators=100, random_state=42)
#     sfm = SelectFromModel(rf_selector)
#     sfm.fit(X_train, y_train)
#     X_train_transformed = sfm.transform(X_train)
#     X_test_transformed = sfm.transform(X_test)
    
#     # Model with tuned parameters
#     rf_classifier = RandomForestClassifier(random_state=42, n_jobs=-1)
#     param_grid = {
#         'n_estimators': [100, 200],
#         'max_depth': [10, 20],
#         'min_samples_split': [2, 5],
#         'min_samples_leaf': [1, 2]
#     }
#     grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=3, n_jobs=-1)
#     grid_search.fit(X_train_transformed, y_train)
    
#     # Best model
#     best_rf_classifier = grid_search.best_estimator_
    
#     # Predictions
#     y_pred = best_rf_classifier.predict(X_test_transformed)
#     accuracy = accuracy_score(y_test, y_pred)
#     roc_auc = roc_auc_score(y_test, best_rf_classifier.predict_proba(X_test_transformed)[:, 1])
    
#     return best_rf_classifier, accuracy, roc_auc


def save_model(model, filename):
    dump(model, filename)

def load_model(filename):
    return load(filename)

def evaluate_model(model, df, target_name):
    y_pred = model.predict(df.drop(target_name, axis=1))
    accuracy = accuracy_score(df[target_name], y_pred)
    roc_auc = roc_auc_score(df[target_name], model.predict_proba(df.drop(target_name, axis=1))[:, 1])
    return accuracy, roc_auc

# Example usage:
# Assuming 'df' is your first part of the dataset and 'target' is the target column name
# rf_model, train_accuracy, train_roc_auc = train_random_forest(df, 'target')
# print(f"Training Accuracy: {train_accuracy}")
# print(f"Training AUC/ROC: {train_roc_auc}")

# save_model(rf_model, 'random_forest_classifier.joblib')

# # In a new session, after loading the model
# rf_model_loaded = load_model('random_forest_classifier.joblib')

# # Assuming 'df2' is your second part of the dataset
# unseen_accuracy, unseen_roc_auc = evaluate_model(rf_model_loaded, df2, 'target')
# print(f"Accuracy on unseen data: {unseen_accuracy}")
# print(f"AUC/ROC on unseen data: {unseen_roc_auc}")