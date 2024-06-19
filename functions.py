
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
import math
from xgboost import XGBClassifier
from keras.models import Sequential
from keras.layers import LSTM, Dense

def everything_loop(directory_path, report, lvl1, lvl2, shifts, chunks, threshold, target, test_size,model_name):
    results_df = pd.DataFrame(columns=['File', 'Train Accuracy', 'Train ROC AUC', 'Unseen Accuracy', 'Unseen ROC AUC', 'Shape'])
    feature_importances_df = pd.DataFrame()
    for i in os.listdir(directory_path):
        if i.endswith('.csv'):
            df = pd.read_csv(os.path.join(directory_path, i))
            df_shape, train_accuracy, train_roc_auc, unseen_accuracy, unseen_roc_auc, feature_importances = everything(df, report, lvl1, lvl2, shifts, chunks, threshold, target, test_size,model_name)
            temp_results_df = pd.DataFrame({
                'File': [i],
                'Train Accuracy': [train_accuracy],
                'Train ROC AUC': [train_roc_auc],
                'Unseen Accuracy': [unseen_accuracy],
                'Unseen ROC AUC': [unseen_roc_auc],
                'Shape': [df_shape]
            })
            results_df = pd.concat([results_df, temp_results_df], ignore_index=True)
            temp_feature_importances_df = pd.DataFrame(feature_importances)
            temp_feature_importances_df['File'] = i
            feature_importances_df = pd.concat([feature_importances_df, temp_feature_importances_df], ignore_index=True)
    return results_df, feature_importances_df

def everything(df,report,lvl1,lvl2,shift,chunk,threshold,target_shift,testsize,model_name):
    if report == 'annual':
        df = chosen_period(df,'years')
    if report == 'quarter':
        df = chosen_period(df,'all')
        df = calculate_missing_quarters(df)
        df = chosen_period(df,'quarters')
    na_maxed = na_max(df,threshold)
    na_maxed = na_maxed.dropna(subset=['NetIncomeLoss'])
    na_maxed = na_maxed.drop(['End',	'EntityName',	'FP',	'Filed',	'Form'],axis=1)
    df_1 = percent_diff(na_maxed)
    merged = process_in_chunks(df_1,lvl1,lvl2,shift,chunk)
    df_targeted=target_dropna(merged,lvl1,lvl2,target_shift)
    df_shuffled = df_targeted.sample(frac=1, random_state=42)
    train_set, test_set = train_test_split(df_shuffled, test_size=testsize, random_state=42)
    df = train_set
    df2 = test_set
    rf_model, train_accuracy, train_roc_auc = train_model(df, 'target',model_name)
    feature_importances = rf_model.feature_importances_
    importances = pd.Series(feature_importances, index=df.columns[:-1])
    sorted_importances = importances.sort_values(ascending=False)
    save_model(rf_model, 'trained_model.joblib')
    rf_model_loaded = load_model('trained_model.joblib')
    unseen_accuracy, unseen_roc_auc = evaluate_model(rf_model_loaded, df2, 'target')
    df_shape = df_info(df)
    return df_shape, train_accuracy, train_roc_auc, unseen_accuracy, unseen_roc_auc, sorted_importances

def calculate_missing_quarters(df):
    for cik in df['CIK'].unique():
        years = df['Frame'].str.extract(r'(CY\d{4})')[0].unique()
        for year in years:
            known_quarters_sum = df[(df['CIK'] == cik) & df['Frame'].str.startswith(year) & 
                                    (df['Frame'].str.len() > 6) & (~df['NetIncomeLoss'].isna())]['NetIncomeLoss'].sum()
            annual_frame = year
            annual_row = df[(df['CIK'] == cik) & (df['Frame'] == annual_frame)]
            if not annual_row.empty:
                annual_value = annual_row['NetIncomeLoss'].values[0]
                missing_count = df[(df['CIK'] == cik) & df['Frame'].str.startswith(year) & 
                                   (df['Frame'].str.len() > 6) & (df['NetIncomeLoss'].isna())].shape[0]
                if missing_count > 1:
                    continue
                for quarter in ['Q1', 'Q2', 'Q3', 'Q4']:
                    quarter_frame = year + quarter
                    quarter_row = df[(df['CIK'] == cik) & (df['Frame'] == quarter_frame)]
                    if not quarter_row.empty and np.isnan(quarter_row['NetIncomeLoss'].values[0]):
                        missing_value = annual_value - known_quarters_sum
                        df.loc[(df['CIK'] == cik) & (df['Frame'] == quarter_frame), 'NetIncomeLoss'] = missing_value
    return df

def cik_to_file(num_list):
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

def convertor_multiple(file_paths):
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
            pass
            #print(f"Key {e} missing in file {path}. Skipping this file.")
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
                        pass
                        #print(f"Key {e} missing in entry for {key}. Skipping this entry.")
        df = pd.DataFrame.from_dict(values, orient='index')
        all_dfs.append(df)
    merged_df = pd.concat(all_dfs, ignore_index=True)
    return merged_df

def chosen_period(df,period):
    quarters = [
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
    years = [
        "CY2012","CY2013","CY2014","CY2015","CY2016",
        "CY2017","CY2018","CY2019","CY2020","CY2021",
        "CY2022","CY2023"
    ]
    if period == 'years':
        chosen = years
    if period == 'quarters':
        chosen = quarters
    if period == 'all':
        chosen = quarters + years
    df = df[df['Frame'].isin(chosen)]
    df = df.sort_values(['CIK','Frame'])
    return df

def percent_diff(df):
    df_2 = df[df.columns[2:]].pct_change(fill_method=None).round(3)
    df_1 = df[df.columns[:2]]
    df = pd.concat([df_1,df_2],axis=1)
    pivoted = pd.pivot_table(df, index=['CIK', 'Frame'])
    rows_to_drop = pivoted.groupby(level=0).head(1).index
    pivoted = pivoted.drop(rows_to_drop)
    pivoted = pivoted.reset_index()
    pivoted = pivoted.fillna(0)
    return pivoted

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

def pivot_lag(df, level_1, level_2, shift):
    shift_periods = [-i for i in range(1, shift + 1)]
    pivoted = pd.pivot_table(df, index=[level_1, level_2])
    new_columns = {}
    for col in pivoted.columns:
        for period in shift_periods:
            new_col_name = f'{col}_{abs(period)}'
            new_columns[new_col_name] = pivoted.groupby(level=level_1)[col].shift(periods=period)
    pivoted = pd.concat([pivoted, pd.DataFrame(new_columns, index=pivoted.index)], axis=1)
    desired_order = sorted(list(pivoted.columns))
    pivoted = pivoted[desired_order]
    pivoted = pivoted.reset_index()
    return pivoted

def target_dropna(df,lvl1,lvl2,last):
    df = pd.pivot_table(df, index=[lvl1,lvl2])
    df['target_lag'] = df.groupby(level=lvl1)[last].shift(-1)
    df['target'] = 0
    df.loc[df[last] < df[df.columns[-2]], 'target'] = 1
    df.loc[df[last] > df[df.columns[-2]], 'target'] = 0
    df = df.drop('target_lag',axis=1)
    df = df.reset_index()
    df = df.drop(lvl1,axis=1)
    df = df.drop(lvl2,axis=1)
    df = df.dropna()
    df = df.replace([np.inf, -np.inf, np.nan], 0).round(2)
    return df


def na_max(df,threshold):
    missing_values = df.isnull().sum()
    missing_percentage = missing_values / len(df)
    columns_to_drop = missing_percentage[missing_percentage > threshold].index
    df = df.drop(columns=columns_to_drop)
    return df

def df_info(df):
    # Calculate the percentage of missing values
    missing_percentage = ((df.isnull().sum().sum()) / (df.shape[0] * df.shape[1])).round(2)
    
    # Get the shape of the DataFrame
    shape = df.shape
    
    return missing_percentage, shape

def train_model(df, target_name, model_name):
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(target_name, axis=1), df[target_name], test_size=0.3, random_state=42
    )
    if model_name == 'RandomForest':
        model = RandomForestClassifier(random_state=42)
    elif model_name == 'XGBoost':
        model = XGBClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    return model, accuracy, roc_auc

def evaluate_model(model, df, target_name):
    y_pred = model.predict(df.drop(target_name, axis=1))
    accuracy = accuracy_score(df[target_name], y_pred)
    roc_auc = roc_auc_score(df[target_name], model.predict_proba(df.drop(target_name, axis=1))[:, 1])
    return accuracy, roc_auc

def save_model(model, filename):
    dump(model, filename)

def load_model(filename):
    return load(filename)

# Example usage:
# df = pd.read_csv('your_data.csv')
# model, accuracy, roc_auc = train_model(df, 'target_column', 'RandomForest')
# save_model(model, 'random_forest_model.joblib')
# loaded_model = load_model('random_forest_model.joblib')
# eval_accuracy, eval_roc_auc = evaluate_model(loaded_model, df, 'target_column')