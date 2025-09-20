import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import json
from sklearn.model_selection import train_test_split


testcsv = lambda f: f.to_csv('__test.csv', index=False)
data = pd.read_csv("./processed_data/processed_train.csv")

# raw data
# TRAIN_DATA_PATH: "./raw_data/train_data.csv"
# s1 TEST_DATA_PATH: "./raw_data/same_season_test_data.csv"

# DROP_ATTRs_s1 = ['id','date', 'home_team_season', 'away_team_season',   'home_team_abbr', 'away_team_abbr', 'home_pitcher', 'away_pitcher']
# DROP_ATTRs_s1 = ['id','date', 'home_team_season', 'away_team_season',  'home_pitcher', 'away_pitcher']
DROP_ATTRs_s1 = ['id','date', 'home_team_season', 'away_team_season']

# remove data with: missing ration > tolerate_missing_ratio
def remove_data_with_missing_data(data, tolerate_missing_ratio=0.5):
    null_count_rows = data.isnull().sum(axis=1)
    processed_data = data[null_count_rows <= tolerate_missing_ratio * data.shape[1]] 
    print(processed_data.shape)
    # testcsv(processed_data)
    return processed_data

def remove_attr(data, drop_attrs):
    drop_cols = []
    for drop_attr in drop_attrs:
        if drop_attr in data.columns:
            drop_cols.append(drop_attr)
    processed_data = data.drop(columns=drop_cols)
    # print(processed_data.shape)
    # testcsv(processed_data)
    return processed_data


def team_win_proc(data):
    with open('./processed_data/team_win_rate.json', "r", encoding="utf-8") as json_file:
        team_win_rate = json.load(json_file)
    # 使用 map 映射勝率
    data['h_win_rate'] = data['home_team_abbr'].apply(lambda x: team_win_rate.get(x,0.5))
    data['a_win_rate'] = data['away_team_abbr'].apply(lambda x: team_win_rate.get(x,0.5))
    data["home_away_win_diff"] = data["h_win_rate"] - data["a_win_rate"]
    
    data.drop(columns=['home_team_abbr', 'away_team_abbr'], inplace=True)
    return data
    

def pitcher_win_proce(data, pitcher_win_rate=None):
    if pitcher_win_rate:
        pass
    else:
        with open('./processed_data/team_win_rate.json', "r", encoding="utf-8") as json_file:
            pitcher_win_rate = json.load(json_file)

    data['hp_win_rate'] = data['home_pitcher'].apply(lambda x: pitcher_win_rate.get(x,0.5))
    data['ap_win_rate'] = data['away_pitcher'].apply(lambda x: pitcher_win_rate.get(x,0.5))
    data["hp_ap_win_diff"] = data["hp_win_rate"] - data["ap_win_rate"]
    
    data = data.drop(columns=['home_pitcher', 'away_pitcher'])
    return data



def bool_to_dig(data):
    bool_columns = data.select_dtypes(include=['bool']).columns
    data[bool_columns] = data[bool_columns].astype(int)
    return data

def trans_home_team_win(data):
    data["home_team_win"] = data["home_team_win"].apply(lambda x: 1 if x else -1)
    return data


def fill_empty_by_median(data):
    # 處理數值型欄位
    numeric_columns = data.select_dtypes(include=["number"]).columns
    for col in numeric_columns:
        median_value = data[col].median()
        data[col] = data[col].fillna(median_value).infer_objects(copy=False)

    # 處理布林值欄位
    bool_columns = [col for col in data.columns if data[col].dtype == 'bool' or data[col].dtype == 'object']
    for col in bool_columns:
        try:
            mode_value = data[col].mode()[0]
            data[col] = data[col].fillna(mode_value).infer_objects(copy=False)
        except Exception as e:
            print(f"Error processing column {col}: {e}")
            continue
    return data

def fill_empty_by_mean(data):
    # 處理數值型欄位
    numeric_columns = data.select_dtypes(include=["number"]).columns
    for col in numeric_columns:
        median_value = data[col].mean()
        data[col] = data[col].fillna(median_value).infer_objects(copy=False)

    # 處理布林值欄位
    bool_columns = [col for col in data.columns if data[col].dtype == 'bool' or data[col].dtype == 'object' and col not in ["home_pitcher", "away_pitcher"] ]
    for col in bool_columns:
        try:
            mode_value = data[col].mode()[0]
            data[col] = data[col].fillna(mode_value).infer_objects(copy=False)
        except Exception as e:
            print(f"Error processing column {col}: {e}")
            continue
    return data



# get ratio
def ratio_transform(data):
    for col in data.columns:
        corsp_away = "away"+col[4:]
        if col[:4] == "home" and ( corsp_away in data.columns) and corsp_away != "away_pitcher":
            data[f"{col[5:]}_ratio"] = np.where(
                data[corsp_away] == 0,  # 條件：分母為 0
                np.nan,                 # 分母為 0，結果設為 NaN
                data[col] / data[corsp_away]  # 否則進行正常除法
            )
            data.drop(columns=[col, corsp_away], inplace=True)
    data = fill_empty_by_median(data)    
    return data

def diff_transform(data):
    for col in data.columns:
        corsp_away = "away"+col[4:]
        if col[:4] == "home" and ( corsp_away in data.columns) and corsp_away != "away_pitcher":
            data[f"{col[5:]}_diff"] = data[col] - data[corsp_away]
            data.drop(columns=[col, corsp_away], inplace=True)
    data = fill_empty_by_median(data)    
    return data

def p_count(data):
    pitcher_dict = {}
    
    home_pitchers = data["home_pitcher"].to_list()
    away_pitchers = data["away_pitcher"].to_list()
    wins = data["home_team_win"].to_list()

    game_time = { str(pitcher): (home_pitchers+away_pitchers).count(pitcher) for pitcher in set(home_pitchers+away_pitchers)}
    # print(game_time)
    for home, away, win in zip(home_pitchers, away_pitchers, wins):
        if pd.notna(home) and pd.notna(away):  # Both pitchers are present
            if win:
                pitcher_dict[home] = pitcher_dict.get(home, 0) + 1 / game_time[home]
            else:
                pitcher_dict[away] = pitcher_dict.get(away, 0) + 1 / game_time[away]
        elif pd.isna(home) and pd.notna(away):  # Only away pitcher is present
            if not win:
                pitcher_dict[away] = pitcher_dict.get(away, 0) + 1 / game_time[away]
        elif pd.notna(home) and pd.isna(away):  # Only home pitcher is present
            if win:
                pitcher_dict[home] = pitcher_dict.get(home, 0) + 1 / game_time[home]          
    
    return pitcher_dict



# standarlize
def standardization(data):
    # 確定需要標準化的列
    columns_to_scale = [col for col in data.columns if col not in ["home_team_win", "is_night_game"] + [  str(y)+".0" for y in range(2016,2024)]]
    
    # 建立 StandardScaler
    scaler = StandardScaler()
    
    # 複製數據以保留原始數據的非標準化列
    data_scaled = data.copy()
    
    # 對指定列進行標準化
    data_scaled[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])
    
    return data_scaled


def get_y_and_x(data):
    y = data['home_team_win']
    x = data.drop(columns=['home_team_win'])
    return y, x

def gen_submission_csv(y_preds, submit_name="submission.csv"):
    dict = {
        "id": [],
        "home_team_win": []
    }
    id = 0
    for y_pred in y_preds:
        dict['id'].append(id)
        dict['home_team_win'].append( "True" if y_pred>=0 else "False")
        id += 1
    df = pd.DataFrame(dict)
    df.to_csv(submit_name, index=False)

def pb_train_test_split(data):
    y = data['home_team_win']
    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2)
    p_win_rate = p_count(X_train)
    p_X_train = pitcher_win_proce(X_train, p_win_rate)
    p_X_test = pitcher_win_proce(X_test, p_win_rate)
    p_X_train = p_X_train.drop(columns=['home_team_win'])
    p_X_test = p_X_test.drop(columns=['home_team_win'])
    return p_X_train, p_X_test, y_train, y_test

if __name__ == "__main__":

    # data = pd.read_csv("./processed_data/processed_train.csv")
    # data = ratio_transform(data)

    # testcsv(data)
    
    pass