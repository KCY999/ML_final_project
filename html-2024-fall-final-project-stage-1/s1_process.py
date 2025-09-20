from data_process import *
import pandas as pd

# 可能再搞
def year_proc(data):
    data = pd.get_dummies(data, columns=["season"], prefix="", prefix_sep="")
    return data

def s1_data_proc(name="train", raw_data_path="./raw_data/train_data.csv"):
    raw = pd.read_csv(raw_data_path)
    d = remove_data_with_missing_data(raw, 0.8) if name == "train" else raw
    d = team_win_proc(d)
    # d = pitcher_win_proce(d)
    d = remove_attr(d, DROP_ATTRs_s1)
    d = year_proc(d)
    # d = fill_empty_by_median(d)
    d = fill_empty_by_mean(d)
    d = bool_to_dig(d)
    if name == "train":
        d = trans_home_team_win(d)
    # d = ratio_transform(d)
    d = diff_transform(d)
    # d = standardization(d)
    d.to_csv(f'./processed_data/_processed_{name}.csv', index=False)


if __name__ == '__main__':
    s1_data = s1_data_proc()
    s1_data = s1_data_proc("test", "./raw_data/same_season_test_data.csv")

    # s1_data.to_csv('s1_data.csv', index=False)
    # raw = pd.read_csv("./raw_data/train_data.csv")
    # d = remove_data_with_missing_data(raw, 0.8)
    # d = remove_attr(d, DROP_ATTRs_s1)
    # d = year_proc(d)
    # testcsv(d)

    