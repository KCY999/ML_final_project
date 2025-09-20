import pandas as pd
import json
import numpy as np

data_path = "./raw_data/train_data.csv"
output_path = "./processed_data/pitcher_win_rate.json"

    



def p_count():
    data = pd.read_csv(data_path, usecols=["home_team_win","home_pitcher", "away_pitcher"])
    pitcher_dict = {}
    
    home_pitchers = data["home_pitcher"].to_list()
    away_pitchers = data["away_pitcher"].to_list()
    wins = data["home_team_win"].to_list()

    game_time = { str(pitcher): (home_pitchers+away_pitchers).count(pitcher) for pitcher in set(home_pitchers+away_pitchers)}
    print(game_time)
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
        
        
    print("Dictionary:", pitcher_dict)

    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(pitcher_dict, json_file, ensure_ascii=False, indent=4)
    
    return pitcher_dict


if __name__ == "__main__":
    p_count()