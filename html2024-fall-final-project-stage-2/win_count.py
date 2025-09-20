import pandas as pd
import json

data_path = "./raw_data/train_data.csv"
output_path = "./processed_data/team_win_rate.json"

def ranking():
    data = pd.read_csv(data_path, usecols=["home_team_win","home_team_abbr", "away_team_abbr"])
    rank_dict = {}
    
    home_teams = data["home_team_abbr"].to_list()
    away_teams = data["away_team_abbr"].to_list()
    wins = data["home_team_win"].to_list()

    game_time = { str(team): (home_teams+away_teams).count(team) for team in set(home_teams+away_teams)}
    print(game_time)
    for home, away, win in zip(home_teams, away_teams, wins):
        try:
            if win:
                rank_dict[home] += 1 / game_time[home]
            else:
                rank_dict[away] += 1 / game_time[away]
        except:
            if win:
                rank_dict[home] = 1 / game_time[home]
            else:
                rank_dict[away] = 1 / game_time[away]
    print("Rank Dictionary:", rank_dict)  # 確認 rank_dict 是否有內容

    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(rank_dict, json_file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    ranking()