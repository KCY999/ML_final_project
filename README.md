# 113-1 TAICA: Machine Learning Final Project

_本專案僅包含本人與組員自行撰寫的程式碼與報告，未公開任何課程講義或教師提供之資源。僅作為個人115推甄備審資料所用，如有疑慮請與我聯絡 (Email: b1126006@cgu.edu.tw)。_

## 簡介
- **開課學校**：國立台灣大學
- **教師**：林軒田 教授
- **課程屬性**：研究所課程
- **授課語言**：英文
- **學分學程**：人工智慧視覺技術學分學程(機器學習)

## 要求
本專案為 **NTU 機器學習 (Fall 2024)** 的期末專案，主要目標為：
- 針對 **HTMLB (Hyper Thrill Machine Learning Baseball)** 平台的比賽數據，進行 **比賽勝負預測**。
- 任務分為兩階段：
  - **Stage 1**：預測 2016–2023 年 8–12 月的比賽結果。
  - **Stage 2**：預測 2024 年所有比賽結果。
- **任務類型**：二元分類問題（主場球隊是否獲勝）。
- **評估方式**：0/1 Error (Accuracy)。
- **報告要求**：
  - 至少比較四種機器學習模型。
  - 分析準確率、穩定性、效率、可解釋性。
  - 提供最佳模型推薦並解釋優缺點。
- 限制：
  - 嚴禁使用外部資料來源。
  - 最多四人一組，需在報告中說明分工。
  - 報告需具備重現性，包含資料前處理、模型設計、實驗設定與參數。

---

## 實作內容

### Data Preprocessing
1. **移除缺失值過高欄位**：刪除缺失率超過 80% 的欄位。  
2. **年份處理**：  
   - Stage 1：One-Hot Encoding。  
   - Stage 2：移除年份欄位。  
3. **缺失值補齊**：以各欄位平均值填補。  
4. **對稱屬性差異化**：計算主場 − 客場屬性差異，降低冗餘。  
5. **新增特徵**：計算球隊歷史勝率，幫助模型理解球隊實力。

### Models
- **Logistic Regression**  
  - 線性基準模型，搭配 L1/L2 正則化與 Grid Search。
  - 優點：計算效率高。  
  - 缺點：無法捕捉非線性結構，易 underfitting。

- **Support Vector Machine (SVM)**  
  - 測試 Linear、Polynomial、RBF kernel。
  - 優點：具備非線性擬合能力。  
  - 缺點：對雜訊敏感，計算成本高。

- **AdaBoost Decision Tree**  
  - 弱分類器加權集成，調整 `n_estimators`、`max_depth`、`learning_rate`。
  - 優點：能聚焦難分類樣本。  
  - 缺點：過度強調雜訊，計算效率低。

- **Random Forest**  
  - 使用 Bagging + 多棵決策樹隨機子集，設定 `n_estimators=200`。  
  - 優點：對雜訊具韌性、表現穩定，計算成本合理。  
  - 缺點：仍需訓練多棵樹，資源消耗高於單一模型。

### Workload
- 蘇茂傑：資料前處理、Logistic Regression  
- 楊佩潔：Support Vector Machine  
- 郭兆揚：Adaboost Decision Tree、Random Forest  

---

## 結果

| Model                  | Stage 1 Accuracy | Stage 2 Accuracy | Notes                                |
|------------------------|------------------|------------------|--------------------------------------|
| Logistic Regression    | 0.574            | 0.569            | 計算最快，但易 underfitting           |
| SVM                    | 0.581            | 0.545             | 能處理非線性，但受雜訊影響大           |
| AdaBoost Decision Tree | 0.582            | 0.583            | 成效尚可，但過度受雜訊干擾             |
| Random Forest          | **0.597**        | **0.592**        | 表現最佳，穩定性與效率兼具             |

### Final Recommendation
- **最佳模型：Random Forest**
  - **優點**：最高準確率 (~59%)、抗雜訊、穩定性佳、效率中等。  
  - **缺點**：需訓練多棵決策樹，計算資源需求較高。  
  - **整體結論**：在準確率與效率之間達到良好平衡，為本專案最佳選擇。  


## Kaggle 競賽結果
| Competition            | Score            | Rank           | 
|------------------------|------------------|----------------|
| Stage 1 Public         | 0.60038          | 20 / 141       | 
| Stage 1 Private        | 0.58665          | 39 / 141       | 
| Stage 2 Public         | 0.59800          | 21 / 130       | 
| Stage 2 Private        | 0.54575          | 53 / 130       | 

<img src="Figs\stage1-pub.png" alt="stage1 public" width="900"/> 

<img src="Figs\stage1-pri.png" alt="stage1 private" width="900"/> 

<img src="Figs\stage2-pub.png" alt="stage2 public" width="900"/> 

<img src="Figs\stage2-pri.png" alt="stage2 public" width="900"/> 

