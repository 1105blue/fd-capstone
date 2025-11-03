# ✈️ Flight Delays Capstone – Predicting U.S. Airline Delays (fd-capstone)

This project explores airline delay patterns using data from the Bureau of Transportation Statistics (BTS).  
It was developed as part of the **WGU Data Analytics Capstone (D502/BHN1)** and walks through the full data analytics workflow — cleaning and preparing data, building models, evaluating results, and visualizing findings.

---

## 📁 Folder Structure

```

C:\fd-capstone
│
├── data
│   └── raw
│       ├── Airline_Delay_Cause.csv
│       ├── Download_Column_Definitions.xlsx
│       └── On_Time_Reporting_2023_01.csv
│
├── notebooks
│   ├── 01_quick_check.ipynb        # initial data validation and setup
│   ├── 02_modeling.ipynb           # optional exploratory modeling
│   └── 03_visuals.ipynb            # final visual generation (Task 3)
│
├── scripts
│   ├── train_model.py              # trains LinearRegression + RandomForest
│   └── evaluate_model.py           # reloads saved model and computes test metrics
│
├── outputs
│   ├── charts
│   │   ├── delay_distribution.png
│   │   ├── avg_delay_by_month.png
│   │   └── pred_vs_actual.png
│   └── model
│       ├── fd_model.pkl
│       └── metrics.json
│
└── README.md

````

---

## ⚙️ Setup & Environment

1. Open **C:\fd-capstone** in VS Code.  
2. Create and activate your virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   python -m pip install --upgrade pip
   pip install pandas numpy scikit-learn matplotlib seaborn jupyter
````

3. Place your BTS dataset at:

   ```
   data\raw\Airline_Delay_Cause.csv
   ```

---

## 🚀 Running the Project

### Step 1: Train the Model

Run the training script:

```bash
python scripts/train_model.py --data "data/raw/Airline_Delay_Cause.csv"
```

**Outputs created:**

* `outputs/model/fd_model.pkl`
* `outputs/model/metrics.json`
* Prints R², RMSE, and MAE for LinearRegression and RandomForest models.

### Step 2: Evaluate and Plot

```bash
python scripts/evaluate_model.py --data "data/raw/Airline_Delay_Cause.csv"
```

**Outputs created:**

* `outputs/charts/pred_vs_actual.png`
* Confirms metrics for the saved model.

---

## 📊 Results Summary

| Model                 | R²     | RMSE  | MAE   |
| --------------------- | ------ | ----- | ----- |
| **Linear Regression** | -0.157 | 97.79 | 69.24 |
| **Random Forest**     | -0.205 | 99.77 | 46.52 |

**Interpretation:**
The results show that while both models executed successfully, the aggregated dataset offers limited predictive power for flight delays at the monthly level.
The Linear Regression model performed slightly better in R² but still fell below zero, suggesting that more granular, flight-level data would be needed for stronger performance.
This baseline confirms the workflow — data preprocessing, encoding, and training — worked correctly and is ready for future enhancement with additional variables such as weather, route patterns, and carrier performance history.

---

## 🖼️ Visuals

The following charts were produced for analysis and reporting:

* **Figure 1:** Delay Distribution (Minutes)
* **Figure 2:** Average Delay by Month
* **Figure 3:** Predicted vs. Actual Delay

All visuals are saved in `outputs/charts` and referenced in the Task 3 capstone report.

---

## 📚 References

Bureau of Transportation Statistics (BTS). (2025). *Airline On-Time Performance Data.*
[https://www.transtats.bts.gov/OT_Delay/OT_DelayCause1.asp](https://www.transtats.bts.gov/OT_Delay/OT_DelayCause1.asp)

Federal Aviation Administration (FAA). (2023). *Seasonal Impacts on Flight Operations.* FAA Technical Report Series.

Scikit-learn Developers. (2024). *Linear Regression User Guide.*
[https://scikit-learn.org/stable/modules/linear_model.html](https://scikit-learn.org/stable/modules/linear_model.html)
