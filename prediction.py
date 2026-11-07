import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import wilcoxon
from xgboost import XGBRegressor

# 1. Baca dan siapkan data
df = pd.read_csv("Walmart.csv")
df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
df = df.sort_values(["Store", "Date"]).reset_index(drop=True)
df["Month"] = df["Date"].dt.month
df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
df["Sales_Lag1"] = df.groupby("Store")["Weekly_Sales"].shift(1)
df = df.dropna(subset=["Sales_Lag1"]).reset_index(drop=True)

# 2. Pembagian 80/20 secara kronologis
unique_dates = np.sort(df["Date"].unique())
cutoff = unique_dates[int(len(unique_dates) * 0.8)]
train = df[df["Date"] < cutoff].copy()
test = df[df["Date"] >= cutoff].copy()

TARGET = "Weekly_Sales"
LAG = "Sales_Lag1"
ECON = ["Temperature", "Fuel_Price", "CPI", "Unemployment"]
BASE = ["Store", "Holiday_Flag", "Month", "WeekOfYear", LAG]
SCENARIOS = {
    "Lengkap + Lag1": BASE + ECON,
    "Tanpa indikator ekonomi + Lag1": BASE,
}
SEEDS = [1, 2, 3, 4, 5]

def make_models(seed):
    return {
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=seed),
    }

def metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }

# 3. Model naif (tidak memakai seed, jadi hanya satu baris)
rows = [{"Skenario": "-", "Model": "Naif (minggu lalu)", "Seed": "-", **metrics(test[TARGET], test[LAG])}]

# 4. Jalankan setiap model, skenario, dan seed
for seed in SEEDS:
    for sc_name, feats in SCENARIOS.items():
        for m_name, model in make_models(seed).items():
            model.fit(train[feats], train[TARGET])
            pred = model.predict(test[feats])
            rows.append({"Skenario": sc_name, "Model": m_name, "Seed": seed,
                         **metrics(test[TARGET], pred)})

res = pd.DataFrame(rows)
res.round(4).to_csv("hasil_per_seed.csv", index=False)

# 5. Ringkasan: rata-rata dan standar deviasi dari 5 seed
seed_res = res[res["Seed"] != "-"]
summary = seed_res.groupby(["Skenario", "Model"])[["MAE", "RMSE", "R2"]].agg(["mean", "std"]).round(2)
print(f"Rata-rata penjualan mingguan (data uji): {test[TARGET].mean():,.2f}\n")
print("Model naif:")
print(res[res["Model"] == "Naif (minggu lalu)"][["MAE", "RMSE", "R2"]].round(2).to_string(index=False))
print("\nRingkasan dari 5 seed:")
print(summary.to_string())

# 6. Uji Wilcoxon memakai error absolut per baris (seed 42)
def abs_error(feats, model):
    model.fit(train[feats], train[TARGET])
    return np.abs(test[TARGET].values - model.predict(test[feats]))

naive_err = np.abs(test[TARGET].values - test[LAG].values)
feats = BASE
xgb_err = abs_error(feats, XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42))
rf_err = abs_error(feats, RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))
p_rf_naive = wilcoxon(rf_err, naive_err).pvalue

print("\nWilcoxon (skenario tanpa indikator ekonomi + Lag1):")
print(f"XGBoost vs naif         : p-value = {wilcoxon(xgb_err, naive_err).pvalue:.10f}")
print(f"XGBoost vs Random Forest: p-value = {wilcoxon(xgb_err, rf_err).pvalue:.10f}")
print(f"\nMAE Random Forest (seed 42): {rf_err.mean():,.2f}")
print(f"MAE naif                   : {naive_err.mean():,.2f}")
print(f"Wilcoxon Random Forest vs naif: p-value = {p_rf_naive:.10f}")