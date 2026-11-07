import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# INPUT: baca dan siapkan data (sama dengan prediction.py)
df = pd.read_csv("Walmart.csv")
df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
df = df.sort_values(["Store", "Date"]).reset_index(drop=True)
df["Month"] = df["Date"].dt.month
df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
df["Sales_Lag1"] = df.groupby("Store")["Weekly_Sales"].shift(1)
df = df.dropna(subset=["Sales_Lag1"]).reset_index(drop=True)

unique_dates = np.sort(df["Date"].unique())
cutoff = unique_dates[int(len(unique_dates) * 0.8)]
train = df[df["Date"] < cutoff].copy()
test = df[df["Date"] >= cutoff].copy()

FEATS = ["Store", "Holiday_Flag", "Month", "WeekOfYear", "Sales_Lag1"]
TARGET = "Weekly_Sales"

# PROCESS: learning dan prediksi
rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(train[FEATS], train[TARGET])
xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)
xgb.fit(train[FEATS], train[TARGET])

test["Pred_Naif"] = test["Sales_Lag1"]
test["Pred_RF"] = rf.predict(test[FEATS])
test["Pred_XGB"] = xgb.predict(test[FEATS])

# OUTPUT 1: garis aktual vs prediksi untuk satu toko
store_id = 1
t = test[test["Store"] == store_id].sort_values("Date")
plt.figure(figsize=(10, 4))
plt.plot(t["Date"], t[TARGET], label="Aktual", color="black", linewidth=2)
plt.plot(t["Date"], t["Pred_Naif"], label="Naif", linestyle="--")
plt.plot(t["Date"], t["Pred_RF"], label="Random Forest")
plt.plot(t["Date"], t["Pred_XGB"], label="XGBoost")
plt.title(f"Penjualan Mingguan Aktual vs Prediksi (Toko {store_id}, Data Uji)")
plt.xlabel("Tanggal")
plt.ylabel("Penjualan mingguan (USD)")
plt.legend()
plt.tight_layout()
plt.savefig("gambar1_prediksi_toko1.png", dpi=300)
plt.close()

# OUTPUT 2: batang MAE dari file hasil eksperimen
res = pd.read_csv("hasil_eksperimen_80_20.csv")
sub = res[res["Model"].isin(["Naif (minggu lalu)", "Random Forest", "XGBoost"])
          & res["Skenario"].isin(["-", "Tanpa indikator ekonomi + Lag1"])]
plt.figure(figsize=(6, 4))
plt.bar(sub["Model"], sub["MAE"], color=["gray", "steelblue", "darkorange"])
plt.title("Perbandingan MAE pada Data Uji")
plt.ylabel("MAE (USD)")
plt.tight_layout()
plt.savefig("gambar2_mae.png", dpi=300)
plt.close()

# OUTPUT 3: boxplot error absolut per baris data uji
errors = {
    "Naif": (test[TARGET] - test["Pred_Naif"]).abs(),
    "Random Forest": (test[TARGET] - test["Pred_RF"]).abs(),
    "XGBoost": (test[TARGET] - test["Pred_XGB"]).abs(),
}
plt.figure(figsize=(6, 4))
plt.boxplot(list(errors.values()))
plt.xticks(range(1, len(errors) + 1), list(errors.keys()))
plt.title("Sebaran Error Absolut per Model")
plt.ylabel("Error absolut (USD)")
plt.tight_layout()
plt.savefig("gambar3_boxplot_error.png", dpi=300)
plt.close()

# OUTPUT 4: tingkat kepentingan fitur Random Forest
imp = pd.Series(rf.feature_importances_, index=FEATS).sort_values()
plt.figure(figsize=(6, 4))
imp.plot(kind="barh")
plt.title("Tingkat Kepentingan Fitur (Random Forest)")
plt.xlabel("Tingkat kepentingan")
plt.tight_layout()
plt.savefig("gambar4_kepentingan_fitur.png", dpi=300)
plt.close()

print("Empat gambar tersimpan.")