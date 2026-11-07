import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import wilcoxon
from xgboost import XGBRegressor

# =====================================================================
# INPUT: membaca dan menyiapkan data
# =====================================================================
df = pd.read_csv("Walmart.csv")
df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
df = df.sort_values(["Store", "Date"]).reset_index(drop=True)

df["Month"] = df["Date"].dt.month
df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
df["Sales_Lag1"] = df.groupby("Store")["Weekly_Sales"].shift(1)
df = df.dropna(subset=["Sales_Lag1"]).reset_index(drop=True)

# Pembagian 80/20 secara kronologis
unique_dates = np.sort(df["Date"].unique())
cutoff = unique_dates[int(len(unique_dates) * 0.8)]
train = df[df["Date"] < cutoff].copy()
test = df[df["Date"] >= cutoff].copy()

TARGET = "Weekly_Sales"
LAG = "Sales_Lag1"
ECON = ["Temperature", "Fuel_Price", "CPI", "Unemployment"]
FEATS_FULL = ["Store", "Holiday_Flag", "Month", "WeekOfYear"] + ECON + [LAG]
FEATS_NOECON = ["Store", "Holiday_Flag", "Month", "WeekOfYear", LAG]
SCENARIOS = {
    "Lengkap + Lag1": FEATS_FULL,
    "Tanpa indikator ekonomi + Lag1": FEATS_NOECON,
}
SEEDS = [1, 2, 3, 4, 5]

print(f"Jumlah minggu unik      : {len(unique_dates)}")
print(f"Tanggal awal data uji   : {pd.Timestamp(cutoff).date()}")
print(f"Baris latih / uji       : {len(train)} / {len(test)} "
      f"({len(train)/len(df):.0%} / {len(test)/len(df):.0%})\n")

print("Pengecekan rentang nilai (latih vs uji):")
for col in ECON:
    tr_min, tr_max = train[col].min(), train[col].max()
    te_min, te_max = test[col].min(), test[col].max()
    outside = ((test[col] < tr_min) | (test[col] > tr_max)).mean()
    print(f"{col:13s} latih [{tr_min:.2f}, {tr_max:.2f}]  "
          f"uji [{te_min:.2f}, {te_max:.2f}]  "
          f"di luar rentang latih: {outside:.1%}")
print()


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


# =====================================================================
# PROCESS 1: eksperimen utama (seed 42) dan model naif
# =====================================================================
rows = [{"Skenario": "-", "Model": "Naif (minggu lalu)", "Seed": "-",
         **metrics(test[TARGET], test[LAG])}]

fitted_42 = {}  # simpan model seed 42 untuk visualisasi dan Wilcoxon
for sc_name, feats in SCENARIOS.items():
    for m_name, model in make_models(42).items():
        model.fit(train[feats], train[TARGET])
        pred = model.predict(test[feats])
        fitted_42[(sc_name, m_name)] = model
        rows.append({"Skenario": sc_name, "Model": m_name, "Seed": 42,
                     **metrics(test[TARGET], pred)})

res = pd.DataFrame(rows)
res.round(4).to_csv("hasil_eksperimen_80_20.csv", index=False)

print(f"Rata-rata penjualan mingguan (data uji): {test[TARGET].mean():,.2f}\n")
print("Hasil eksperimen (seed 42):")
print(res.round(3).to_string(index=False))

# =====================================================================
# PROCESS 2: uji stabilitas dengan 5 seed
# =====================================================================
seed_rows = []
for seed in SEEDS:
    for sc_name, feats in SCENARIOS.items():
        for m_name, model in make_models(seed).items():
            model.fit(train[feats], train[TARGET])
            pred = model.predict(test[feats])
            seed_rows.append({"Seed": seed, "Skenario": sc_name, "Model": m_name,
                              **metrics(test[TARGET], pred)})

seed_df = pd.DataFrame(seed_rows)
seed_df.round(4).to_csv("hasil_per_seed.csv", index=False)
summary = seed_df.groupby(["Skenario", "Model"])[["MAE", "RMSE", "R2"]].agg(["mean", "std"]).round(2)
print("\nRingkasan dari 5 seed:")
print(summary.to_string())

# =====================================================================
# PROCESS 3: uji Wilcoxon (skenario tanpa indikator ekonomi + Lag1, seed 42)
# =====================================================================
y = test[TARGET].values
naive_err = np.abs(y - test[LAG].values)
rf_best = fitted_42[("Tanpa indikator ekonomi + Lag1", "Random Forest")]
xgb_best = fitted_42[("Tanpa indikator ekonomi + Lag1", "XGBoost")]
rf_err = np.abs(y - rf_best.predict(test[FEATS_NOECON]))
xgb_err = np.abs(y - xgb_best.predict(test[FEATS_NOECON]))

print("\nUji Wilcoxon (tanpa indikator ekonomi + Lag1):")
print(f"XGBoost vs naif         : p-value = {wilcoxon(xgb_err, naive_err).pvalue:.4f}")
print(f"Random Forest vs naif   : p-value = {wilcoxon(rf_err, naive_err).pvalue:.4f}")
print(f"XGBoost vs Random Forest: p-value = {wilcoxon(xgb_err, rf_err).pvalue:.4f}")

# =====================================================================
# OUTPUT: visualisasi
# =====================================================================
os.makedirs("gambar", exist_ok=True)
test["Pred_Naif"] = test[LAG]
test["Pred_RF"] = rf_best.predict(test[FEATS_NOECON])
test["Pred_XGB"] = xgb_best.predict(test[FEATS_NOECON])

# Gambar 4.1: penjualan aktual vs prediksi untuk satu toko
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
plt.savefig("gambar/gambar4_1_prediksi_toko1.png", dpi=300)
plt.close()

# Gambar 4.2: batang MAE
mae_values = {
    "Naif (minggu lalu)": mean_absolute_error(y, test["Pred_Naif"]),
    "Random Forest": mean_absolute_error(y, test["Pred_RF"]),
    "XGBoost": mean_absolute_error(y, test["Pred_XGB"]),
}
plt.figure(figsize=(6, 4))
plt.bar(list(mae_values.keys()), list(mae_values.values()),
        color=["gray", "steelblue", "darkorange"])
plt.title("Perbandingan MAE pada Data Uji")
plt.ylabel("MAE (USD)")
plt.tight_layout()
plt.savefig("gambar/gambar4_2_mae.png", dpi=300)
plt.close()

# Gambar 4.3: boxplot error absolut per baris
plt.figure(figsize=(6, 4))
plt.boxplot([naive_err, rf_err, xgb_err])
plt.xticks([1, 2, 3], ["Naif", "Random Forest", "XGBoost"])
plt.title("Sebaran Error Absolut per Model")
plt.ylabel("Error absolut (USD)")
plt.tight_layout()
plt.savefig("gambar/gambar4_3_boxplot_error.png", dpi=300)
plt.close()

# Gambar 4.4: tingkat kepentingan fitur Random Forest
imp = pd.Series(rf_best.feature_importances_, index=FEATS_NOECON).sort_values()
plt.figure(figsize=(6, 4))
imp.plot(kind="barh")
plt.title("Tingkat Kepentingan Fitur (Random Forest)")
plt.xlabel("Tingkat kepentingan")
plt.tight_layout()
plt.savefig("gambar/gambar4_4_kepentingan_fitur.png", dpi=300)
plt.close()

print("\nHasil CSV tersimpan: hasil_eksperimen_80_20.csv dan hasil_per_seed.csv")
print("Empat gambar tersimpan di folder gambar/")