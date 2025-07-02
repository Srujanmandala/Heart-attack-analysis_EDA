# heart_attack_dashboard_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set(style="whitegrid")

# Load dataset
df = pd.read_csv("D:/EDA BEGINNER PROJECT/heart_attack_analysis/heart_attack_youngsters_india.csv")

if "Hypertension" in df.columns:
    hypertension_counts = df["Hypertension"].value_counts()
    hypertension_percent = df["Hypertension"].value_counts(normalize=True) * 100

    # PIE chart for Hypertension presence
    plt.pie(hypertension_percent, labels=hypertension_percent.index, autopct="%.1f%%", startangle=90)
    plt.title("Hypertension Status Distribution")
    plt.axis("equal")
    plt.show()
else:
    print("⚠️ 'Hypertension' column not found in the dataset.")


# Drop duplicates
df.drop_duplicates(inplace=True)

# Feature Engineering

df["Heart Attack"] = df["Heart Attack Likelihood"].apply(lambda x: 1 if x == "Yes" else 0)
df["Age Group"] = pd.cut(df["Age"], bins=[17, 25, 30, 35], labels=["18-25", "26-30", "31-35"])
df["Chol Category"] = pd.cut(df["Cholesterol Levels (mg/dL)"], bins=[0, 200, 240, 500], labels=["Low", "Medium", "High"])
df["Trigly Category"] = pd.cut(df["Triglyceride Levels (mg/dL)"], bins=[0, 150, 200, 1000], labels=["Normal", "Borderline", "High"])

# ==============================
# KPI Calculations
# ==============================
prevalence = df["Heart Attack"].mean() * 100
male_rate = df[df["Gender"] == "Male"]["Heart Attack"].mean() * 100
female_rate = df[df["Gender"] == "Female"]["Heart Attack"].mean() * 100

cholesterol_avg = df["Cholesterol Levels (mg/dL)"].mean()

print(f"Prevalence Rate: {prevalence:.2f}%")
print(f"Male Heart Attack Rate: {male_rate:.2f}%")
print(f"Female Heart Attack Rate: {female_rate:.2f}%")
print(f"Average Cholesterol: {cholesterol_avg:.2f} mg/dL")

# ==============================
# PIE CHARTS
# ==============================

# 1. Heart Attack by Gender
gender_counts = df.groupby("Gender")["Heart Attack"].mean() * 100
gender_counts.plot.pie(autopct="%.1f%%", startangle=90, figsize=(5, 5), title="Heart Attack % by Gender")
plt.ylabel("")
plt.show()

# 2. ECG Result Breakdown
ecg_counts = df["ECG Results"].value_counts(normalize=True) * 100
ecq_labels = ecg_counts.index
plt.pie(ecg_counts, labels=ecq_labels, autopct="%.1f%%", startangle=90)
plt.title("ECG Result Breakdown")
plt.axis("equal")
plt.show()

# 3. Heart Attack by Physical Activity
activity_counts = df.groupby("Physical Activity Level")["Heart Attack"].sum().sort_values(ascending=False)
activity_counts.plot.pie(autopct="%.1f%%", figsize=(6, 6), title="Heart Attack by Physical Activity")
plt.ylabel("")
plt.show()

# 4. Heart Attack by Diet
diet_attack = df.groupby("Diet Type")["Heart Attack"].sum()
diet_attack.plot.pie(autopct="%.1f%%", figsize=(6, 6), title="Heart Attack by Diet Type")
plt.ylabel("")
plt.show()

# ==============================
# BAR CHARTS
# ==============================

# Chest Pain by Chol & Triglyceride
ct_pivot = df.groupby("Chest Pain Type")[["Cholesterol Levels (mg/dL)", "Triglyceride Levels (mg/dL)"]].mean().sort_values(by="Cholesterol Levels (mg/dL)", ascending=False)
ct_pivot.plot(kind="bar", stacked=False, figsize=(10, 5), title="Chest Pain Type by Cholesterol & Triglyceride")
plt.ylabel("mg/dL")
plt.xticks(rotation=30)
plt.show()

# Cholesterol level by Age Group
chol_age = df.groupby("Age Group", observed=True)["Cholesterol Levels (mg/dL)"].mean().sort_values(ascending=False)
chol_age.plot(kind="barh", title="Cholesterol Level by Age Group")
plt.xlabel("Cholesterol (mg/dL)")
plt.show()

# ==============================
# Additional Visuals
# ==============================

# Alcohol / Stress vs Cholesterol
sns.barplot(data=df, x="Alcohol Consumption", y="Cholesterol Levels (mg/dL)", hue="Stress Level")
plt.title("Alcohol & Stress vs Cholesterol")
plt.xticks(rotation=30)
plt.show()

# Age vs Heart Attack Count
sns.histplot(data=df, x="Age", hue="Heart Attack", multiple="stack", bins=10)
plt.title("Heart Attack Count by Age")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# ==============================
# Tabular View by Age
# ==============================data/heart_attack_cleaned.csv
# ==============================
# Tabular View by Age
# ==============================

# Check which required columns are present
expected_cols = ["BMI", "Blood Oxygen Level (%)", "Cholesterol Levels (mg/dL)",
                 "Resting Heart Rate (bpm)", "Triglyceride Levels (mg/dL)"]
present_cols = [col for col in expected_cols if col in df.columns]

if present_cols:
    age_table = df.groupby("Age")[present_cols].mean()
    print("\nTabular Summary by Age:\n")
    print(age_table.head(20).round(2))
else:
    print("⚠️ None of the expected columns are present for tabular view.")

# ==============================
# Diabetes (Yes/No) Pie Chart
# ==============================
if "Diabetes" in df.columns:
    diabetes_counts = df["Diabetes"].value_counts().loc[["Yes", "No"]]
    diabetes_counts.plot.pie(
        autopct="%.1f%%",
        labels=["Diabetic", "Non-Diabetic"],
        colors=["#ff9999", "#66b3ff"],
        startangle=90,
        figsize=(6, 6),
        title="Diabetes Distribution (Yes/No)"
    )
    plt.ylabel("")
    plt.show()
else:
    print("⚠️ 'Diabetes' column not found in the dataset.")



