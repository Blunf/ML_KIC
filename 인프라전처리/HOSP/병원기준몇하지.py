import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("HOSP.csv")

radius_cols = [col for col in df.columns if col.endswith('m')]
df = df[radius_cols]

mean_counts = df.mean()
mean_counts.index = [int(col.replace("m", "")) for col in mean_counts.index]  # make x-axis numeric

# 4. Compute incremental change (1st derivative)
growth = mean_counts.diff().fillna(0)

# 5. Create summary DataFrame
summary_df = pd.DataFrame({
    "Radius(m)": mean_counts.index,
    "AverageHospitalCount": mean_counts.values,
    "Increment": growth.values
})

# 6. Plot
plt.figure(figsize=(12, 5))

# Left: Average hospital count
plt.subplot(1, 2, 1)
plt.plot(summary_df["Radius(m)"], summary_df["AverageHospitalCount"], marker='o')
plt.title("Average Number of Hospitals by Radius")
plt.xlabel("Radius (m)")
plt.ylabel("Average Hospital Count")
plt.grid(True)

# Right: Incremental change
plt.subplot(1, 2, 2)
plt.plot(summary_df["Radius(m)"], summary_df["Increment"], marker='o', color='orange')
plt.title("Increment per Radius Step")
plt.xlabel("Radius (m)")
plt.ylabel("Increase in Count")
plt.grid(True)

plt.tight_layout()
plt.show()

# 7. Print summary table
print("\n📊 Summary of Hospital Accessibility by Radius:")
print(summary_df)
