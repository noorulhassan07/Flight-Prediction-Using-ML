import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Sample 50k rows from each CSV
data_2019 = pd.read_csv("../data/Jan_2019_ontime.csv").sample(n=50000, random_state=42)
data_2020 = pd.read_csv("../data/Jan_2020_ontime.csv").sample(n=50000, random_state=42)
data = pd.concat([data_2019, data_2020], ignore_index=True)

# Clean & label
data = data.dropna(subset=["DEP_DEL15", "ORIGIN"])
data["DELAYED"] = data["DEP_DEL15"]

# Make sure output folder exists
output_dir = "../static/images"
os.makedirs(output_dir, exist_ok=True)

# --- Graph 1: Delay Distribution ---
plt.figure(figsize=(8, 5))
sns.countplot(x='DELAYED', data=data)
plt.title('Delay Distribution')
plt.xticks([0, 1], ['On Time', 'Delayed'])
plt.savefig(f"{output_dir}/delay_distribution.png")
plt.close()

# --- Graph 2: Top Delay Origin Airports ---
delay_counts = data[data['DELAYED'] == 1]['ORIGIN'].value_counts().head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=delay_counts.values, y=delay_counts.index)
plt.title('Top 10 Delay Origin Airports')
plt.xlabel('Number of Delayed Flights')
plt.ylabel('Origin Airport')
plt.savefig(f"{output_dir}/top_delay_airports.png")
plt.close()
