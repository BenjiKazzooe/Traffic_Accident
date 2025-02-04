# Traffic Accident Analysis

## 🚀 Introduction
Every day, thousands of traffic accidents occur across the US. But **where and when do they happen most?** Can we use data to **predict and prevent accidents?**

In this project, we analyze a dataset containing **7.7 million traffic accidents** recorded from 2016 to 2023. Using **data wrangling, visualization, and SQL**, we uncover insights about accident severity, locations, and trends.

---

## 📥 Data Collection & Data Wrangling
### **📌 Where does the data come from?**
- The dataset comes from **US Accidents Dataset (2016-2023)**, which contains traffic accident reports collected via APIs and sensors.
- **Size**: 7.7 million records.
- **Features**: Accident location, time, severity, weather conditions, etc.

### **🔧 Data Processing Steps**
1. **Loading Data**: Read the dataset into a Pandas DataFrame.
2. **Data Cleaning**: Handle missing values using forward-fill.
3. **Feature Engineering**: Extract date-time features (Year, Month, Hour).
4. **Storing Data**: Save into an **SQLite database** for efficient queries.

```python
import pandas as pd
import sqlite3

# Load dataset
df = pd.read_csv("US_Accidents.csv")

# Store in SQLite database
conn = sqlite3.connect("accidents.db")
df.to_sql("accidents", conn, if_exists="replace", index=False)
```

---

## 📊 Exploratory Data Analysis (EDA)
### **1️⃣ How Severe Are Most Accidents?**
Understanding accident severity helps prioritize safety measures.

```python
query = "SELECT Severity, COUNT(*) as count FROM accidents GROUP BY Severity ORDER BY count DESC"
sql_result = pd.read_sql(query, conn)
print(sql_result)
```

### **🔍 Key Observations:**
✔️ Most accidents fall into **Severity Level 2** (moderate).  
✔️ **Severe accidents (Level 4) are rare** but highly impactful.

#### **📈 Visualization**
```python
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
sns.barplot(x="Severity", y="count", data=sql_result, palette='mako')
plt.xlabel("Accident Severity")
plt.ylabel("Number of Accidents")
plt.title("Distribution of Accident Severity")
plt.show()
```

![Accident Severity Chart](accident_severity.png)

---

## 🌎 Heatmap: Where Do Most Accidents Happen?
Accidents aren't evenly distributed. Let's visualize accident hotspots.

```python
import folium
from folium.plugins import HeatMap

# Sample 50,000 rows for faster rendering
df_sample = df.sample(50000, random_state=42)

# Create heatmap
map_center = [df_sample['Start_Lat'].mean(), df_sample['Start_Lng'].mean()]
accident_map = folium.Map(location=map_center, zoom_start=5)
heat_data = df_sample[['Start_Lat', 'Start_Lng']].values.tolist()
HeatMap(heat_data).add_to(accident_map)
accident_map.save("accident_heatmap.html")
```

📌 **Findings:**
- **Major cities** (e.g., Los Angeles, New York) show the highest accident density.
- Some **rural highways** also have high accident rates.

🔗 **[View Interactive Heatmap](accident_heatmap.html)**

---

## 🎯 Key Takeaways & Next Steps
✔️ **Most accidents are of moderate severity, but severe accidents exist.**  
✔️ **Accidents are clustered in major cities and highways.**  
✔️ **Using this data, we can identify high-risk areas and improve road safety.**  

### 🚀 What’s Next?
🔹 **Predicting accidents using Machine Learning** 📊  
🔹 **Building a real-time dashboard with Plotly Dash** 🖥️  
🔹 **Analyzing weather & time patterns to predict accident risks** ☁️  

Would you like to contribute or suggest further improvements? **Open an issue or pull request!** 🚀

---

## 📜 References
- Dataset: [US Accidents on Kaggle](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)
- Libraries: Pandas, SQLite, Seaborn, Folium, Matplotlib

---

📌 **Author:** _Your Name_ | 🏷️ **Project License:** MIT

---
