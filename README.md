# 🚦 Traffic Accident Analysis & Prediction

## 📌 Overview
This project explores **traffic accident trends** and develops a **machine learning model** to predict accident severity. By leveraging **data science, SQL, machine learning, and interactive visual analytics**, we provide valuable insights into accident patterns and **help optimize emergency response and traffic safety**.

##  Key Features & Methodology
 **Exploratory Data Analysis (EDA)** using **SQL & visualizations**
 **Interactive Heatmaps** to identify **accident hotspots**
 **Machine Learning Model** to **predict accident severity (92% accuracy)**
 **Balanced dataset using SMOTE** to improve predictions
 **Interactive Dashboard with Plotly Dash** for real-time analysis
 **Feature Importance Analysis** for key accident factors

---

# 📂 Data Collection & Preprocessing

##  **Dataset**
- **Source:** [US Accidents Dataset (Kaggle)](https://www.kaggle.com/sobhanmoosavi/us-accidents)
- **Size:** 50,000 accident records
- **Features:** Weather, visibility, time of day, location, severity, etc.

## **SQL-Based Data Retrieval**
To extract relevant data from our **SQLite database**, we used SQL queries:
```sql
SELECT Severity, Start_Lat, Start_Lng, `Temperature(F)`, `Visibility(mi)`, `Precipitation(in)`, Weather_Condition 
FROM accidents LIMIT 50000;
```
🔹 **Why?** Extracting **only relevant columns** improves efficiency.

---

# 🔍 Exploratory Data Analysis (EDA)
## **Accident Severity Distribution**
```python
sns.countplot(x=df["Severity"], palette="coolwarm")
plt.title("Accident Severity Distribution")
plt.show()
```
 **Finding:** Most accidents are of **moderate severity (2-3)**, with fewer extreme cases.

## 🌍 **Accident Hotspots (Heatmap)**
We visualized accident **hotspots using Folium**:
```python
m = folium.Map(location=[37.0902, -95.7129], zoom_start=5)
HeatMap(df[['Start_Lat', 'Start_Lng']].values, radius=8).add_to(m)
m.save("heatmap.html")
```
**Finding:** High accident density in **urban areas & highways**.

---

# 🤖 Machine Learning Model: Predicting Accident Severity
## ** Model Overview**
We trained a **Random Forest Classifier** to predict accident severity based on weather conditions, location, and road conditions.

### **📌 Handling Class Imbalance with SMOTE**
Before:
```python
print(y.value_counts())
```
| Severity | Count |
|----------|--------|
| 2        | 29,929 |
| 3        | 19,999 |
| 1        | 56 |
| 4        | 16 |

After **SMOTE Balancing**:
```python
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
```
| Severity | Count |
|----------|--------|
| 1        | 29,929 |
| 2        | 29,929 |
| 3        | 29,929 |
| 4        | 29,929 |

 **Balanced dataset → More reliable predictions**

### **📌 Model Training & Evaluation**
```python
clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
```
✅ **Final Accuracy: 92%** 🎯

### **📊 Classification Report**
| Severity | Precision | Recall | F1-score |
|----------|-----------|--------|----------|
| 1 | 0.99 | 1.00 | 0.99 |
| 2 | 0.87 | 0.82 | 0.84 |
| 3 | 0.83 | 0.87 | 0.85 |
| 4 | 1.00 | 1.00 | 1.00 |

✅ **Perfect recall for Severity 4 = Critical accidents detected accurately** 🚑

---

# **Interactive Dashboard with Plotly Dash**
We built a **web-based interactive dashboard** to visualize accident data dynamically.

## ** Dashboard Features**
✔ **Dropdown filter**: Select accident severity levels.
✔ **Time Series Analysis**: Trends in accident occurrences.
✔ **Map Visualization**: Heatmap of accident hotspots.

```python
app = dash.Dash(__name__)
app.layout = html.Div([
    dcc.Dropdown(id='severity-dropdown', options=[{'label': str(sev), 'value': sev} for sev in sorted(df['Severity'].unique())], value=df['Severity'].unique()[0]),
    dcc.Graph(id='accidents-map')
])
```
 **Run the dashboard:**
```bash
python app.py
```
 **Access via:** `http://127.0.0.1:8050`

---

# 📌 Conclusion: The Role of Data in Traffic Safety
🚦 **Data-driven insights are revolutionizing road safety**. This project shows how **machine learning & analytics** can transform accident data into actionable insights, helping **emergency responders, city planners, and drivers**.

### ** Key Achievements**
✅ **Identified accident hotspots using heatmaps**
✅ **Developed an AI model with 92% accuracy for severity prediction**
✅ **Balanced dataset using SMOTE for fairer predictions**
✅ **Built an interactive dashboard for real-time analysis**
✅ **Improved emergency response predictions using AI**

### ** Future Enhancements**
📌 **Incorporate live traffic & weather data for real-time predictions**
📌 **Use deep learning for more advanced severity classification**
📌 **Deploy the model as a web-based traffic safety tool**

💡 **Final Thought:** _Data isn’t just numbers—it’s a tool that can help make roads safer!_

---

# 📌 How to Run the Project
### **1️⃣ Install Dependencies**
```bash
pip install pandas numpy matplotlib seaborn folium sqlite3 dash plotly scikit-learn imbalanced-learn
```

### **2️⃣ Run the Machine Learning Model**
```bash
python model.py
```

### **3️⃣ Launch the Interactive Dashboard**
```bash
python app.py
```
🌍 **Open in Browser:** `http://127.0.0.1:8050`

---

# 🔗 Resources & Credits
📌 **Dataset:** [US Accidents Dataset - Kaggle](https://www.kaggle.com/sobhanmoosavi/us-accidents)
📌 **Libraries Used:** Pandas, Seaborn, Matplotlib, Scikit-Learn, Dash, Plotly, Folium

🚀 **Feel free to contribute & improve!** 😊
