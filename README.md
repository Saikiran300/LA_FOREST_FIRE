# 🔥 LA Wildfire Risk Prediction using Ensemble Learning

This project forecasts daily wildfire risks in Los Angeles County using a stacked ensemble of Random Forest, XGBoost, and SVM models. It integrates weather data, satellite imagery, and historical fire records to generate reliable early warnings and support proactive wildfire management.

---

## 📘 Project Overview

Wildfires have devastating impacts on the environment, human life, and property. To reduce these risks, this machine learning pipeline uses historical fire occurrences and climate data to predict future wildfire events. Ensemble learning helps improve robustness and predictive performance.

---

## 🛠️ Tech Stack

- **Language**: Python 3.10  
- **Data Libraries**: `pandas`, `numpy`, `scipy`, `tqdm`  
- **Machine Learning**: `scikit-learn`, `xgboost`, `imbalanced-learn`, `joblib`  
- **Visualization**: `matplotlib`, `seaborn`, `plotly`  
- **Geospatial Analysis**: `geopandas`, `rasterio`, `pyproj`, `folium`  
- **Synthetic Data**: `ctgan`, `sdv`  
- **Feature Engineering**: `feature-engine`, `category_encoders`  
- **Model Interpretation**: `shap`, `eli5`, `lime`  
- **Notebook Tools**: `jupyter`, `ipywidgets`, `notebook`  
- **Environment Management**: `python-dotenv`

> 📦 All dependencies are listed in `requirements.txt`.

---

## 📁 Project Structure

```
1_preprocess.ipynb                    # Clean and prepare raw datasets
2_build_features.ipynb                # Feature engineering and transformation
3_random_forest.ipynb                 # Train Random Forest model
4_evaluate.ipynb                      # Evaluate and compare models
5_visualize.ipynb                     # Visualize predictions and metrics
Data_Preprocessing.ipynb              # Additional cleaning steps
XGBoostWildfireFirst-Copy1.ipynb      # XGBoost model tuning and testing
Ensemble + SVM Code.ipynb             # Final ensemble model (RF + XGBoost + SVM)
Best K for KNN Imputation.ipynb       # Optimal K selection for imputation
SVM Grid Search for best parameters.ipynb # Grid search for SVM hyperparameters
SVM model deployment.ipynb            # Save and deploy final SVM model
requirements.txt                      # All required Python dependencies
README.md                             # Project documentation
```

---

## 🔍 Methodology

### 🔹 Data Sources
- **CAL FIRE**: Wildfire incident history  
- **NOAA**: Daily weather observations (temperature, humidity, wind speed)  
- **MODIS**: Satellite-derived vegetation indices (NDVI) and surface temperatures (LST)

### 🔹 Data Processing
- Log transformations and PCA
- Winsorization to handle outliers
- KNN imputation for missing values (tuned with optimal `k`)
- SMOTE–Tomek sampling to balance fire/no-fire labels
- Feature engineering: Dryness Score, Spread Score, NDVI deviation

### 🔹 Models Used
- Random Forest  
- XGBoost  
- Support Vector Machine (RBF kernel)  
- Logistic Regression (meta-learner for stacking)

---

## 📊 Evaluation Metrics

| Metric              | Value    |
|---------------------|----------|
| Accuracy            | 95%      |
| Precision (Fire)    | 82%      |
| Recall (Fire)       | 83%      |
| F1-Score            | 83%      |
| ROC-AUC             | 92%      |

---

## 🌍 Impact

This model can:
- Optimize the allocation of firefighting resources
- Enable earlier and more targeted evacuations
- Reduce property and ecological damage
- Support sustainable and scalable AI-driven disaster management

---

## ▶️ How to Run

1. **Clone the repo**:
   ```bash
   git clone https://github.com/saikiran300/LA_FOREST_FIRE.git
   ```

2. **Install required libraries**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebooks in order**:
   - `1_preprocess.ipynb`
   - `2_build_features.ipynb`
   - `Best K for KNN Imputation.ipynb`
   - `3_random_forest.ipynb`
   - `XGBoostWildfireFirst-Copy1.ipynb`
   - `SVM Grid Search for best parameters.ipynb`
   - `Ensemble + SVM Code.ipynb`
   - `4_evaluate.ipynb`
   - `5_visualize.ipynb`
   - `SVM model deployment.ipynb`

---

## 👨‍💻 Contributors

- **Sai Kiran Reddy Pothuganti**  
- **Vivek Varma Rudraraju**

---

## 🚀 Future Improvements

- Integrate real-time weather and satellite APIs  
- Add features based on land use and human activity  
- Expand model to predict severity/duration of wildfires  
- Deploy as a live web app or RESTful API

---

## 📜 License

This project is licensed under the **MIT License**.  
You are free to use, modify, and distribute with proper credit.
