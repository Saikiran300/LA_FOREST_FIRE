# LA Fire Prediction - Stacking Ensemble Model

This project predicts daily wildfire occurrence in Los Angeles County using an ensemble of Random Forest, XGBoost, and SVM models. It leverages meteorological, satellite, and historical fire data to provide early warnings and support sustainable wildfire management.

---

##  Project Description

Wildfires cause devastating environmental, economic, and human losses.  
Our model uses historical fire data, weather patterns, and satellite imagery to predict wildfire risks with high accuracy.  
The ensemble approach improves robustness and generalization over single models.

---

##  Tech Stack

- **Programming**: Python 3.10
- **Libraries**:
  - `scikit-learn`
  - `xgboost`
  - `random forrest`
  - `XGBoost`
  - `imbalanced-learn`
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
- **Tools**:
  - Google Colab Pro (GPU acceleration)
  - Dask (for large dataset handling)
  - GitHub (version control)

---

## 📂 Repository Structure

```
Random_forest/               # Random Forest model scripts
SVM project files/            # SVM training, tuning, deployment notebooks
Data Preprocessing.ipynb      # Raw weather and satellite data preprocessing
ML Ensemble Model.ipynb       # Final stacking ensemble creation and evaluation
XGBoostWildfireFirst-Copy1.ipynb  # XGBoost standalone model tuning and results
README.md                     # Project overview
```

---

## 🔥 Model Architecture

- **Data Sources**:
  - CAL FIRE wildfire records
  - NOAA GHCN daily weather data
  - MODIS land surface temperature and vegetation indices
- **Preprocessing**:
  - Winsorization, log transforms, PCA
  - Feature engineering (Dryness Score, Spread Score, etc.)
  - KNN Imputation for missing values
  - SMOTE–Tomek resampling for balancing fire vs no-fire days
- **Models**:
  - Support Vector Machine (SVM, RBF kernel)
  - Random Forest
  - XGBoost
- **Meta-Model**:
  - Logistic Regression for stacking

---

## 📊 Key Results

| Metric | Stacking Ensemble Score |
|:-------|:-------------------------|
| Accuracy | 95% |
| Precision (Fire Class) | 82% |
| Recall (Fire Class) | 83% |
| F1-Score | 83% |
| ROC-AUC | 92% |

---

##  Sustainability Impact

Accurate wildfire prediction helps:
- Minimize unnecessary patrols, saving fuel and emissions
- Enable targeted evacuations and pre-deployment of resources
- Reduce ecological and economic damage
- Support climate adaptation strategies

Our solution uses open datasets and scalable pipelines, promoting sustainable AI practices.

---

##  How to Run

1. Clone the repository:

```bash
git clone https://github.com/shashidharbabu/la-fire-prediction.git
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```
*(or manually install sklearn, xgboost, imblearn, etc.)*

3. Open and run the notebooks:

```bash
jupyter notebook
```

or

```bash
colab.research.google.com
```

Start with `Data Preprocessing.ipynb` → `SVM project files/` → `ML Ensemble Model.ipynb`

---

## 🧯 Team Members

- **Vineeth Rayadurgam** (@vineeth917)
- **Shashidhar Babu P V D** (@shashidharbabu)
- **Vaheedur Rehman Mahmud**
- **Akshith Reddy J**

This project was a collaborative effort involving data collection, preprocessing, modeling, tuning, and deployment.

---

##  Future Improvements

- Integrate real-time climate projections
- Add land-use/human activity features
- Deploy the model as an operational wildfire early-warning system (API-based)
- Extend to other wildfire-prone regions

---

##  License

This project is licensed under the MIT License.  
You are free to use, modify, and distribute with attribution.

---

