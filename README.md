🎙️ Human Voice Classification & Clustering System
📌 Project Overview

This project develops a complete Machine Learning pipeline to classify and cluster human voice profiles using pre-extracted acoustic features.

The system includes:

📊 Exploratory Data Analysis (EDA)
🎯 Feature Selection (Top 20 Important Features)
🤖 Supervised Learning (Multiple Model Comparison)
🧠 Unsupervised Learning (K-Means Clustering)
🌐 Interactive Web Application using Streamlit

## 📂 Project Structure

```bash
HUMAN VOICE RECOGNITION/
│
├── app.py
│
├── models/
│   ├── kmeans.pkl
│   ├── pca.pkl
│   ├── scaler_20.pkl
│   ├── selected_features.pkl
│   ├── svm_20.pkl
│   └── top_20_features.pkl
│
├── human.ipynb          # EDA
├── human2.ipynb         # Models with SMOTE
├── human3.ipynb         # Models without SMOTE
├── cluster.ipynb        # Clustering
│
├── vocal_gender_features_cleaned.csv
├── X_top20.csv
├── y.csv
└── README.md
```


Model Performance

The following models were trained and evaluated:
| Model               | Accuracy   | Approx Errors | Notes             |
| ------------------- | ---------- | ------------- | ----------------- |
| Logistic Regression | 99.13%     | 26            | Linear baseline   |
| Random Forest       | 99.43%     | 17            | Ensemble          |
| XGBoost             | 99.53%     | 14            | Boosting          |
| LightGBM            | 99.70%     | ~9            | Strong boosting   |
| MLP                 | 99.86%     | 4             | Neural Network    |
| **SVM**             | **99.93%** | **2**         | Best Performer |

Final Selected Model: SVM

Achieved 99.93% accuracy
Selected for deployment in Streamlit app

Clustering

Applied K-Means Clustering
Reduced dimensions using PCA
Evaluated cluster purity
Visualized clusters in 2D space

🛠️ Tech Stack:
Python
Pandas
NumPy
Scikit-learn
XGBoost
LightGBM
Imbalanced-learn (SMOTE)
Matplotlib
Seaborn
Streamlit

Author: Avanthi UC
