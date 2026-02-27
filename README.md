  Heart Disease Risk Clustering 
A machine learning-powered web application that categorizes patients into risk groups using K-Means Clustering and Principal Component Analysis (PCA). Built with a FastAPI backend and a minimal web interface.

  Project Overview
This project analyzes clinical heart disease data to identify patterns among patients. By applying dimensionality reduction, we improved the clustering performance to better distinguish between different risk levels.

-Algorithm: K-Means Clustering (K=3).
-Optimization: PCA was used to reduce noise and improve the Silhouette Score from 0.12 to ~0.40.
-Risk Categories:
  -High Risk: Identified by significant ST depression (oldpeak) and lower max heart rate.
  -Medium Risk: Characterized by advanced age and elevated cholesterol levels.
  -Low Risk: Patients with healthy EKG patterns and optimal heart rate performance.



Installation & Setup
1. Download the Trained Model 
For security and storage efficiency, the trained model is hosted on Kaggle.
Download ZGA-heart_disease.pkl from https://www.kaggle.com/code/zaferakdarma/heart-disease-patients-clusterization-k-means
Place the file directly into the root directory of this project (next to app.py).

2. Environment Setup
2.1. Clone the repository
git clone https://github.com/zaferakdarma/MLPython.git
cd MLPython
2.2. Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\activate  # Windows
2.3. Install dependencies
pip install -r requirements.txt

3. Run the Application
uvicorn app:app --reload


Tech Stack
Backend: FastAPI (Python)
ML Libraries: Scikit-Learn, Pandas, NumPy
Frontend: HTML5, Jinja2
