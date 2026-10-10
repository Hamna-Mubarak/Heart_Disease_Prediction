# Heart Disease Prediction System

**Hamna Mubarak** · FA24-BSE-046 · Section A
Machine Learning Fundamentals (AIC354), Assignment 2: Developing an End-to-End Machine Learning System
COMSATS University Islamabad, Lahore Campus

**Live demo:** https://heartdiseaseprediction-omqtad3saltpkdqo4ayxtn.streamlit.app/

An end-to-end binary classification system that predicts whether a patient has heart disease. It follows
the same steps as the instructor's Titanic Passenger Survival Prediction System and is trained on the
[UCI Heart Disease dataset (Cleveland)](https://archive.ics.uci.edu/dataset/45/heart+disease).

## Inputs and output

| Input attribute | UCI column | Values |
|---|---|---|
| Sex | `sex` | Male, Female |
| ChestPainType | `cp` | Typical Angina, Atypical Angina, Non-Anginal Pain, Asymptomatic |
| MajorVessels | `ca` | Zero, One, Two, Three |
| Thal | `thal` | Normal, Fixed Defect, Reversible Defect |

**Output:** HeartDisease (Yes / No), where Yes means UCI `num` is 1 to 4.

## Results (Support Vector Classifier)

| Metric | Value |
|---|---|
| Test accuracy (60 patients, 80/20 stratified split) | 0.88 |
| Precision / Recall (Heart Disease class) | 0.89 / 0.86 |
| 10-fold cross validation accuracy | 0.80 |

## Files

| File | Description |
|---|---|
| `Heart_Disease_Prediction.ipynb` | Full notebook: loading, preprocessing, EDA, label encoding, training, testing, application, feedback, deployment |
| `Heart_Disease_Report.pdf` | The notebook exported as PDF |
| `processed_cleveland.csv` | UCI `processed.cleveland.data` with a header row added (raw dataset) |
| `sample-data.csv` | Pre-processed data with the 4 selected attributes and output |
| `sample-data-encoded.csv`, `training-data-encoded.csv`, `testing-data-encoded.csv`, `model-predictions.csv` | Intermediate files written by the notebook |
| `heart_disease_model.pkl` | Trained SVC model (load with `pickle`) |
| `streamlit-app/app.py` | Streamlit web app used for the cloud deployment |

## Run locally

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook                  # open Heart_Disease_Prediction.ipynb and Run All
```

Run the web app locally:

```bash
pip install -r streamlit-app/requirements.txt
streamlit run streamlit-app/app.py
```

## Load the trained model without retraining

```python
import pickle
import pandas as pd

model = pickle.load(open("heart_disease_model.pkl", "rb"))
# Label codes (from LabelEncoder): Sex Female=0 Male=1; ChestPainType Asymptomatic=0 Atypical Angina=1
# Non-Anginal Pain=2 Typical Angina=3; MajorVessels One=0 Three=1 Two=2 Zero=3;
# Thal Fixed Defect=0 Normal=1 Reversible Defect=2
patient = pd.DataFrame({"Sex": [1], "ChestPainType": [0], "MajorVessels": [2], "Thal": [2]})
print("Heart disease" if model.predict(patient)[0] == 1 else "No heart disease")
```

## Deployment (Streamlit Community Cloud)

At [share.streamlit.io](https://share.streamlit.io), choose **Create app**, then **Deploy a public app from GitHub**:
repository `Hamna-Mubarak/Heart_Disease_Prediction`, branch `main`, main file path `streamlit-app/app.py`,
and Python 3.12 under Advanced settings.
