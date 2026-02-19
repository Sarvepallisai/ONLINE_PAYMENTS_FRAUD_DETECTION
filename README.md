# 🛡️ Online Payments Fraud Detection

A Machine Learning web application that detects fraudulent online payment transactions in real-time using a Random Forest Classifier trained on 6.3 million real transaction records.

**Model Accuracy: 99.24%**

---

## 📁 Project Structure

online payments fraud detection/
├── data/
│   └── PS_20174392719_1491204439457_log.csv
├── flask/
│   ├── templates/
│   │   ├── home.html
│   │   ├── predict.html
│   │   └── submit.html
│   ├── app.py
│   ├── app_ibm.py
│   └── payments.pkl
├── training/
│   ├── ONLINE PAYMENTS FRAUD DETECTION.ipynb
│   └── payments.pkl
└── training_ibm/
└── online payments fraud prediction using ibm.ipynb

---

## 🚀 How to Run

**Step 1 — Install packages**
pip3 install -r requirements.txt

**Step 2 — Train model**
cd training
python3 train_model.py

**Step 3 — Run app**
cd ../flask
python3 app.py

**Step 4 — Open browser**
http://127.0.0.1:5000
