# 🛡️ Online Payments Fraud Detection

A Machine Learning web application that detects fraudulent online payment
transactions in real-time using a **Random Forest Classifier** trained on
6.3 million real transaction records.

**Model Accuracy: 99.24%**

---

## 📁 Project Structure

## 📁 Project Structure

fraud_detection/
├── data/
│   └── fraud_dataset.csv
├── flask/
│   ├── templates/
│   │   ├── home.html        ← Landing page
│   │   ├── predict.html     ← Input form
│   │   └── submit.html      ← Results page
│   ├── app.py               ← Flask backend
│   ├── app_ibm.py           ← IBM Cloud deployment
│   └── payments.pkl         ← Trained model
├── training/
│   ├── ONLINE PAYMENTS FRAUD DETECTION.ipynb
│   └── payments.pkl         ← Saved model
├── training_ibm/
│   └── online payments fraud prediction using ibm.ipynb
└── requirements.txt


---

## 🚀 How to Run (Step by Step)

### Step 1 — Install Python & VS Code
- Download Python 3.10+: https://www.python.org/downloads/
- Download VS Code: https://code.visualstudio.com/
- Install VS Code Extension: **Python** (by Microsoft)

### Step 2 — Open Project in VS Code
1. Open VS Code
2. File → Open Folder → Select `fraud_detection` folder
3. Open Terminal in VS Code: `Ctrl + ~`

### Step 3 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Train the Model
```bash
cd training
python train_model.py
```
This will:
- Load the CSV dataset
- Train a Random Forest model
- Save `payments.pkl` in both `training/` and `flask/` folders
- Show accuracy ~99.24%

### Step 5 — Run the Web App
```bash
cd ../flask
python app.py
```

### Step 6 — Open in Browser
Visit: **http://127.0.0.1:5000**

---

## 🧪 Test Cases

### ✅ Legitimate Transaction
| Field | Value |
|-------|-------|
| Step | 1 |
| Type | PAYMENT |
| Amount | 9839.64 |
| Sender Old Balance | 170136.00 |
| Sender New Balance | 160296.36 |
| Receiver Old Balance | 0.00 |
| Receiver New Balance | 0.00 |

### 🚨 Fraudulent Transaction
| Field | Value |
|-------|-------|
| Step | 1 |
| Type | TRANSFER |
| Amount | 181.00 |
| Sender Old Balance | 181.00 |
| Sender New Balance | 0.00 |
| Receiver Old Balance | 0.00 |
| Receiver New Balance | 0.00 |

---

## 🔬 How It Works (Technical Architecture)

```
User → UI (Web Form) → Flask App → ML Model → Prediction Result
                                      ↑
                           Trained on CSV Data
                           (Data Preprocessing → Train/Test Split → Random Forest)
```

1. **Data** - PaySim dataset with 6.3M transactions
2. **Preprocessing** - Encode transaction types, balance dataset
3. **Training** - Random Forest with 100 trees
4. **Evaluation** - 99.24% accuracy, 98% fraud recall
5. **Deployment** - Flask web app with pickle model

---

## 📊 Model Performance

| Metric | Legitimate | Fraud |
|--------|-----------|-------|
| Precision | 100% | 98% |
| Recall | 99% | 99% |
| F1-Score | 100% | 98% |

**Overall Accuracy: 99.24%**

---

## ⚙️ Tech Stack

- **Python 3.10+**
- **scikit-learn** - Machine Learning
- **pandas / numpy** - Data processing
- **Flask** - Web framework
- **HTML/CSS** - Frontend UI
