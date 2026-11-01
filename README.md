# Automated Loan Approval Decisions

Classification model that predicts loan eligibility to automate manual approvals (SYL Bank use case).

**Live app:** https://noe-loan-approval-3fc5d244781f.herokuapp.com/
**Portfolio:** https://noe2019.github.io/DA-Portfolio-Website/

## Features
- SMOTE oversampling for class imbalance
- Random Forest vs. Gradient Boosting
- Adjustable decision threshold with confusion matrix
- Single-applicant decision form

## Tech stack
Python, scikit-learn, imbalanced-learn (SMOTE), Streamlit

## Data
All data in this app is **simulated** (generated in `data.py` with a fixed random seed) to reproduce the structure of the original project without exposing real or confidential records.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment
Deployed on Heroku (stack `heroku-24`, Python 3.12) using the `Procfile` in this repository.

---
Author: Noé Carème Fouotsa Manfouo
