import pandas as pd
import plotly.express as px
import streamlit as st
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data import HOME, PURPOSES, simulate

st.set_page_config(page_title="Automated Loan Approval", page_icon="🏦", layout="wide")
st.title("🏦 Automated Loan Approval Decisions")
st.caption("Classification model predicting loan eligibility (SYL Bank use case) · SMOTE for class imbalance · simulated applications")
CAT = ["home_ownership", "purpose"]


@st.cache_resource
def train():
    df = simulate()
    X, y = df.drop(columns="approved"), df["approved"]
    num = [c for c in X if c not in CAT]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
                             ("num", StandardScaler(), num)])
    out = {}
    for k, m in {"Random forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=0, n_jobs=-1),
                 "Gradient boosting": HistGradientBoostingClassifier(random_state=0)}.items():
        p = Pipeline([("pre", pre), ("smote", SMOTE(random_state=0)), ("model", m)]).fit(Xtr, ytr)
        out[k] = dict(pipe=p, proba=p.predict_proba(Xte)[:, 1])
    return df, Xte, yte, out


df, Xte, yte, models = train()

with st.sidebar:
    st.header("Settings")
    model_name = st.radio("Model", list(models))
    thr = st.slider("Approval threshold", 0.1, 0.9, 0.5, 0.05, help="Minimum predicted probability of repayment to approve")
    st.markdown("[Portfolio](https://noe2019.github.io/DA-Portfolio-Website/)")

m = models[model_name]
pred = (m["proba"] >= thr).astype(int)
c = st.columns(5)
c[0].metric("Applications", f"{len(df):,}")
c[1].metric("Historical approval rate", f"{df.approved.mean():.0%}")
c[2].metric("ROC-AUC", f"{roc_auc_score(yte, m['proba']):.3f}")
c[3].metric("Precision (approve)", f"{precision_score(yte, pred):.2f}")
c[4].metric("Recall (approve)", f"{recall_score(yte, pred):.2f}")

t1, t2, t3 = st.tabs(["Decide on an application", "Model evaluation", "Portfolio insights"])
with t1:
    a, b, c3 = st.columns(3)
    app = {"annual_income": a.number_input("Annual income ($)", 10000, 500000, 65000, 1000),
           "loan_amount": a.number_input("Loan amount ($)", 1000, 300000, 15000, 500),
           "term_months": a.selectbox("Term (months)", [12, 24, 36, 60], index=2),
           "credit_score": b.slider("Credit score", 300, 850, 690),
           "employment_years": b.slider("Years employed", 0, 40, 5),
           "debt_to_income": b.slider("Debt-to-income (%)", 0.0, 60.0, 15.0, 0.5),
           "previous_defaults": c3.selectbox("Previous defaults", [0, 1, 2, 3, 4]),
           "home_ownership": c3.selectbox("Home ownership", HOME),
           "purpose": c3.selectbox("Purpose", PURPOSES)}
    p = m["pipe"].predict_proba(pd.DataFrame([app])[Xte.columns])[0, 1]
    if p >= thr:
        st.success(f"✅ APPROVE — predicted repayment probability {p:.0%} (threshold {thr:.0%})")
    else:
        st.error(f"❌ REFER / DECLINE — predicted repayment probability {p:.0%} (threshold {thr:.0%})")
with t2:
    cm = confusion_matrix(yte, pred)
    l, r = st.columns(2)
    l.plotly_chart(px.imshow(cm, text_auto=True, x=["Declined", "Approved"], y=["Declined", "Approved"],
                             labels=dict(x="Predicted", y="Actual"), color_continuous_scale="Blues", title="Confusion matrix"),
                   use_container_width=True)
    thrs = [i / 20 for i in range(2, 19)]
    curve = pd.DataFrame({"threshold": thrs, "F1": [f1_score(yte, m["proba"] >= t) for t in thrs],
                          "Approval rate": [(m["proba"] >= t).mean() for t in thrs]})
    r.plotly_chart(px.line(curve, x="threshold", y=["F1", "Approval rate"], title="Threshold trade-off"), use_container_width=True)
with t3:
    l, r = st.columns(2)
    df["score_band"] = pd.cut(df.credit_score, [299, 580, 670, 740, 800, 850], labels=["Poor", "Fair", "Good", "Very good", "Excellent"])
    l.plotly_chart(px.bar(df.groupby("score_band", observed=True).approved.mean().reset_index(), x="score_band", y="approved",
                          labels={"approved": "Approval rate", "score_band": "Credit score band"}, title="Approval rate by credit score"),
                   use_container_width=True)
    r.plotly_chart(px.bar(df.groupby("purpose").approved.mean().sort_values().reset_index(), x="approved", y="purpose", orientation="h",
                          labels={"approved": "Approval rate", "purpose": ""}, title="Approval rate by purpose"), use_container_width=True)
