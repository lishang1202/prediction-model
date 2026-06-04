# ============================================================
# Stroke Risk Prediction Web App
# BRF Model + SHAP Explanation
# ============================================================
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import os, warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Stroke Risk Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(__file__)
MODEL_GZ = os.path.join(BASE_DIR, "BRF_28vars.joblib.gz")
MODEL_PKL = os.path.join(BASE_DIR, "BRF_28vars.joblib")

# Auto-decompress model on first run
if not os.path.exists(MODEL_PKL):
    with st.spinner("Decompressing model (one-time)..."):
        import gzip, shutil
        with gzip.open(MODEL_GZ, "rb") as f_in:
            with open(MODEL_PKL, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

# ── Load model ──
@st.cache_resource
def load_model():
    art = joblib.load(MODEL_PKL)
    return art["model"], list(art["features"]), art["optimal_threshold"], art["test_metrics"]

model, features, threshold, metrics = load_model()

# ── Sidebar: Patient Inputs ──
st.sidebar.markdown("## 🧑‍⚕️ Patient Profile")
st.sidebar.markdown("---")

yes_no = {"No": 0, "Yes": 1}
srh_map = {"Poor": 0, "Fair": 1, "Good/Very Good": 2}
vision_map = {"Poor": 0, "Fair": 1, "Good/Very Good": 2}

with st.sidebar.expander("Demographics", expanded=True):
    age = st.number_input("Age", min_value=45, max_value=100, value=65, step=1,
                          help="Range: 45-100 years")
    hchild = st.number_input("Number of Children", min_value=0, max_value=15, value=2, step=1,
                             help="Range: 0-15")
    pension = st.selectbox("Pension Status", list(yes_no.keys()), index=0)

with st.sidebar.expander("Chronic Conditions", expanded=True):
    hibpe = st.selectbox("Hypertension", list(yes_no.keys()), index=0)
    dyslipe = st.selectbox("Dyslipidemia", list(yes_no.keys()), index=0)
    diabe = st.selectbox("Diabetes", list(yes_no.keys()), index=0)
    hearte = st.selectbox("Heart Disease", list(yes_no.keys()), index=0)
    memrye = st.selectbox("Memory Disease", list(yes_no.keys()), index=0)
    livere = st.selectbox("Liver Disease", list(yes_no.keys()), index=0)
    arthre = st.selectbox("Arthritis/Rheumatism", list(yes_no.keys()), index=0)
    chronic_count = st.number_input("Chronic Disease Count", min_value=0, max_value=15, value=1, step=1,
                                    help="Total number of chronic conditions")

with st.sidebar.expander("Physical Function"):
    disability = st.selectbox("Disability", list(yes_no.keys()), index=0)
    iadl_diff = st.selectbox("IADL Difficulty", list(yes_no.keys()), index=0)
    badl_dif = st.selectbox("BADL Difficulty", list(yes_no.keys()), index=0)
    body_pain = st.selectbox("Body Pain", list(yes_no.keys()), index=0)
    hospital = st.selectbox("Hospitalization (past year)", list(yes_no.keys()), index=0)

with st.sidebar.expander("Lifestyle & Mental Health"):
    srh = st.selectbox("Self-Rated Health", list(srh_map.keys()), index=1)
    depression = st.selectbox("Depression", list(yes_no.keys()), index=0)
    vision = st.selectbox("Vision Impairment", list(vision_map.keys()), index=1)

with st.sidebar.expander("Biomarkers"):
    systo = st.number_input("Systolic BP (mmHg)", min_value=60.0, max_value=220.0, value=130.0, step=1.0,
                            help="Normal <120 | Range 60-220")
    hdl_c = st.number_input("HDL Cholesterol (mg/dL)", min_value=5.0, max_value=150.0, value=50.0, step=1.0,
                            help="Normal ≥40 | Range 5-150")
    hbalc = st.number_input("HbA1c (%)", min_value=3.0, max_value=20.0, value=5.6, step=0.1,
                            help="Normal <5.7 | Range 3-20")
    crp = st.number_input("C-Reactive Protein (mg/L)", min_value=0.1, max_value=200.0, value=2.0, step=0.1,
                          help="Normal <3 | Range 0.1-200")
    wbc = st.number_input("White Blood Cell (×10⁹/L)", min_value=1.0, max_value=40.0, value=6.5, step=0.1,
                          help="Normal 4-10 | Range 1-40")
    creatinine = st.number_input("Creatinine (mg/dL)", min_value=0.2, max_value=12.0, value=0.9, step=0.1,
                                 help="Normal 0.6-1.2 | Range 0.2-12")
    ua = st.number_input("Uric Acid (mg/dL)", min_value=1.0, max_value=16.0, value=5.0, step=0.1,
                         help="Normal 3.4-7.0 | Range 1-16")
    tyg_bmi = st.number_input("TyG-BMI Index", min_value=80.0, max_value=600.0, value=200.0, step=1.0,
                              help="Range 80-600")
    tyg_whtr = st.number_input("TyG-WHtR Index", min_value=2.0, max_value=12.0, value=5.0, step=0.1,
                               help="Range 2-12")

# ── Build input DataFrame ──
input_dict = {
    "age": age, "dyslipe": yes_no[dyslipe], "systo": systo, "srh": srh_map[srh],
    "memrye": yes_no[memrye], "iadl_diff": yes_no[iadl_diff], "hibpe": yes_no[hibpe],
    "hdl_c": hdl_c, "hchild": hchild, "hbalc": hbalc, "hearte": yes_no[hearte],
    "disability": yes_no[disability], "crp": crp, "chronic_count": chronic_count,
    "body_pain": yes_no[body_pain], "wbc": wbc, "tyg_bmi": tyg_bmi,
    "creatinine": creatinine, "hospital": yes_no[hospital], "depression": yes_no[depression],
    "vision": vision_map[vision], "livere": yes_no[livere], "diabe": yes_no[diabe],
    "pension": yes_no[pension], "arthre": yes_no[arthre], "badl_dif": yes_no[badl_dif],
    "ua": ua, "tyg_whtr": tyg_whtr,
}
X = pd.DataFrame([input_dict])[features]

# ============================================================
# MAIN PANEL
# ============================================================
st.title("🧠 Incidence_Stroke Risk Prediction")
st.markdown("Based on **Balanced Random Forest** (BRF, 28 variables).")
st.markdown("---")

# ── Predict ──
y_prob = model.predict_proba(X)[0, 1]
y_pred = int(y_prob >= threshold)
risk_label = "⚠️ High Risk" if y_pred == 1 else "✅ Low Risk"
risk_pct = y_prob * 100

col1, col2, col3 = st.columns([1, 1, 1.2])

with col1:
    st.subheader("Prediction")
    st.markdown(f"## {risk_label}")
    st.metric("Stroke Probability", f"{risk_pct:.1f}%")
    st.metric("Risk Score (log-odds)", f"{np.log(y_prob/(1-y_prob)):.2f}" if 0 < y_prob < 1 else "N/A")

with col2:
    st.subheader("Model Threshold")
    st.metric("Optimal Threshold", f"{threshold:.2f}")
    st.metric("Test AUC", f"{metrics['AUC']:.4f}")
    st.metric("Test F1 (optimal)", f"{metrics['F1_optimal']:.4f}")

with col3:
    st.subheader("Risk Gauge")
    gauge_color = "#C4523E" if y_pred == 1 else "#3B7A9E"
    st.markdown(
        f'<div style="background:#eee; border-radius:10px; height:30px; width:100%;">'
        f'<div style="background:{gauge_color}; border-radius:10px; height:30px; width:{risk_pct:.0f}%; '
        f'text-align:center; color:white; font-weight:bold; line-height:30px;">{risk_pct:.0f}%</div>'
        f'</div>', unsafe_allow_html=True,
    )
    st.markdown(f"Threshold line: **{threshold*100:.0f}%**")
    st.markdown(
        f'<div style="background:#eee; border-radius:10px; height:10px; width:100%; position:relative;">'
        f'<div style="background:{"#C4523E" if risk_pct/100 >= threshold else "#3B7A9E"}; '
        f'border-radius:10px; height:10px; width:{max(risk_pct, threshold*100):.0f}%;"></div>'
        f'<div style="position:absolute; top:-5px; left:{threshold*100:.0f}%; width:2px; height:20px; '
        f'background:red;"></div></div>', unsafe_allow_html=True,
    )

st.markdown("---")

# ============================================================
# SHAP EXPLANATION
# ============================================================
st.subheader("🔍 Model Explanation — Current Patient")

st.info(
    "Below shows **which specific features are driving the risk for this patient**. "
    "Red/pink bars push risk higher, blue bars pull risk lower."
)

@st.cache_resource
def load_shap_background(_model):
    X_bg = pd.read_csv(os.path.join(BASE_DIR, "shap_background.csv"))
    explainer = shap.TreeExplainer(_model)
    shap_values = explainer.shap_values(X_bg)
    return explainer, X_bg, shap_values

with st.spinner("Computing SHAP values (one-time setup)..."):
    explainer, X_bg, shap_values = load_shap_background(model)

# Parse multi-output
if isinstance(shap_values, list):
    shap_v = shap_values[1]
elif shap_values.ndim == 3:
    shap_v = shap_values[:, :, 1]
else:
    shap_v = shap_values

if isinstance(explainer.expected_value, list):
    expected_ev = explainer.expected_value[1]
elif isinstance(explainer.expected_value, np.ndarray) and explainer.expected_value.ndim > 0:
    expected_ev = explainer.expected_value[1]
else:
    expected_ev = explainer.expected_value

# Per-instance SHAP for current patient
shap_pred = explainer.shap_values(X)
if isinstance(shap_pred, list):
    shap_pred_v = shap_pred[1]
elif shap_pred.ndim == 3:
    shap_pred_v = shap_pred[:, :, 1]
else:
    shap_pred_v = shap_pred

# ── Per-patient explanation: waterfall + force ──
st.markdown("### 📊 Feature Contribution for This Patient")

col_w, col_f = st.columns(2)

with col_w:
    st.markdown("**Waterfall Plot** — each feature's contribution")
    st.caption("Red = pushes risk higher | Blue = pushes risk lower | Base = average prediction")
    plt.figure(figsize=(10, 5))
    shap.plots.waterfall(
        shap.Explanation(values=shap_pred_v[0], base_values=float(expected_ev),
                         data=X.iloc[0].values, feature_names=features),
        max_display=15, show=False,
    )
    st.pyplot(plt.gcf())
    plt.close()

with col_f:
    st.markdown("**Force Plot** — overall risk push direction")
    st.caption("Red area = features pushing toward high risk | Blue = toward low risk")
    shap.plots.force(float(expected_ev), shap_pred_v[:1], feature_names=features,
                     matplotlib=True, show=False)
    st.pyplot(plt.gcf())
    plt.close()

# ── Top drivers summary table ──
st.markdown("### 📋 Key Drivers")
contributions = pd.DataFrame({
    "Feature": features,
    "Value": X.iloc[0].values,
    "SHAP Contribution": shap_pred_v[0],
})
contributions["|SHAP|"] = np.abs(contributions["SHAP Contribution"])
contributions = contributions.sort_values("|SHAP|", ascending=False).head(8)
contributions["Impact"] = contributions["SHAP Contribution"].apply(
    lambda v: "🔺 Increases risk" if v > 0 else "🔻 Decreases risk"
)
contributions["SHAP Contribution"] = contributions["SHAP Contribution"].map("{:+.4f}".format)

st.dataframe(
    contributions[["Feature", "Value", "SHAP Contribution", "Impact"]],
    use_container_width=True,
    hide_index=True,
)

# ── Global importance ──
st.markdown("---")
st.markdown("### 🌍 Global Feature Importance (all patients)")
st.caption("Average |SHAP| across the training population — shows which variables matter most overall")
plt.figure(figsize=(10, 4))
shap.summary_plot(shap_v, X_bg, feature_names=features, plot_type="bar",
                  max_display=15, show=False)
st.pyplot(plt.gcf())
plt.close()

st.markdown("---")
st.caption("Model: BalancedRandomForest (n_estimators=500) | Data: CHARLS | "
           f"Threshold: {threshold:.2f} | AUC: {metrics['AUC']:.4f}")
