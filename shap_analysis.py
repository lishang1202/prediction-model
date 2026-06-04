# ============================================================
# SHAP Analysis for BRF 28-vars Model
# TreeExplainer + summary + dependence + waterfall
# ============================================================
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})

DATA = "D:/桌面/claude/模型构建"
OUT = "D:/桌面/claude/shap"

# ── Load model ──
print("Loading BRF 28-vars model...")
art = joblib.load(f"{DATA}/实验7_BRF参数调优/BRF_28vars.joblib")
model = art["model"]
features = list(art["features"])
threshold = art["optimal_threshold"]

# ── Load training data for SHAP ──
train = pd.read_stata(f"{DATA}/两种方法及以上共识变量/train_2plusmethod.dta")
X = train[features]
y = train["stroke_incident"].values.ravel()
print(f"Training set: {X.shape[0]} samples, {len(features)} features")

# ── TreeExplainer ──
print("Running TreeExplainer (this may take a few minutes)...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# BRF multi-output: shape (n_samples, n_features, n_classes)
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

print(f"  SHAP values shape: {shap_v.shape}")

# ── Figure 1: SHAP Summary (BeeSwarm + Bar) ──
print("Figure 1: SHAP Summary...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

shap.summary_plot(shap_v, X,plot_type="dot", show=False, color_bar=True, max_display=20)
axes[0] = plt.gca()
axes[0].set_title("(a) SHAP BeeSwarm — 28 vars", fontsize=10, fontweight="bold")

shap.summary_plot(shap_v, X,plot_type="bar", show=False, max_display=20)
axes[1] = plt.gca()
axes[1].set_title("(b) Feature Importance (|SHAP|) — 28 vars", fontsize=10, fontweight="bold")

plt.tight_layout()
for fmt in ["png", "svg", "pdf"]:
    fig.savefig(f"{OUT}/SHAP_Summary_28vars.{fmt}", dpi=300, bbox_inches="tight")
plt.close()

# ── Figure 2: Top-6 Dependence Plots ──
print("Figure 2: Dependence plots...")
mean_abs_shap = np.abs(shap_v).mean(axis=0).flatten()
top6_idx = np.argsort(mean_abs_shap)[-6:][::-1].tolist()
top6_names = [features[i] for i in top6_idx]

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

for i, (name, col_idx) in enumerate(zip(top6_names, top6_idx)):
    ax = axes[i]
    shap.dependence_plot(col_idx, shap_v, X,feature_names=features, ax=ax, show=False)
    ax.set_title(f"({chr(97+i)}) {name}", fontsize=9, fontweight="bold")

plt.tight_layout()
for fmt in ["png", "svg", "pdf"]:
    fig.savefig(f"{OUT}/SHAP_Dependence_28vars.{fmt}", dpi=300, bbox_inches="tight")
plt.close()

# ── Figure 3: Waterfall for a TP and a FP case (saved individually) ──
print("Figure 3: Waterfall plots...")
y_prob = model.predict_proba(X)[:, 1]
y_pred = (y_prob >= threshold).astype(int)

tp_idx = np.where((y == 1) & (y_pred == 1))[0]
fp_idx = np.where((y == 0) & (y_pred == 1))[0]

for idx_set, case_label in [(tp_idx, "TruePositive"), (fp_idx, "FalsePositive")]:
    if len(idx_set) > 0:
        sidx = int(idx_set[0])
        plt.figure(figsize=(10, 5))
        shap.plots.waterfall(
            shap.Explanation(values=shap_v[sidx],
                             base_values=float(expected_ev),
                             data=X.iloc[sidx].values,
                             feature_names=features),
            max_display=15, show=False
        )
        plt.title(f"{case_label}", fontsize=10, fontweight="bold")
        for fmt in ["png", "svg", "pdf"]:
            plt.savefig(f"{OUT}/SHAP_Waterfall_{case_label}_28vars.{fmt}", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved: SHAP_Waterfall_{case_label}_28vars")
    else:
        print(f"  No {case_label} in test set")

# ── Print top features ──
print(f"\nTop features by mean |SHAP| (28 vars):")
for rank, idx in enumerate(top6_idx, 1):
    print(f"  {rank}. {features[idx]:25s}  {mean_abs_shap[idx]:.4f}")

# ── Figure 4: Force Plot for individual samples ──
print("Figure 4: Force plots for individual cases...")
for sidx, case_label in [(int(tp_idx[0]), "TruePositive"), (int(fp_idx[0]), "FalsePositive")]:
    plt.figure(figsize=(12, 1.5))
    shap.plots.force(float(expected_ev), shap_v[sidx:sidx+1], feature_names=features, matplotlib=True, show=False)
    plt.title(f"{case_label} — force plot", fontsize=9, fontweight="bold")
    for fmt in ["png", "svg", "pdf"]:
        plt.savefig(f"{OUT}/SHAP_Force_{case_label}_28vars.{fmt}", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: SHAP_Force_{case_label}_28vars")

print(f"\nDone! All SHAP plots saved to {OUT}")
