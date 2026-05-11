"""
reports/ altındaki CSV dosyalarını PNG olarak görselleştirir.
Çıktılar:
  reports/figures/table_model_comparison.png
  reports/figures/table_loocv_predictions.png
  reports/figures/chart_ovr_f1.png
"""

import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT    = pathlib.Path(__file__).parent.parent
REPORTS = ROOT / "reports"
FIGS    = REPORTS / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

TARGET_LABELS = {
    "group":            "Grup (4 sınıf)",
    "anxiety_level":    "Kaygı Düzeyi (3 sınıf)",
    "rearing_profile":  "Rearing Profili (3 sınıf)",
    "grooming_profile": "Grooming Profili (3 sınıf)",
}
RANDOM_BASELINE = {"group": 0.250, "anxiety_level": 0.333,
                   "rearing_profile": 0.333, "grooming_profile": 0.333}

# ── 1. Model Karşılaştırma Heatmap ───────────────────────────────────────────

df_cmp = pd.read_csv(REPORTS / "model_comparison.csv")
pivot  = df_cmp.pivot(index="model", columns="target", values="f1_macro")
pivot  = pivot[["group", "anxiety_level", "rearing_profile", "grooming_profile"]]
pivot.index = ["Logistic Reg", "Random Forest", "SVM", "XGBoost"]
col_labels = [TARGET_LABELS[c] for c in pivot.columns]

fig, ax = plt.subplots(figsize=(11, 4))
im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0.0, vmax=0.75, aspect="auto")
plt.colorbar(im, ax=ax, label="F1-Macro (LOOCV)", fraction=0.046, pad=0.04)

ax.set_xticks(range(len(col_labels)))
ax.set_xticklabels(col_labels, fontsize=10)
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index, fontsize=10)

# hücre değerleri
for r in range(pivot.shape[0]):
    for c in range(pivot.shape[1]):
        val  = pivot.values[r, c]
        base = RANDOM_BASELINE[pivot.columns[c]]
        txt  = f"{val:.3f}"
        color = "white" if val < 0.35 else "black"
        ax.text(c, r, txt, ha="center", va="center",
                fontsize=11, fontweight="bold", color=color)

# rastgele baz çizgisi (alt bilgi)
baseline_str = "  |  ".join(
    f"{TARGET_LABELS[t]}: baz={v:.2f}" for t, v in RANDOM_BASELINE.items()
)
fig.text(0.5, -0.04, f"Rastgele tahmin bazları — {baseline_str}",
         ha="center", fontsize=7.5, color="gray")

ax.set_title("Model Karşılaştırması — F1-Macro (LOOCV, n=12)",
             fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
fig.savefig(FIGS / "table_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("[OK] table_model_comparison.png")

# ── 2. LOOCV Tahmin Tablosu (XGBoost — en tutarlı model) ────────────────────

df_loo = pd.read_csv(REPORTS / "loocv_predictions.csv")
df_xgb = df_loo[df_loo["model"] == "XGBoost"].copy()

targets = ["group", "anxiety_level", "rearing_profile", "grooming_profile"]
col_labels_short = ["Grup", "Kaygı", "Rearing", "Grooming"]

# sıçan sırası — kohort bazlı
subject_order = ["MA1_1","MA1_2","MA1_3",
                 "MA3_1","MA3_2","MA3_3",
                 "MA5_1","MA5_2","MA5_3",
                 "MA7_1","MA7_2","MA7_3"]
group_of = {
    "MA1_1":"Control","MA1_2":"Control","MA1_3":"Control",
    "MA3_1":"Aspartame","MA3_2":"Aspartame","MA3_3":"Aspartame",
    "MA5_1":"Grapefruit","MA5_2":"Grapefruit","MA5_3":"Grapefruit",
    "MA7_1":"ASP+GF","MA7_2":"ASP+GF","MA7_3":"ASP+GF",
}
group_colors = {"Control":"#4CAF50","Aspartame":"#2196F3",
                "Grapefruit":"#FF9800","ASP+GF":"#9C27B0"}

fig, ax = plt.subplots(figsize=(13, 6))
ax.set_xlim(-0.5, len(targets) - 0.5)
ax.set_ylim(-0.5, len(subject_order) - 0.5)
ax.set_xticks(range(len(targets)))
ax.set_xticklabels(col_labels_short, fontsize=11, fontweight="bold")
ax.set_yticks(range(len(subject_order)))
ax.set_yticklabels(subject_order, fontsize=9)
ax.invert_yaxis()

# grup renk şeridi (sol)
for r, sid in enumerate(subject_order):
    grp = group_of[sid]
    ax.add_patch(mpatches.FancyBboxPatch(
        (-0.48, r - 0.45), 0.06, 0.9,
        boxstyle="square,pad=0", color=group_colors[grp], zorder=3))

# hücre kutucukları
for c, tgt in enumerate(targets):
    sub = df_xgb[df_xgb["target"] == tgt].set_index("subject_id")
    for r, sid in enumerate(subject_order):
        if sid not in sub.index:
            continue
        row     = sub.loc[sid]
        correct = row["correct"]
        pred    = str(row["predicted"]).replace("Aspartame+Grapefruit","ASP+GF")
        true_l  = str(row["true"]).replace("Aspartame+Grapefruit","ASP+GF")

        bg  = "#C8E6C9" if correct else "#FFCDD2"
        ec  = "#2E7D32" if correct else "#C62828"
        ax.add_patch(mpatches.FancyBboxPatch(
            (c - 0.42, r - 0.42), 0.84, 0.84,
            boxstyle="round,pad=0.03",
            facecolor=bg, edgecolor=ec, linewidth=1.2, zorder=2))

        mark = "✓" if correct else "✗"
        ax.text(c, r - 0.10, mark, ha="center", va="center",
                fontsize=14, color=ec, fontweight="bold", zorder=4)
        ax.text(c, r + 0.22, pred, ha="center", va="center",
                fontsize=6.5, color="#333333", zorder=4)

# ızgara
ax.set_axisbelow(True)
ax.grid(True, which="both", color="#DDDDDD", linewidth=0.8)

# kohort ayraç çizgileri
for y in [2.5, 5.5, 8.5]:
    ax.axhline(y, color="#888888", linewidth=1.5, linestyle="--")

# legend
patches = [mpatches.Patch(color=v, label=k) for k, v in group_colors.items()]
patches += [mpatches.Patch(color="#C8E6C9", label="Doğru tahmin"),
            mpatches.Patch(color="#FFCDD2", label="Yanlış tahmin")]
ax.legend(handles=patches, bbox_to_anchor=(1.01, 1), loc="upper left",
          fontsize=8, framealpha=0.9)

ax.set_title("LOOCV Tahmin Tablosu — XGBoost  (✓ doğru / ✗ yanlış, altında tahmin edilen etiket)",
             fontsize=11, fontweight="bold", pad=10)
plt.tight_layout()
fig.savefig(FIGS / "table_loocv_predictions.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("[OK] table_loocv_predictions.png")

# ── 3. OvR İkili F1 Bar Grafiği ──────────────────────────────────────────────

df_ovr = pd.read_csv(REPORTS / "ovr_binary_f1.csv")
grp_order  = ["Control", "Aspartame", "Grapefruit", "ASP+GF"]
grp_colors = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0"]
f1_vals    = [df_ovr.set_index("group").loc[g, "f1_binary"] for g in grp_order]
acc_vals   = [df_ovr.set_index("group").loc[g, "accuracy"]  for g in grp_order]

x     = np.arange(len(grp_order))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))
bars1 = ax.bar(x - width/2, f1_vals, width, label="F1 (ikili)",
               color=grp_colors, alpha=0.85, edgecolor="white", linewidth=1.2)
bars2 = ax.bar(x + width/2, acc_vals, width, label="Accuracy",
               color=grp_colors, alpha=0.40, edgecolor=grp_colors, linewidth=1.2)

# değer etiketleri
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
            f"{bar.get_height():.3f}", ha="center", va="bottom",
            fontsize=9, fontweight="bold")
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
            f"{bar.get_height():.3f}", ha="center", va="bottom",
            fontsize=9, color="#555555")

# rastgele baz çizgisi (ikili: 0.5 doğruluk, ama F1 için ~0.33)
ax.axhline(0.333, color="red", linewidth=1.2, linestyle="--",
           label="F1 rastgele baz (0.333)")
ax.axhline(0.750, color="gray", linewidth=1.0, linestyle=":",
           label="Acc. çoğunluk baz (0.750)")

ax.set_xticks(x)
ax.set_xticklabels(grp_order, fontsize=11)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Skor", fontsize=11)
ax.set_title("One-vs-Rest İkili Sınıflandırma — XGBoost (LOOCV, n=12)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9, loc="upper left")
ax.set_axisbelow(True)
ax.yaxis.grid(True, color="#EEEEEE", linewidth=0.8)

# not
fig.text(0.5, -0.04,
         "Not: F1=0.000 → model tüm örnekleri negatif tahmin etti (n=3 pozitif / n=9 negatif dengesizliği).",
         ha="center", fontsize=8, color="gray")

plt.tight_layout()
fig.savefig(FIGS / "chart_ovr_f1.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("[OK] chart_ovr_f1.png")
