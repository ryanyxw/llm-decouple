import matplotlib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines

sns.set()
sns.set_context('paper', font_scale=1.4)
sns.set_style('whitegrid', {'font.family':'serif', 'font.serif':'Times New Roman'})

# ---------------------------
# Data for both plots
# ---------------------------

data1 = {
    "Method": [
        "Control (OLMo 1B)",
        "Low-risk Baseline",
        "Toxic Baseline",
        "Unlikelihood",
        "Masked"
    ],
    "Generation Score": [0.15928025890778, 0.162872340508015, 0.177123587315563, 0.166595857347425, 0.162394999356116],
    "Generation SE": [0, 0.000702803724476045, 0.0019427213112138, 0.000944197313461597, 0.00142535170314515],
    "ROCAUC (Base)": [0.8135, 0.813866666666666, 0.822833333333333, 0.831566666666666, 0.829766666666666],
    "ROCAUC SE": [0, 0.00184059169230379, 0.0020077627128501, 0.00254055986830551, 0.00148136573621927],
}

data2 = {
    "Method": [
        "Control (OLMo 1B)",
        "Low-risk Baseline",
        "Toxic Baseline",
        "Unlikelihood",
        "Masked"
    ],
    "Generation Score": [0.161068972735045, 0.161914871288323, 0.169150705140612, 0.162843433546159, 0.161669042902275],
    "Generation SE": [0, 0.000248932249495689, 0.000605386787534553, 0.000520020618091152, 0.000381600661417872],
    "ROCAUC (Base)": [0.8504, 0.845533333333333, 0.8539, 0.858933333333333, 0.858966666666666],
    "ROCAUC SE": [0, 0.000696020433927357, 0.00040414518843273, 0.000835330939076129, 0.000868587614719691],
}

# ---------------------------
# Create figure and axes
# ---------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ---------------------------
# Plot 1 (Pretrained)
# ---------------------------

df1 = pd.DataFrame(data1)
df1["Generation CI"] = df1["Generation SE"] * 2
df1["ROCAUC CI"] = df1["ROCAUC SE"] * 2

ax = axes[0]
baseline_df = df1.iloc[:3].copy().sort_values(by="Generation Score")
min_y = df1["ROCAUC (Base)"].min() - 0.003
x_vals = list(baseline_df["Generation Score"]) + [baseline_df["Generation Score"].iloc[-1], baseline_df["Generation Score"].iloc[0]]
y_vals = list(baseline_df["ROCAUC (Base)"]) + [min_y, min_y]

ax.fill(x_vals, y_vals, facecolor='none', hatch='///', edgecolor='mediumseagreen', linewidth=0.0)
ax.text(0.173, 0.8125, "Previous Pareto Frontier", fontsize=11, ha='center', va='center',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.8))

for _, row in df1.iterrows():
    ax.errorbar(row["Generation Score"], row["ROCAUC (Base)"], xerr=row["Generation CI"], yerr=row["ROCAUC CI"],
                fmt='none', ecolor='gray', capsize=4)
    ax.annotate(
        row["Method"], (row["Generation Score"], row["ROCAUC (Base)"]),
        ha="right" if row["Method"] in ["Toxic Baseline", "Masked"] else "left",
        textcoords="offset points",
        xytext=(-8, 14) if row["Method"] in ["Toxic Baseline", "Masked"] else (10, -18 if row["Method"] == "Control (OLMo 1B)" else 12),
        bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white')
    )

sns.scatterplot(data=df1.iloc[:3], x="Generation Score", y="ROCAUC (Base)", s=100,
                color=matplotlib.rcParams['axes.prop_cycle'].by_key()['color'][0], ax=ax, zorder=5)
sns.scatterplot(data=df1.iloc[3:], x="Generation Score", y="ROCAUC (Base)", s=500,
                color=matplotlib.rcParams['axes.prop_cycle'].by_key()['color'][1], ax=ax, marker="*", zorder=6)

ax.set_title("(a) Pretraining")
ax.set_xlabel("Generation (RealToxicityPrompts)")
ax.set_ylabel("Understanding (CivilComments AUROC)")
ax.set_xlim(right=df1["Generation Score"].max() + 0.001)

# ---------------------------
# Plot 2 (IT)
# ---------------------------

df2 = pd.DataFrame(data2)
df2["Generation CI"] = df2["Generation SE"] * 2
df2["ROCAUC CI"] = df2["ROCAUC SE"] * 2

ax = axes[1]
baseline_df = pd.concat([df2.iloc[:1].copy(), df2.iloc[2:3].copy()]).sort_values(by="Generation Score")
min_y = df2["ROCAUC (Base)"].min() - 0.003
x_vals = list(baseline_df["Generation Score"]) + [baseline_df["Generation Score"].iloc[-1], baseline_df["Generation Score"].iloc[0]]
y_vals = list(baseline_df["ROCAUC (Base)"]) + [min_y, min_y]

ax.fill(x_vals, y_vals, facecolor='none', hatch='///', edgecolor='mediumseagreen', linewidth=0.0)
ax.text(0.167, 0.844, "Previous Pareto Frontier", fontsize=11, ha='center', va='center',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.8))

for _, row in df2.iterrows():
    ax.errorbar(row["Generation Score"], row["ROCAUC (Base)"], xerr=row["Generation CI"], yerr=row["ROCAUC CI"],
                fmt='none', ecolor='gray', capsize=4)
    ax.annotate(
        row["Method"], (row["Generation Score"], row["ROCAUC (Base)"]),
        ha="right" if row["Method"] in ["Toxic Baseline", "Masked"] else "left",
        textcoords="offset points",
        xytext=(-8, 14) if row["Method"] in ["Toxic Baseline", "Masked"] else (10, -18 if row["Method"] == "Control (OLMo 1B)" else 12),
        bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white')
    )

sns.scatterplot(data=df2.iloc[:3], x="Generation Score", y="ROCAUC (Base)", s=100,
                color=matplotlib.rcParams['axes.prop_cycle'].by_key()['color'][0], ax=ax, zorder=5)
sns.scatterplot(data=df2.iloc[3:], x="Generation Score", y="ROCAUC (Base)", s=500,
                color=matplotlib.rcParams['axes.prop_cycle'].by_key()['color'][1], ax=ax, marker="*", zorder=6)

ax.set_title("(b) Pretraining + Instruction-tuning")
ax.set_xlabel("Generation (RealToxicityPrompts)")
ax.set_ylabel("Understanding (CivilComments AUROC)")
ax.set_xlim(right=df2["Generation Score"].max() + 0.0005)
# ---------------------------
# Shared legend
# ---------------------------

baseline_handle = mlines.Line2D([], [], color=matplotlib.rcParams['axes.prop_cycle'].by_key()['color'][0],
                                marker='o', linestyle='None', markersize=8, label='Baselines')
our_method_handle = mlines.Line2D([], [], color=matplotlib.rcParams['axes.prop_cycle'].by_key()['color'][1],
                                  marker='*', linestyle='None', markersize=12, label='SLUNG')

plt.legend(handles=[baseline_handle, our_method_handle], loc='upper right')

plt.tight_layout()

# Save
plt.savefig("figure2.pdf", format="pdf")