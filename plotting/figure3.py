import matplotlib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines

# Set style again for clean replot
sns.set()
sns.set_context('paper', font_scale=1.6)
sns.set_style('whitegrid', {'font.family': 'serif', 'font.serif': 'Palatino'})

# Rebuild data (in case of context reset)
token_labels = ['20M', '40M', '80M', '160M', '320M']

masked = [
    (0.152104938, 0.8256),
    (0.1541042105, 0.8297),
    (0.155904074, 0.832),
    (0.1572641481, 0.8374),
    (0.166484858995485, 0.8427)
]
vanilla = [
    (0.1636266313, 0.8161),
    (0.1699153257, 0.8196),
    (0.1892873926, 0.826),
    (0.2041662896, 0.8335),
    (0.23693523341020012, 0.8364)
]
unlikelihood = [
    (0.1535531354, 0.8366),
    (0.1572332991, 0.8387),
    (0.1591481161, 0.8377),
    (0.1657531677, 0.8374),
    (0.16826801053445006, 0.8359)
]

control = [(0.1503543585, 0.8146)]

# model_737 = [(0.1606898429, 0.8118)]

# Build DataFrame
df = pd.DataFrame({
    "Generation Toxicity": [pt[0] for pt in control + masked + vanilla + unlikelihood],
    "Understanding Toxicity": [pt[1] for pt in control + masked + vanilla + unlikelihood],
    "Method": (
        ["Control"] +
        ["Masked"] * len(masked) +
        ["Vanilla"] * len(vanilla) +
        ["Unlikelihood"] * len(unlikelihood)
    ),
    "Tokens Inserted": (
        [None] +
        token_labels +
        token_labels +
        token_labels
    )
})

# Plot setup
plt.figure(figsize=(8, 6))
palette = sns.color_palette()

# Plot lines for masked and vanilla
for method, marker, color in zip(["Masked", "Vanilla", "Unlikelihood"], ['o', 'o', 'o'], palette):
    subset = df[df["Method"] == method]
    plt.plot(subset["Generation Toxicity"], subset["Understanding Toxicity"],
             marker=marker, linestyle='-', color=color, label=method, linewidth=4)
    plt.scatter(subset["Generation Toxicity"], subset["Understanding Toxicity"],
                color=color, zorder=5, s=100)

    # Add token size text at each point
    for _, row in subset.iterrows():
        if row["Method"] == "Unlikelihood":
            if row["Tokens Inserted"] in ["80M", "160M"]:
                plt.text(row["Generation Toxicity"] + 0.0010, row["Understanding Toxicity"] + 0.0002,
                         row["Tokens Inserted"], fontsize=12,
                         ha='left', va='bottom')
            elif row["Tokens Inserted"] == "320M":
                plt.text(row["Generation Toxicity"] + 0.0005, row["Understanding Toxicity"] - 0.0004,
                         row["Tokens Inserted"], fontsize=12,
                         ha='left', va='top')
            else:
                plt.text(row["Generation Toxicity"] - 0.0005, row["Understanding Toxicity"] + 0.0002,
                         row["Tokens Inserted"], fontsize=12,
                         ha='right', va='bottom')
        else:
            plt.text(row["Generation Toxicity"] + 0.0005, row["Understanding Toxicity"] - 0.0004,
                     row["Tokens Inserted"], fontsize=12,
                     ha='left', va='top')

# Plot the baseline point
control_point = df[df["Method"] == "Control"].iloc[0]
plt.scatter(control_point["Generation Toxicity"], control_point["Understanding Toxicity"],
            color="black", s=100, label="Control", zorder=6)

# Legend and labels
control_handle = mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=8, label='Control')
masked_handle = mlines.Line2D([], [], color=palette[0], marker='o', linestyle='-', linewidth=2, markersize=8, label='Masked SLUNG')
vanilla_handle = mlines.Line2D([], [], color=palette[1], marker='o', linestyle='-', linewidth=2, markersize=8, label='Toxic Baseline')
unlikelihood_handle = mlines.Line2D([], [], color=palette[2], marker='o', linestyle='-', linewidth=2, markersize=8, label='Unlikelihood SLUNG')

# plt.legend(handles=[filtered_handle, masked_handle, vanilla_handle], loc='lower right')
plt.legend(handles=[control_handle, masked_handle, vanilla_handle, unlikelihood_handle], loc='lower right')


# plt.title("Toxic Data Quantity Scaling")
plt.xlabel("Generation (RealToxicityPrompts)")
plt.ylabel("Understanding (CivilComments AUROC)")
plt.tight_layout()


output_path = "figure3.pdf"
plt.savefig(output_path, format="pdf")