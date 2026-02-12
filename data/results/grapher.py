"""
Plot four separate influence‑vs‑budget graphs (Facebook, Wikipedia, Erdos‑Renyi,
Bitcoin) with three lines on each graph:

    • Degree Discount
    • Amplified Coverage
    • CELF   (any CELF/CELF++ variants are auto‑detected)

Formatting matches your previous plots:
    – all 60 budget points shown
    – x‑axis tick marks every 5
    – light dotted grid
    – legend for the three lines
"""

import ast
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------  Load & clean data  --------------------------- #
csv_path = "main.csv"          # adjust if your file lives elsewhere
df = pd.read_csv(csv_path)

# Each cell in “Average Influence” is a string like "[123.4]".
# Convert it to a single float.
def parse_influence(val):
    parsed = ast.literal_eval(val)
    return parsed[0] if isinstance(parsed, list) else float(parsed)

df["Influence"] = df["Average Influence"].apply(parse_influence)

# Normalise model names (strip spaces, unify CELF variants)
df["Model"] = df["Model"].str.strip()
df.loc[df["Model"].str.contains(r"celf", case=False, regex=True), "Model"] = "CELF"

# ---------------------------  Plotting helpers  ---------------------------- #
network_map = {
    "Facebook": "Facebook",
    "Wikipedia": "Wikipedia",
    "Random":   "Erdos‑Renyi",
    "Bitcoin":  "Bitcoin",
}

models = ["Degree Discount", "Amplified Coverage", "CELF"]
colors = {                       # optional custom colours
    "Degree Discount":   "#daa520",   # goldenrod
    "Amplified Coverage": "#d55e00",  # vermillion
    "CELF":              "#c51b7d",   # magenta
}

# ---------------------------  Create one graph per network  ---------------- #
for raw_name, nice_name in network_map.items():
    fig, ax = plt.subplots(figsize=(6, 5))

    for model in models:
        sub = (
            df[(df["Network"] == raw_name) & (df["Model"] == model)]
            .sort_values("Budget")
        )
        if sub.empty:
            continue                              # skip if that model missing
        ax.plot(
            sub["Budget"],
            sub["Influence"],
            label=model,
            linewidth=2,
            color=colors.get(model),
        )

    ax.set_xlabel(r"Budget $k$")
    ax.set_ylabel("Influence")
    ax.set_xticks(range(5, 61, 5))               # ticks at 5,10,…,60
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend()
    ax.set_title(nice_name)                      # remove if you prefer no title
    plt.tight_layout()
    plt.show()
    # To save instead of—or in addition to—showing, uncomment:
    # fig.savefig(f"{nice_name.lower().replace('‑','_')}_influence.png", dpi=300)
