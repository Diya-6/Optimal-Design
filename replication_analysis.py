
import numpy as np
import pandas as pd

from pathlib import Path


RESULTS_DIR = Path("results")

criteria = ["D", "A", "E", "I"]

rows = []

for criterion in criteria:

    design = pd.read_csv(
        RESULTS_DIR / f"{criterion}_optimal_design.csv"
    ).to_numpy()

    n = len(design)

    unique_points = np.unique(
        design,
        axis=0
    )

    m = len(unique_points)

    p = 6

    rows.append({

        "Design": criterion + "-optimal",

        "Total runs": n,

        "Unique settings": m,

        "Replicated runs": n - m,

        "Residual df": n - p,

        "Pure error df": n - m,

        "Lack of fit df": m - p

    })


df = pd.DataFrame(rows)

print("\nREPLICATION ANALYSIS")

print(df.to_string(index=False))

df.to_csv(
    RESULTS_DIR / "replication_analysis.csv",
    index=False
)
