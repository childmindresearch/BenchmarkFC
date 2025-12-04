"""Generate list of method/func combinations for sparse behavioral prediction."""

import pandas as pd
from pathlib import Path

# Load the CSV with method/func IDs
csv_path = Path("output/mean_sparsity_by_function_all_thresholds.csv")
df = pd.read_csv(csv_path)

# Extract unique method/func combinations
combinations = df[["method", "func_id", "func"]].drop_duplicates()
combinations = combinations.sort_values(["method", "func_id"])

print(f"Total combinations: {len(combinations)}")
print(f"PySPI methods: {len(combinations[combinations['method'] == 'pyspi'])}")
print(f"Skarf methods: {len(combinations[combinations['method'] == 'skarf'])}")

# Save to file for array job
output_file = Path("resources/sparse_prediction_method_func_list.txt")
output_file.parent.mkdir(exist_ok=True)

with output_file.open("w") as f:
    for _, row in combinations.iterrows():
        f.write(f"{row['method']}\t{row['func']}\n")

print(f"\nSaved to: {output_file}")
print("\nFirst 10 combinations:")
print(combinations.head(10).to_string(index=False))
