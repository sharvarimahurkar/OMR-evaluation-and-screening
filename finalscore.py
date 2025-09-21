import pandas as pd

# Load the CSV file generated after bubble detection
df = pd.read_csv("omr_results.csv")

# Count how many answers were marked correct
score = df["Result"].value_counts().get("Correct", 0)
total = len(df)

print(f"Final Score: {score}/{total}")
