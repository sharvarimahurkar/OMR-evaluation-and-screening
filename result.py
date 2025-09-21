import pandas as pd

# Example: assuming you already have
# results = [(q_no, marked, correct, status), ...]
# where status is "Correct" / "Wrong" / "Unanswered"

results = [
    (1, "A", "A", "Correct"),
    (2, "-", "C", "Wrong"),
    (3, "B", "B", "Correct"),
    (4, "D", "B", "Wrong"),
    # 👆 just sample data – you will generate this dynamically
]

# Convert to DataFrame
df = pd.DataFrame(results, columns=["Q.No", "Marked", "Correct", "Result"])

# Save to CSV
df.to_csv("omr_results.csv", index=False)

print("✅ Results saved to omr_results.csv")
