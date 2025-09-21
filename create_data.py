import pandas as pd
import numpy as np

# Create 200 fake students
np.random.seed(42)
data = pd.DataFrame({
    "student_id": range(1, 201),
    "attendance": np.random.uniform(50, 100, 200),  # 50% - 100%
    "avg_marks": np.random.uniform(40, 100, 200),   # 40 - 100 marks
    "fee_pending": np.random.uniform(0, 5000, 200)  # 0 - 5000 currency units
})

# Decide dropout based on simple rules for simulation
data["dropout"] = (
    (data["attendance"] < 70) | 
    (data["avg_marks"] < 50) | 
    (data["fee_pending"] > 3000)
).astype(int)

data.to_csv("students.csv", index=False)
print("Sample data created as students.csv")
