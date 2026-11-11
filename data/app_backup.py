
import pandas as pd

# Load existing dataset
df = pd.read_csv("data/news.csv")

# Additional training examples
new_data = [
    ("The sun rises in the east", "REAL"),
    ("India launched a new space mission", "REAL"),
    ("The Indian cricket team won a match", "REAL"),
    ("Scientists published a climate research study", "REAL"),
    ("Schools announced examination schedules", "REAL"),
    ("The government released an official report", "REAL"),

    ("Aliens secretly control every human mind", "FAKE"),
    ("Drinking petrol makes people healthy", "FAKE"),
    ("The moon is made entirely of cheese", "FAKE"),
    ("Every person will receive one billion rupees tomorrow", "FAKE"),
    ("Eating stones gives humans superpowers", "FAKE"),
    ("A cat became the president of India", "FAKE")
]

new_df = pd.DataFrame(
    new_data,
    columns=["text", "label"]
)

# Combine and remove duplicates
df = pd.concat([df, new_df], ignore_index=True)
df = df.drop_duplicates(subset=["text"])

# Save dataset
df.to_csv("data/news.csv", index=False)

print("Dataset updated successfully!")
print(df["label"].value_counts())
print("Total News:", len(df))
