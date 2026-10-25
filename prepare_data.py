
import pandas as pd
from pathlib import Path

# Create the data folder if it does not exist
data_folder = Path(__file__).parent / "data"
data_folder.mkdir(exist_ok=True)

# Small sample dataset to test the project
messages = [
    "Congratulations! You won a free prize. Claim now!",
    "Win cash now by clicking this link",
    "You have won a lottery reward",
    "Claim your free gift today",
    "Urgent! You won a cash prize",
    "Get a free offer now",
    "Hello, how are you?",
    "Please send me the class notes",
    "Your meeting is scheduled for tomorrow",
    "Thank you for your help",
    "Can we meet for lunch today?",
    "Please complete your college assignment"
]

labels = [
    "spam", "spam", "spam", "spam", "spam", "spam",
    "ham", "ham", "ham", "ham", "ham", "ham"
]

df = pd.DataFrame({
    "label": labels,
    "message": messages
})

output_file = data_folder / "spam.csv"
df.to_csv(output_file, index=False)

print("Dataset created successfully!")
print("Saved to:", output_file)
print(df.head())
