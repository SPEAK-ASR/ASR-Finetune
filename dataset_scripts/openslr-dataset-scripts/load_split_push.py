from datasets import load_dataset, DatasetDict

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

REPO_ID = os.getenv('HF_REPO_ID')
TOKEN = os.getenv('HF_TOKEN')
PRIVATE = os.getenv('HF_PRIVATE', 'False')

# Load your dataset
dataset = load_dataset("irudachirath/large-sinhala-asr-dataset")

# First split: train vs temp (val+test)
train_test = dataset["train"].train_test_split(
    test_size=0.3, seed=42
)

# Split temp into validation and test
val_test = train_test["test"].train_test_split(
    test_size=0.67, seed=42
)

# Final DatasetDict
dataset_splits = {
    "train": train_test["train"],
    "validation": val_test["train"],
    "test": val_test["test"],
}

dataset_splits = DatasetDict({
    split: ds.select_columns(["audio", "text"])
    for split, ds in dataset_splits.items()
})



dataset_splits.push_to_hub(
    REPO_ID,
    private=PRIVATE,
    token=TOKEN
)
