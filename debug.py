from datasets import load_dataset

dataset = load_dataset("keivalya/MedQuad-MedicalQnADataset", split="train")

print(dataset.column_names)
print(dataset[0])