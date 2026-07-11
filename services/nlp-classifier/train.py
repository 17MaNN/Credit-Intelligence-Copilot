"""Fine-tune distilbert-base-uncased for 4-class intent classification.
Run once at build time: python train.py
Requires internet access to download the pretrained base model from
Hugging Face Hub - this happens during `docker build`, not at runtime."""
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from data import generate, LABELS

MODEL_NAME = "distilbert-base-uncased"
OUT_DIR = "nlp_model"

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.enc = tokenizer(texts, truncation=True, padding=True, max_length=64, return_tensors="pt")
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.enc.items()}
        item["labels"] = self.labels[idx]
        return item


texts, labels = generate()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(LABELS))

dataset = TextDataset(texts, labels, tokenizer)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
model.train()
for epoch in range(3):
    total_loss = 0.0
    for batch in loader:
        opt.zero_grad()
        out = model(**batch)
        out.loss.backward()
        opt.step()
        total_loss += out.loss.item()
    print(f"epoch {epoch} loss {total_loss:.4f}")

model.save_pretrained(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)