# models/train_bert.py

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

# -----------------------------
# LOAD DATA
# -----------------------------
train = json.load(open("data/train.json"))

texts = [x["text"] for x in train]
y_task = [x["task_type"] for x in train]
y_domain = [x["domain"] for x in train]
y_metric = [x["metric"] for x in train]

# -----------------------------
# LABEL ENCODING
# -----------------------------
task_enc = LabelEncoder()
domain_enc = LabelEncoder()
metric_enc = LabelEncoder()

y_task_enc = task_enc.fit_transform(y_task)
y_domain_enc = domain_enc.fit_transform(y_domain)
y_metric_enc = metric_enc.fit_transform(y_metric)

# Save encoders
joblib.dump(task_enc, "models/task_enc.pkl")
joblib.dump(domain_enc, "models/domain_enc.pkl")
joblib.dump(metric_enc, "models/metric_enc.pkl")

# -----------------------------
# TOKENIZER
# -----------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

encodings = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=64,
    return_tensors="pt"
)

# -----------------------------
# DATASET CLASS
# -----------------------------
class PromptDataset(Dataset):
    def __init__(self, encodings, y_task, y_domain, y_metric):
        self.encodings = encodings
        self.y_task = y_task
        self.y_domain = y_domain
        self.y_metric = y_metric

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["task"] = torch.tensor(self.y_task[idx], dtype=torch.long)
        item["domain"] = torch.tensor(self.y_domain[idx], dtype=torch.long)
        item["metric"] = torch.tensor(self.y_metric[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.y_task)

dataset = PromptDataset(encodings, y_task_enc, y_domain_enc, y_metric_enc)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# -----------------------------
# MODEL
# -----------------------------
class MultiHeadBERT(nn.Module):
    def __init__(self, num_task, num_domain, num_metric):
        super(MultiHeadBERT, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        self.task_head = nn.Linear(768, num_task)
        self.domain_head = nn.Linear(768, num_domain)
        self.metric_head = nn.Linear(768, num_metric)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]

        task_logits = self.task_head(cls_output)
        domain_logits = self.domain_head(cls_output)
        metric_logits = self.metric_head(cls_output)

        return task_logits, domain_logits, metric_logits

# -----------------------------
# INITIALIZE MODEL
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MultiHeadBERT(
    num_task=len(task_enc.classes_),
    num_domain=len(domain_enc.classes_),
    num_metric=len(metric_enc.classes_)
).to(device)

# -----------------------------
# LOSS + OPTIMIZER
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=2e-5)

# -----------------------------
# TRAINING LOOP
# -----------------------------
epochs = 2  # keep small for demo

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in dataloader:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        task_labels = batch["task"].to(device)
        domain_labels = batch["domain"].to(device)
        metric_labels = batch["metric"].to(device)

        t_logits, d_logits, m_logits = model(input_ids, attention_mask)

        loss_task = criterion(t_logits, task_labels)
        loss_domain = criterion(d_logits, domain_labels)
        loss_metric = criterion(m_logits, metric_labels)

        loss = loss_task + loss_domain + loss_metric
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# -----------------------------
# SAVE MODEL
# -----------------------------
torch.save(model.state_dict(), "models/bert_model.pth")

print("BERT training complete")
