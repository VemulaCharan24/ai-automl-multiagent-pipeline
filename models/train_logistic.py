import json
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

train = json.load(open("data/train.json"))

X = [x["text"] for x in train]
y_task = [x["task_type"] for x in train]
y_domain = [x["domain"] for x in train]
y_metric = [x["metric"] for x in train]

embedder = SentenceTransformer("all-MiniLM-L6-v2")
X_emb = embedder.encode(X, convert_to_numpy=True)

task_enc = LabelEncoder()
domain_enc = LabelEncoder()
metric_enc = LabelEncoder()

y_task_enc = task_enc.fit_transform(y_task)
y_domain_enc = domain_enc.fit_transform(y_domain)
y_metric_enc = metric_enc.fit_transform(y_metric)

task_model = LogisticRegression(max_iter=1000)
domain_model = LogisticRegression(max_iter=1000)
metric_model = LogisticRegression(max_iter=1000)

task_model.fit(X_emb, y_task_enc)
domain_model.fit(X_emb, y_domain_enc)
metric_model.fit(X_emb, y_metric_enc)

joblib.dump(embedder, "models/embedder.pkl")
joblib.dump(task_model, "models/task_model.pkl")
joblib.dump(domain_model, "models/domain_model.pkl")
joblib.dump(metric_model, "models/metric_model.pkl")

joblib.dump(task_enc, "models/task_encoder.pkl")
joblib.dump(domain_enc, "models/domain_encoder.pkl")
joblib.dump(metric_enc, "models/metric_encoder.pkl")

print("Training complete")
