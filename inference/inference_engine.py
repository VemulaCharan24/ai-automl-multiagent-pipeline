# inference/inference_engine.py

import joblib
import numpy as np

# -----------------------------
# LOAD MODELS
# -----------------------------
embedder = joblib.load("models/embedder.pkl")

task_model = joblib.load("models/task_model.pkl")
domain_model = joblib.load("models/domain_model.pkl")
metric_model = joblib.load("models/metric_model.pkl")

task_enc = joblib.load("models/task_encoder.pkl")
domain_enc = joblib.load("models/domain_encoder.pkl")
metric_enc = joblib.load("models/metric_encoder.pkl")

# -----------------------------
# MULTI-INTENT SPLIT
# -----------------------------
def split_into_subtasks(text):
    text = text.lower()
    for sep in [" and ", " then ", ","]:
        if sep in text:
            return [t.strip() for t in text.split(sep) if len(t.strip()) > 3]
    return [text]

# -----------------------------
# CONTEXT AUGMENTATION
# -----------------------------
def generate_variants(text):
    text = text.lower()
    variants = [text]

    if "predict" in text:
        variants += [
            text.replace("predict", "estimate"),
            text.replace("predict", "forecast")
        ]

    if "which" in text or "who" in text:
        variants.append("classify " + text)

    if "heatmap" in text:
        variants.append("plot heatmap from dataset")

    if "cluster" in text:
        variants.append(text.replace("cluster", "group"))

    return list(set(variants))


def get_embedding(text):
    variants = generate_variants(text)
    embs = embedder.encode(variants, convert_to_numpy=True)
    return np.mean(embs, axis=0).reshape(1, -1)

# -----------------------------
# SEMANTIC CORRECTIONS
# -----------------------------
def semantic_task_override(text, task, conf):
    text = text.lower()

    if any(w in text for w in ["team", "winner", "player"]):
        if "predict" in text:
            return "classification", conf * 0.7

    if "heatmap" in text:
        return "non_ml", conf * 0.7

    return task, conf


def semantic_domain_override(text, domain, conf):
    text = text.lower()

    if any(w in text for w in ["plot", "heatmap", "graph", "chart"]):
        return "vision", conf * 0.6

    return domain, conf

# -----------------------------
# CONFLICT DETECTION
# -----------------------------
def extract_task_signals(text):
    text = text.lower()

    return {
        "classification": int(any(w in text for w in ["classify", "classification"])),
        "regression": int("predict" in text),
        "clustering": int(any(w in text for w in ["cluster", "segmentation", "group"])),
        "non_ml": int(any(w in text for w in ["write", "code", "program", "implement"]))
    }


def detect_conflict(signals):
    active = [k for k, v in signals.items() if v > 0]
    return len(active) > 1, active

# -----------------------------
# SINGLE PREDICTION
# -----------------------------
def predict_single(text):

    emb = get_embedding(text)

    t_probs = task_model.predict_proba(emb)[0]
    d_probs = domain_model.predict_proba(emb)[0]
    m_probs = metric_model.predict_proba(emb)[0]

    t_idx = np.argmax(t_probs)
    d_idx = np.argmax(d_probs)
    m_idx = np.argmax(m_probs)

    task = str(task_enc.inverse_transform([t_idx])[0])
    domain = str(domain_enc.inverse_transform([d_idx])[0])
    metric = str(metric_enc.inverse_transform([m_idx])[0])

    t_conf = float(t_probs[t_idx])
    d_conf = float(d_probs[d_idx])
    m_conf = float(m_probs[m_idx])

    # -----------------------------
    # SEMANTIC CORRECTIONS
    # -----------------------------
    task, t_conf = semantic_task_override(text, task, t_conf)
    domain, d_conf = semantic_domain_override(text, domain, d_conf)

    # -----------------------------
    # CONFLICT HANDLING
    # -----------------------------
    signals = extract_task_signals(text)
    conflict, active_tasks = detect_conflict(signals)

    needs_clarification = False
    message = None

    if conflict:
        t_conf *= 0.4
        d_conf *= 0.7
        m_conf *= 0.7

        needs_clarification = True
        message = f"Conflicting intent detected: {active_tasks}. Please clarify."

    # -----------------------------
    # FINAL OUTPUT
    # -----------------------------
    return {
        "subtask": text,
        "task_type": task,
        "domain": domain,
        "metric": metric,
        "confidence": {
            "task": round(t_conf, 3),
            "domain": round(d_conf, 3),
            "metric": round(m_conf, 3)
        },
        "needs_clarification": needs_clarification,
        "message": message
    }

# -----------------------------
# FINAL PIPELINE
# -----------------------------
def analyze_prompt(text):

    subtasks = split_into_subtasks(text)
    results = [predict_single(sub) for sub in subtasks]

    return {
        "original_input": text,
        "num_subtasks": len(results),
        "results": results
    }

# -----------------------------
# TEST (for local run)
# -----------------------------
if __name__ == "__main__":

    test_inputs = [
        "Predict house prices based on tabular data",
        "Cluster customers and predict sales",
        "Plot heatmap of correlation",
        "Classify emails and detect fraud"
    ]

    for t in test_inputs:
        print("\n====================")
        print(analyze_prompt(t))
