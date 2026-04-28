# 🤖 AI-Powered AutoML Pipeline with Multi-Agent Validation

## 🚀 Overview
This project is an intelligent AI system that converts natural language input into a complete machine learning pipeline and validates predictions using a multi-agent-inspired framework.

Unlike traditional AutoML systems, this approach focuses on **intent understanding, reliability, and self-correction**.

---

## 🧠 Key Features

- 🔍 NLP-based intent understanding (task, domain, metric)
- ⚙️ Automated ML pipeline configuration
- 🤖 Multi-model prediction (Logistic + Transformer-based BERT)
- 🔀 Multi-intent query handling
- ⚠️ Conflict detection for ambiguous inputs
- 🧠 Semantic correction layer for improved predictions
- 📊 Confidence scoring for outputs

---

## 🏗️ System Architecture

The system follows a layered pipeline:

1. **Intent Understanding**
   - Converts user text → ML task (classification, regression, clustering)

2. **Embedding & Modeling**
   - Sentence Transformers + Logistic Regression
   - Multi-head BERT (advanced model)

3. **Inference Engine**
   - Multi-intent splitting
   - Context augmentation
   - Prediction with confidence scores

4. **Validation Layer**
   - Conflict detection
   - Semantic correction rules

---

## 🛠️ Tech Stack

- Python  
- Scikit-learn  
- Sentence Transformers  
- HuggingFace Transformers (BERT)  
- PyTorch  
- NumPy / Pandas  

---

## 💡 Problem It Solves

Building ML pipelines manually is complex and requires expertise.

Existing AutoML tools:
- Do not understand user intent well  
- Do not validate predictions  

This system:
✅ Converts natural language → ML pipeline  
✅ Detects ambiguous/conflicting queries  
✅ Improves reliability using hybrid ML + rules  

---

## 📈 Impact

- Reduces manual ML pipeline effort  
- Improves prediction reliability  
- Handles real-world ambiguous inputs  
- Demonstrates intelligent AI system design  

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python main.py
