# generate_dataset.py

import random
import json
from sklearn.model_selection import train_test_split

DATA = [
    # Regression
    ("predict house prices", "regression", "tabular", "rmse"),
    ("forecast sales", "regression", "time_series", "rmse"),
    ("estimate temperature", "regression", "time_series", "mae"),
    ("predict stock market trends", "regression", "time_series", "rmse"),
    ("predict customer churn rate", "regression", "tabular", "mae"),
    ("estimate carbon emissions", "regression", "tabular", "rmse"),
    ("forecast energy consumption", "regression", "time_series", "mae"),
    ("predict project completion time", "regression", "tabular", "rmse"),
    ("estimate crop yield", "regression", "vision", "mae"),
    ("forecast website traffic", "regression", "time_series", "rmse"),
    ("predict rainfall amounts", "regression", "time_series", "mae"),
    ("What will be the price of a house?", "regression", "tabular", "rmse"),
    ("Can you forecast next quarter's sales figures?", "regression", "time_series", "rmse"),
    ("I need an estimate for the daily temperature", "regression", "time_series", "mae"),
    ("Determine the future stock market trends", "regression", "time_series", "rmse"),
    ("Calculate the churn rate for customers", "regression", "tabular", "mae"),

    # Classification
    ("classify spam emails", "classification", "nlp", "accuracy"),
    ("detect objects in images", "classification", "vision", "accuracy"),
    ("which team will win IPL", "classification", "time_series", "accuracy"),
    ("predict winner of tournament", "classification", "time_series", "accuracy"),
    ("identify fraudulent transactions", "classification", "tabular", "f1_score"),
    ("categorize customer feedback", "classification", "nlp", "accuracy"),
    ("diagnose medical conditions from images", "classification", "vision", "precision"),
    ("determine sentiment of reviews", "classification", "nlp", "accuracy"),
    ("classify animal species from photos", "classification", "vision", "recall"),
    ("predict customer default", "classification", "tabular", "auc"),
    ("identify malware", "classification", "nlp", "accuracy"),
    ("classify types of cells", "classification", "vision", "accuracy"),
    ("predict disease outbreak", "classification", "time_series", "accuracy"),
    ("Is this email spam or not?", "classification", "nlp", "accuracy"),
    ("Recognize objects in pictures", "classification", "vision", "accuracy"),
    ("Who is going to win the next match?", "classification", "time_series", "accuracy"),
    ("Find fraudulent activity in financial data", "classification", "tabular", "f1_score"),
    ("Sort customer reviews into categories", "classification", "nlp", "accuracy"),

    # Clustering
    ("cluster customers", "clustering", "tabular", "silhouette"),
    ("segment market data", "clustering", "tabular", "davies_bouldin"),
    ("group similar documents", "clustering", "nlp", "coherence"),
    ("identify distinct user behaviors", "clustering", "tabular", "silhouette"),
    ("cluster images by content", "clustering", "vision", "silhouette"),
    ("discover anomalies in network traffic", "clustering", "time_series", "isolation_forest"),
    ("Group my customers into segments", "clustering", "tabular", "silhouette"),
    ("Perform market segmentation", "clustering", "tabular", "davies_bouldin"),
    ("Find similar documents", "clustering", "nlp", "coherence"),
    ("Identify different types of user behavior", "clustering", "tabular", "silhouette"),

    # Non-ML
    ("write a python program", "non_ml", "none", "missing"),
    ("merge datasets", "non_ml", "none", "missing"),
    ("plot heatmap from dataset", "non_ml", "vision", "missing"),
    ("generate a report from data", "non_ml", "none", "missing"),
    ("clean data for analysis", "non_ml", "tabular", "missing"),
    ("convert data format", "non_ml", "none", "missing"),
    ("create a dashboard", "non_ml", "vision", "missing"),
    ("read a CSV file", "non_ml", "none", "missing"),
    ("connect to a database", "non_ml", "none", "missing"),
    ("implement a sorting algorithm", "non_ml", "none", "missing"),
    ("describe the data statistics", "non_ml", "tabular", "missing"),
    ("Can you write some code for me?", "non_ml", "none", "missing"),
    ("Combine these two data tables", "non_ml", "none", "missing"),
    ("Display a heatmap of this data", "non_ml", "vision", "missing"),
    ("Create a summary report from the sales data", "non_ml", "none", "missing"),

    # Logically Inconsistent/Ambiguous Samples
    ("classify customers into segments", "clustering", "tabular", "silhouette"), # Classification and Clustering signals
    ("predict next quarter sales using customer groups", "regression", "time_series", "rmse"), # Regression and Clustering signals
    ("write code to classify images", "non_ml", "vision", "missing"), # Non-ML and Classification signals
    ("group house prices for forecasting", "regression", "tabular", "rmse"), # Clustering and Regression signals
    ("analyze data by writing a program", "non_ml", "none", "missing"), # Analyze (ML implied) and Non-ML signals
    ("classify this text for sentiment analysis and cluster it", "classification", "nlp", "accuracy"), # Two ML tasks
    ("predict fraud using classification", "classification", "tabular", "f1_score"), # Redundant but shows combined intent
    ("cluster and predict stock trends", "clustering", "time_series", "isolation_forest") # Two ML tasks

]

def generate_dataset(n=10000): # Increased dataset size to 10000
    return [
        {
            "text": t,
            "task_type": task,
            "domain": domain,
            "metric": metric
        }
        for t, task, domain, metric in (random.choice(DATA) for _ in range(n))
    ]

if __name__ == "__main__":
    data = generate_dataset(10000) # Increased dataset size to 10000

    train, val = train_test_split(data, test_size=0.2, random_state=42) # Added random_state for reproducibility

    json.dump(train, open("/content/train.json", "w"), indent=2)
    json.dump(val, open("/content/val.json", "w"), indent=2)

    print("Dataset ready")
