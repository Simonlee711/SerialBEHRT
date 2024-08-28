import argparse
import logging
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score, average_precision_score, precision_recall_curve, roc_curve, auc
from lightgbm import LGBMClassifier
from transformers import AutoTokenizer, AutoModel
from gensim.models import Word2Vec
import torch
import matplotlib.pyplot as plt
import boto3
from io import BytesIO
import gzip

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the target antibiotics
ANTIBIOTICS = ['CLINDAMYCIN', 'DAPTOMYCIN', 'ERYTHROMYCIN', 'GENTAMICIN', 'LEVOFLOXACIN', 
               'NITROFURANTOIN', 'OXACILLIN', 'RIFAMPIN', 'TETRACYCLINE', 'TRIMETHOPRIM/SULFA', 
               'VANCOMYCIN']

def read_csv_from_s3(bucket, key):
    logging.info(f"Reading CSV from S3: {bucket}/{key}")
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=bucket, Key=key)
    body = response['Body']
    with gzip.GzipFile(fileobj=body) as gz:
        data = pd.read_csv(gz)
    return data

def encode_texts(model_name, texts):
    logging.info(f"Encoding texts using {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    if torch.cuda.is_available():
        model.to('cuda')

    embeddings = []
    with torch.no_grad():
        for text in texts:
            encoded_input = tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding='max_length')
            if torch.cuda.is_available():
                encoded_input = {key: val.to('cuda') for key, val in encoded_input.items()}
            output = model(**encoded_input)
            cls_embedding = output.last_hidden_state[:, 0, :]
            embeddings.append(cls_embedding.cpu().numpy())
    return np.vstack(embeddings)

def get_w2v_features(texts, model):
    return np.array([
        np.mean([model.wv[word] for word in text.split() if word in model.wv] or [np.zeros(100)], axis=0)
        for text in texts
    ])

def train_and_evaluate(X_train, X_test, y_train, y_test):
    model = LGBMClassifier(n_estimators=1000, learning_rate=0.05, num_leaves=30)
    model.fit(X_train, y_train)

    y_test_proba = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_test_proba)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-5)
    best_threshold = thresholds[np.argmax(f1_scores)]
    best_f1 = np.max(f1_scores)

    y_test_pred = (y_test_proba >= best_threshold).astype(int)
    mcc = matthews_corrcoef(y_test, y_test_pred)
    roc_auc = roc_auc_score(y_test, y_test_proba)
    prc_auc = average_precision_score(y_test, y_test_proba)
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    auprc = auc(recall, precision)

    return best_f1, mcc, roc_auc, prc_auc, best_threshold, fpr, tpr, auprc, precision, recall

def process_data(data_path, model_name):
    logging.info(f"Processing data from {data_path}")
    temp = pd.read_csv(data_path)
    temp["info"] = temp["arrival"] + " " + temp["triage"] + " " + temp["medrecon"] + " " + temp["codes"] + " " + temp["pyxis"] + " " + temp["vitals"]
    temp['info_cleaned'] = temp['info'].apply(lambda x: x.lower())

    train_val, test = train_test_split(temp, test_size=0.1, random_state=42)
    train, val = train_test_split(train_val, test_size=0.1, random_state=42)

    # Word2Vec
    tokenized_train = [text.split() for text in train['info_cleaned']]
    model_w2v = Word2Vec(tokenized_train, vector_size=100, window=5, min_count=1, workers=4)
    X_train_w2v = get_w2v_features(train['info_cleaned'], model_w2v)
    X_test_w2v = get_w2v_features(test['info_cleaned'], model_w2v)

    # Transformer models
    X_train_transformer = encode_texts(model_name, train['info'].tolist())
    X_test_transformer = encode_texts(model_name, test['info'].tolist())

    results = {}
    for antibiotic in ANTIBIOTICS:
        logging.info(f"Training and evaluating for {antibiotic}")
        y_train = train[antibiotic].astype(int)
        y_test = test[antibiotic].astype(int)

        results[antibiotic] = {}
        for name, (X_train, X_test) in [
            ("Word2Vec", (X_train_w2v, X_test_w2v)),
            (model_name, (X_train_transformer, X_test_transformer))
        ]:
            metrics = train_and_evaluate(X_train, X_test, y_train, y_test)
            results[antibiotic][name] = {
                'Optimal Threshold': metrics[4],
                'Test Metrics': {
                    'F1 Score': metrics[0],
                    'Matthews Correlation Coefficient': metrics[1],
                    'ROC AUC': metrics[2],
                    'PRC AUC': metrics[3],
                    'fpr': metrics[5],
                    'tpr': metrics[6],
                    'auprc': metrics[7],
                    'precision': metrics[8],
                    'recall': metrics[9]
                }
            }

    return results

def save_results(results, output_path):
    logging.info(f"Saving results to {output_path}")
    with open(output_path, 'wb') as file:
        pickle.dump(results, file)

def plot_results(results, plot_type):
    logging.info(f"Plotting {plot_type} curves")
    fig, axs = plt.subplots(3, 4, figsize=(20, 15))
    fig.tight_layout(pad=3.0)

    model_colors = {
        "Word2Vec": '#2ca02c',
        "distilbert-base-uncased": '#d62728',
        "EMBO/BioMegatron345mUncased": '#9467bd'
    }

    for i, antibiotic in enumerate(ANTIBIOTICS):
        row, col = divmod(i, 4)
        for model, metrics in results[antibiotic].items():
            if plot_type == 'roc':
                fpr, tpr = metrics['Test Metrics']['fpr'], metrics['Test Metrics']['tpr']
                axs[row, col].plot(fpr, tpr, label=f'{model} (AUC = {metrics["Test Metrics"]["ROC AUC"]:.4f})', color=model_colors.get(model, 'gray'))
                axs[row, col].set_title(f'{antibiotic} - ROC Curve')
                axs[row, col].set_xlabel('False Positive Rate')
                axs[row, col].set_ylabel('True Positive Rate')
            elif plot_type == 'prc':
                precision, recall = metrics['Test Metrics']['precision'], metrics['Test Metrics']['recall']
                axs[row, col].plot(recall, precision, label=f'{model} (AUC = {metrics["Test Metrics"]["auprc"]:.4f})', color=model_colors.get(model, 'gray'))
                axs[row, col].set_title(f'{antibiotic} - Precision-Recall Curve')
                axs[row, col].set_xlabel('Recall')
                axs[row, col].set_ylabel('Precision')

        axs[row, col].legend(loc='best')

    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Antibiotic Prediction Script")
    parser.add_argument("--data_path", required=True, help="Path to the input data CSV file")
    parser.add_argument("--model_name", default="distilbert-base-uncased", help="Name of the transformer model to use")
    parser.add_argument("--output_path", required=True, help="Path to save the results pickle file")
    parser.add_argument("--plot", choices=['roc', 'prc', 'both', 'none'], default='none', help="Type of plot to generate")

    args = parser.parse_args()

    results = process_data(args.data_path, args.model_name)
    save_results(results, args.output_path)

    if args.plot in ['roc', 'both']:
        plot_results(results, 'roc')
    if args.plot in ['prc', 'both']:
        plot_results(results, 'prc')

if __name__ == "__main__":
    main()
