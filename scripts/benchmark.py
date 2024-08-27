import argparse
import logging
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
from transformers import (AutoModel, AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, 
    DataCollatorWithPadding, TrainingArguments, Trainer, TextClassificationPipeline, 
    AdamW, get_scheduler
)
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets, load_metric

# PyTorch data handling
from torch.utils.data import DataLoader

from scripts.train_test_split import custom_train_test_split
from scripts.encoder import encode_texts, encode_texts_biolm
from scripts.train_test import evaluate_antibiotics_with_confidence_intervals, print_results
from scripts.plot import plot_roc_curves, plot_auprc_curves, calculate_confidence_interval_curves




# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(file_path):
    logger.info(f"Loading data from {file_path}")
    return pd.read_csv(file_path)

def save_pickle(results, filename):
    logger.info(f"Saving results to {filename}")
    with open(filename, 'wb') as file:
        pickle.dump(results, file)

def load_pickle(filename):
    logger.info(f"Loading results from {filename}")
    with open(filename, 'rb') as file:
        return pickle.load(file)

def main(args):
    logger.info("Starting benchmarking script")
    
    data = load_data(args.data_file)
    train, val, test = split_data(data, args.test_size, args.val_size)
    
    antibiotics = ['CLINDAMYCIN', 'ERYTHROMYCIN', 'GENTAMICIN', 'LEVOFLOXACIN', 
                   'OXACILLIN', 'TETRACYCLINE', 'TRIMETHOPRIM/SULFA', 'VANCOMYCIN']
    
    model_names = ["DistilBERT", "BioMegatron", "Medbert", "BioBERT", "BlueBERT", "PubMedBERT",
                   "Gatotron", "BiomedRoBERTa", "ClinicalBERT", "Bio+ClinicalBERT", "sciBERT",
                   "bioLM", "RadBERT", "LinkBERT"]
    
    results = {}
    for model_name in model_names:
        logger.info(f"Processing {model_name}")
        X_train = encode_texts(model_name, train['patient_paragraph'].tolist())
        X_test = encode_texts(model_name, test['patient_paragraph'].tolist())
        model_results = evaluate_antibiotics_with_confidence_intervals(X_train, X_test, train, test, antibiotics)
        results[model_name] = model_results
        save_pickle(model_results, f"{args.output_dir}/{model_name}_results.pickle")
    
    dictionaries = [(results[name], name) for name in model_names]
    
    model_colors = {
        "DistilBERT": '#d62728', "BioMegatron": '#9467bd', "Medbert": "#d3ff0d",
        "BioBERT": "#370e19", "BlueBERT": "#3d85c6", "PubMedBERT": "#38761d",
        "Gatotron": "#c90076", "BiomedRoBERTa": "#674ea7", "ClinicalBERT": "#e69138",
        "Bio+ClinicalBERT": "#660000", "sciBERT": "#16537e", "bioLM": "#6a329f",
        "RadBERT": "#c90076", "LinkBERT": "#744700"
    }
    
    plot_bar_chart(antibiotics, dictionaries, model_colors, "ROC AUC")
    plot_bar_chart(antibiotics, dictionaries, model_colors, "PRC AUC")
    
    rank_roc = compute_average_rank(antibiotics, dictionaries, "ROC AUC")
    rank_prc = compute_average_rank(antibiotics, dictionaries, "PRC AUC")
    
    logger.info("ROC AUC Rankings:")
    logger.info(rank_roc)
    logger.info("PRC AUC Rankings:")
    logger.info(rank_prc)
    
    logger.info("Benchmarking script completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Foundation Models for Antibiotic Prediction")
    parser.add_argument("--data_file", type=str, required=True, help="Path to the input CSV file")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save output files")
    parser.add_argument("--test_size", type=float, default=0.1, help="Proportion of data to use for testing")
    parser.add_argument("--val_size", type=float, default=0.1, help="Proportion of training data to use for validation")
    args = parser.parse_args()
    
    main(args)
