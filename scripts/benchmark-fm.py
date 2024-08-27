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

from transformers import (
    AutoModel, AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, 
    DataCollatorWithPadding, TrainingArguments, Trainer, TextClassificationPipeline, 
    AdamW, get_scheduler, pipeline, RobertaTokenizerFast
)

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
        
def plot_bar_chart(antibiotics, dictionaries, model_colors, metric, figsize=(20, 12)):
    logger.info(f"Plotting {metric} bar chart")
    n = len(antibiotics)
    cols = 4
    rows = n // cols + (1 if n % cols > 0 else 0)
    fig, axs = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    for i, antibiotic in enumerate(antibiotics):
        row, col = divmod(i, cols)
        ax = axs[row, col]
        sorted_dictionaries = sorted(dictionaries, key=lambda x: x[0][antibiotic]['Test Metrics'][metric])
        y_positions = range(len(sorted_dictionaries))
        for pos, (dictionary, name) in zip(y_positions, sorted_dictionaries):
            value = dictionary[antibiotic]['Test Metrics'][metric]
            ci = dictionary[antibiotic]['Confidence Intervals'][metric]['95% CI']
            error = [[value - ci[0]], [ci[1] - value]]
            line_color = model_colors[name]
            ax.barh(pos, value, color=line_color, xerr=error, capsize=5,
                    label=f'{name} ({metric} = {value:.4f})')
        ax.set_yticks(y_positions)
        ax.set_yticklabels([name for _, name in sorted_dictionaries])
        ax.legend(loc=4)
        ax.set_title(f'{antibiotic} - {metric}', fontsize=20)
        ax.set_xlabel(metric, fontsize=18)
        ax.set_xlim([0, 1])
        ax.tick_params(axis='both', labelsize=15)
    plt.tight_layout()
    plt.show()

def compute_average_rank(antibiotics, dictionaries, metric):
    logger.info(f"Computing average rank for {metric}")
    model_ranks = {name: [] for _, name in dictionaries}
    for antibiotic in antibiotics:
        sorted_dictionaries = sorted(dictionaries, key=lambda x: x[0][antibiotic]['Test Metrics'][metric], reverse=True)
        for rank, (_, name) in enumerate(sorted_dictionaries, start=1):
            model_ranks[name].append(rank)
    average_ranks = {name: sum(ranks) / len(ranks) for name, ranks in model_ranks.items()}
    return dict(sorted(average_ranks.items(), key=lambda item: item[1]))

def evaluate_antibiotics_with_confidence_intervals(X_train, X_test, train, test, antibiotics, n_bootstraps=1000):
    """
    Function to train and evaluate a model for each antibiotic in the list, including confidence intervals
    for metrics using bootstrapping.

    Parameters:
    - X_train: Features for the training set
    - X_test: Features for the testing set
    - train: Training dataset containing the targets
    - test: Testing dataset containing the targets
    - antibiotics: List of antibiotics to evaluate
    - n_bootstraps: Number of bootstrap samples to use for confidence intervals

    Returns:
    - A dictionary containing evaluation results and confidence intervals for each antibiotic.
    """
    results = {}
    for antibiotic in tqdm(antibiotics,desc="Iterating through Antibiotics Progress: "):
        y_train = train[antibiotic].astype(int).reset_index(drop=True)
        y_test = test[antibiotic].astype(int).reset_index(drop=True)
        
        # Initialize and fit the model
        model = LGBMClassifier(n_estimators=1000, learning_rate=0.05, num_leaves=30)
        model.fit(X_train, y_train)
        
        # Predict on test set and calculate probabilities
        y_test_proba = model.predict_proba(X_test)[:, 1]

        # Initial evaluation
        precision, recall, thresholds = precision_recall_curve(y_test, y_test_proba)
        f1_scores = 2 * recall * precision / (recall + precision)
        f1_scores = np.nan_to_num(f1_scores)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_f1 = f1_scores[optimal_idx]
        y_test_pred = (y_test_proba >= optimal_threshold).astype(int)
        mcc_test = matthews_corrcoef(y_test, y_test_pred)
        roc_auc_test = roc_auc_score(y_test, y_test_proba)
        prc_auc_test = average_precision_score(y_test, y_test_proba)
        fpr, tpr, _ = roc_curve(y_test, y_test_proba)
        auprc = auc(recall, precision)

        # Bootstrap confidence intervals
        roc_aucs = []
        prc_aucs = []
        f1_scores_list = []

        for _ in range(n_bootstraps):
            indices = resample(np.arange(len(y_test)), replace=True)
            y_test_resampled = y_test[indices]
            y_test_proba_resampled = y_test_proba[indices]

            roc_aucs.append(roc_auc_score(y_test_resampled, y_test_proba_resampled))
            pr, rc, _ = precision_recall_curve(y_test_resampled, y_test_proba_resampled)
            prc_aucs.append(auc(rc, pr))
            f1 = 2 * rc * pr / (np.maximum(rc + pr, np.finfo(float).eps))
            f1_scores_list.append(np.max(f1))

        # Store results including confidence intervals
        results[antibiotic] = {
            'Optimal Threshold': optimal_threshold,
            'Test Metrics': {
                'F1 Score': optimal_f1,
                'Matthews Correlation Coefficient': mcc_test,
                'ROC AUC': roc_auc_test,
                'PRC AUC': prc_auc_test,
                'fpr': fpr,
                'tpr': tpr,
                'auprc': auprc,
                'precision': precision,
                'recall': recall
            },
            'Confidence Intervals': {
                'ROC AUC': {'Mean': np.mean(roc_aucs), '95% CI': np.percentile(roc_aucs, [2.5, 97.5])},
                'PRC AUC': {'Mean': np.mean(prc_aucs), '95% CI': np.percentile(prc_aucs, [2.5, 97.5])},
                'F1 Score': {'Mean': np.mean(f1_scores_list), '95% CI': np.percentile(f1_scores_list, [2.5, 97.5])}
            }
        }
    
    return results
    
def encode_texts(model_name, texts):
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
    print("embeddings are generated")
    return np.vstack(embeddings)

def encode_texts_biolm(model_name, texts):
    tokenizer = AutoTokenizer.from_pretrained("roberta-base", max_len=512)
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
    print("embeddings are generated")
    return np.vstack(embeddings)

def print_results(results):
    # Print results
    for antibiotic, res in results.items():
        print(f"Results for {antibiotic}:")

        # Calculate mean and confidence interval half-width for F1 score
        f1_mean = res['Confidence Intervals']['F1 Score']['Mean']
        f1_ci_lower = res['Confidence Intervals']['F1 Score']['95% CI'][0]
        f1_ci_upper = res['Confidence Intervals']['F1 Score']['95% CI'][1]
        f1_error = (f1_ci_upper - f1_ci_lower) / 2

        # Calculate mean and confidence interval half-width for ROC AUC
        roc_auc_mean = res['Confidence Intervals']['ROC AUC']['Mean']
        roc_auc_ci_lower = res['Confidence Intervals']['ROC AUC']['95% CI'][0]
        roc_auc_ci_upper = res['Confidence Intervals']['ROC AUC']['95% CI'][1]
        roc_auc_error = (roc_auc_ci_upper - roc_auc_ci_lower) / 2

        # Calculate mean and confidence interval half-width for PRC AUC
        prc_auc_mean = res['Confidence Intervals']['PRC AUC']['Mean']
        prc_auc_ci_lower = res['Confidence Intervals']['PRC AUC']['95% CI'][0]
        prc_auc_ci_upper = res['Confidence Intervals']['PRC AUC']['95% CI'][1]
        prc_auc_error = (prc_auc_ci_upper - prc_auc_ci_lower) / 2

        # Print the metrics with confidence intervals
        print(f"  Test - F1: {f1_mean:.4f} +/- {f1_error:.4f}, MCC: {res['Test Metrics']['Matthews Correlation Coefficient']:.4f}, "
              f"ROC-AUC: {roc_auc_mean:.4f} +/- {roc_auc_error:.4f}, PRC-AUC: {prc_auc_mean:.4f} +/- {prc_auc_error:.4f}")

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
        "BioMegatron": '#9467bd', "Medbert": "#d3ff0d",
        "Gatotron": "#c90076", "Bio+ClinicalBERT": "#660000", "sciBERT": "#16537e"
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
