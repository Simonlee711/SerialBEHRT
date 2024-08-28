import numpy as np
from collections import defaultdict

# Store the data
data = {
    'Clindamycin': {
        'F1': {'Tabular': 0.7179, 'EHR-shot': 0.7719, 'Word2Vec': 0.7737, 'DistilBERT': 0.7786, 'SerialBEHRT': 0.7792},
        'ROC-AUC': {'Tabular': 0.6029, 'EHR-shot': 0.7664, 'Word2Vec': 0.7263, 'DistilBERT': 0.7443, 'SerialBEHRT': 0.7420},
        'PRC-AUC': {'Tabular': 0.6427, 'EHR-shot': 0.7859, 'Word2Vec': 0.7660, 'DistilBERT': 0.7684, 'SerialBEHRT': 0.7898}
    },
    'Erythromycin': {
        'F1': {'Tabular': 0.5495, 'EHR-shot': 0.6575, 'Word2Vec': 0.6394, 'DistilBERT': 0.6592, 'SerialBEHRT': 0.6517},
        'ROC-AUC': {'Tabular': 0.5879, 'EHR-shot': 0.7590, 'Word2Vec': 0.7320, 'DistilBERT': 0.7597, 'SerialBEHRT': 0.7557},
        'PRC-AUC': {'Tabular': 0.4530, 'EHR-shot': 0.6718, 'Word2Vec': 0.6754, 'DistilBERT': 0.6872, 'SerialBEHRT': 0.6886}
    },
    'Gentamicin': {
        'F1': {'Tabular': 0.9762, 'EHR-shot': 0.9775, 'Word2Vec': 0.9776, 'DistilBERT': 0.9766, 'SerialBEHRT': 0.9782},
        'ROC-AUC': {'Tabular': 0.6158, 'EHR-shot': 0.6310, 'Word2Vec': 0.6727, 'DistilBERT': 0.6777, 'SerialBEHRT': 0.5946},
        'PRC-AUC': {'Tabular': 0.9706, 'EHR-shot': 0.9672, 'Word2Vec': 0.9675, 'DistilBERT': 0.9713, 'SerialBEHRT': 0.9637}
    },
    'Levofloxacin': {
        'F1': {'Tabular': 0.7641, 'EHR-shot': 0.8386, 'Word2Vec': 0.8088, 'DistilBERT': 0.8034, 'SerialBEHRT': 0.8122},
        'ROC-AUC': {'Tabular': 0.6326, 'EHR-shot': 0.7972, 'Word2Vec': 0.7787, 'DistilBERT': 0.7974, 'SerialBEHRT': 0.8067},
        'PRC-AUC': {'Tabular': 0.7324, 'EHR-shot': 0.8290, 'Word2Vec': 0.8157, 'DistilBERT': 0.8459, 'SerialBEHRT': 0.8459}
    },
    'Oxacillin': {
        'F1': {'Tabular': 0.7264, 'EHR-shot': 0.8229, 'Word2Vec': 0.7899, 'DistilBERT': 0.7790, 'SerialBEHRT': 0.7935},
        'ROC-AUC': {'Tabular': 0.5607, 'EHR-shot': 0.7996, 'Word2Vec': 0.7688, 'DistilBERT': 0.7692, 'SerialBEHRT': 0.7785},
        'PRC-AUC': {'Tabular': 0.6069, 'EHR-shot': 0.8408, 'Word2Vec': 0.7847, 'DistilBERT': 0.7807, 'SerialBEHRT': 0.8073}
    },
    'Tetracycline': {
        'F1': {'Tabular': 0.8950, 'EHR-shot': 0.9009, 'Word2Vec': 0.9028, 'DistilBERT': 0.9035, 'SerialBEHRT': 0.9052},
        'ROC-AUC': {'Tabular': 0.5822, 'EHR-shot': 0.6908, 'Word2Vec': 0.6843, 'DistilBERT': 0.6843, 'SerialBEHRT': 0.6827},
        'PRC-AUC': {'Tabular': 0.8467, 'EHR-shot': 0.8571, 'Word2Vec': 0.8717, 'DistilBERT': 0.8760, 'SerialBEHRT': 0.8818}
    },
    'Trimethoprim/sulfa': {
        'F1': {'Tabular': 0.8835, 'EHR-shot': 0.8856, 'Word2Vec': 0.9080, 'DistilBERT': 0.9080, 'SerialBEHRT': 0.9087},
        'ROC-AUC': {'Tabular': 0.5393, 'EHR-shot': 0.7026, 'Word2Vec': 0.7025, 'DistilBERT': 0.6946, 'SerialBEHRT': 0.7227},
        'PRC-AUC': {'Tabular': 0.8159, 'EHR-shot': 0.8707, 'Word2Vec': 0.8748, 'DistilBERT': 0.8742, 'SerialBEHRT': 0.8938}
    },
    'Vancomycin': {
        'F1': {'Tabular': 0.6786, 'EHR-shot': 0.7201, 'Word2Vec': 0.7227, 'DistilBERT': 0.7244, 'SerialBEHRT': 0.7389},
        'ROC-AUC': {'Tabular': 0.5431, 'EHR-shot': 0.7566, 'Word2Vec': 0.7449, 'DistilBERT': 0.7542, 'SerialBEHRT': 0.7718},
        'PRC-AUC': {'Tabular': 0.5537, 'EHR-shot': 0.7781, 'Word2Vec': 0.7676, 'DistilBERT': 0.7754, 'SerialBEHRT': 0.7940}
    }
}

def compute_ranks(scores):
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ranks = {}
    for rank, (model, score) in enumerate(sorted_scores, 1):
        ranks[model] = rank
    return ranks

metrics = ['F1', 'ROC-AUC', 'PRC-AUC']
models = ['Tabular', 'EHR-shot', 'Word2Vec', 'DistilBERT', 'SerialBEHRT']

rank_sums = defaultdict(lambda: defaultdict(float))
rank_counts = defaultdict(lambda: defaultdict(int))

for antibiotic in data:
    for metric in metrics:
        ranks = compute_ranks(data[antibiotic][metric])
        for model in models:
            rank_sums[metric][model] += ranks[model]
            rank_counts[metric][model] += 1

average_ranks = {}
for metric in metrics:
    average_ranks[metric] = {model: rank_sums[metric][model] / rank_counts[metric][model] for model in models}

print("Average Ranks:")
for metric in metrics:
    print(f"\n{metric}:")
    sorted_ranks = sorted(average_ranks[metric].items(), key=lambda x: x[1])
    for model, rank in sorted_ranks:
        print(f"  {model}: {rank:.2f}")

overall_average_ranks = {model: np.mean([average_ranks[metric][model] for metric in metrics]) for model in models}

print("\nOverall Average Ranks:")
sorted_overall_ranks = sorted(overall_average_ranks.items(), key=lambda x: x[1])
for model, rank in sorted_overall_ranks:
    print(f"  {model}: {rank:.2f}")
