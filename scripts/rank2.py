import numpy as np

# Data provided for the table
data = {
    "Antibiotic": [
        "Clindamycin", "Clindamycin", "Clindamycin",
        "Erythromycin", "Erythromycin", "Erythromycin",
        "Gentamicin", "Gentamicin", "Gentamicin",
        "Levofloxacin", "Levofloxacin", "Levofloxacin",
        "Oxacillin", "Oxacillin", "Oxacillin",
        "Tetracycline", "Tetracycline", "Tetracycline",
        "Trimethoprim/Sulfa", "Trimethoprim/Sulfa", "Trimethoprim/Sulfa",
        "Vancomycin", "Vancomycin", "Vancomycin"
    ],
    "Metric": [
        "F1", "ROC-AUC", "PRC-AUC",
        "F1", "ROC-AUC", "PRC-AUC",
        "F1", "ROC-AUC", "PRC-AUC",
        "F1", "ROC-AUC", "PRC-AUC",
        "F1", "ROC-AUC", "PRC-AUC",
        "F1", "ROC-AUC", "PRC-AUC",
        "F1", "ROC-AUC", "PRC-AUC",
        "F1", "ROC-AUC", "PRC-AUC"
    ],
    "Tabular": [
        0.7179, 0.6029, 0.6427,
        0.5495, 0.5879, 0.4530,
        0.9762, 0.6158, 0.9706,
        0.7641, 0.6326, 0.7324,
        0.7264, 0.5607, 0.6069,
        0.8950, 0.5822, 0.8467,
        0.8835, 0.5393, 0.8159,
        0.6786, 0.5431, 0.5537
    ],
    "EHR-shot": [
        0.7719, 0.7664, 0.7859,
        0.6575, 0.7590, 0.6718,
        0.9775, 0.6310, 0.9672,
        0.8386, 0.7972, 0.8290,
        0.8229, 0.7996, 0.8408,
        0.9009, 0.6908, 0.8571,
        0.8856, 0.7026, 0.8707,
        0.7201, 0.7566, 0.7781
    ],
    "Word2Vec": [
        0.7737, 0.7263, 0.7660,
        0.6394, 0.7320, 0.6754,
        0.9776, 0.6727, 0.9675,
        0.8088, 0.7787, 0.8157,
        0.7899, 0.7688, 0.7847,
        0.9028, 0.6843, 0.8717,
        0.9080, 0.7025, 0.8748,
        0.7227, 0.7449, 0.7676
    ],
    "DistilBERT": [
        0.7786, 0.7443, 0.7684,
        0.6592, 0.7597, 0.6872,
        0.9766, 0.6777, 0.9713,
        0.8034, 0.7974, 0.8459,
        0.7790, 0.7692, 0.7807,
        0.9035, 0.6843, 0.8760,
        0.9080, 0.6946, 0.8742,
        0.7244, 0.7542, 0.7754
    ],
    "SerialBERT": [
        0.7792, 0.7420, 0.7898,
        0.6517, 0.7557, 0.6886,
        0.9782, 0.5946, 0.9637,
        0.8122, 0.8067, 0.8459,
        0.7935, 0.7785, 0.8073,
        0.9052, 0.6827, 0.8818,
        0.9087, 0.7227, 0.8938,
        0.7389, 0.7718, 0.7940
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Calculate ranks for each row
df_rank = df.iloc[:, 2:].rank(axis=1, ascending=False)

# Calculate the average rank for each model across all metrics
average_ranks = df_rank.mean().to_frame(name='Average Rank')
average_ranks
