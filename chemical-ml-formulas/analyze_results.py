from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def calculate_metrics(predictions, actuals):
    # Calculate exact match accuracy
    accuracy = sum(p == a for p, a in zip(predictions, actuals)) / len(predictions)
    
    # Calculate character-level metrics
    char_predictions = [[c for c in p] for p in predictions]
    char_actuals = [[c for c in a] for a in actuals]
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        char_actuals, char_predictions, average='weighted'
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Calculate metrics
metrics = calculate_metrics(predictions, actuals) # type: ignore
print("\nModel Performance:")
for metric, value in metrics.items():
    print(f"{metric.capitalize()}: {value:.4f}")