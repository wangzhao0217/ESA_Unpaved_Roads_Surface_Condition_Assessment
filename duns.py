# Standard Library Imports
import os
from itertools import cycle

# Third-Party Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    mean_squared_error,
    roc_auc_score,
    cohen_kappa_score,
    matthews_corrcoef,
    confusion_matrix
)
from sklearn.preprocessing import label_binarize
from pycaret.classification import load_model, predict_model


def calculate_metrics(predictions):
    y_true = predictions['Label']
    # Assuming 'prediction_score' contains prediction scores/probabilities for each class
    y_pred_score = predictions.get('prediction_score', predictions['prediction_label'])
    y_pred_label = predictions['prediction_label']  # Assuming this is the predicted class label

    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred_label)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred_label, average='weighted')

    # RMSE - Using scores if available, else use label as a proxy (not typical)
    if 'prediction_score' in predictions:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred_score))
    else:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred_label))

    # ROC-AUC for multi-class
    y_true_binarized = label_binarize(y_true, classes=np.unique(y_true))
    y_pred_binarized = label_binarize(y_pred_label, classes=np.unique(y_pred_label))
    roc_auc = roc_auc_score(y_true_binarized, y_pred_binarized, multi_class='ovr', average='weighted')
    
    # Kappa
    kappa = cohen_kappa_score(y_true, y_pred_label)

    # MCC
    mcc = matthews_corrcoef(y_true, y_pred_label)

    return {
        'Accuracy': accuracy,
        'AUC': roc_auc,
        'RMSE': rmse,
        'Recall': recall,
        'Precision': precision,
        'F1 Score': f1,
        'Kappa': kappa,
        'MCC': mcc
    }



def plot_confusion_matrix(y_true, y_pred_label, model_name=None, Name=None, cbar=True):
    # Calculate the confusion matrix
    cm = confusion_matrix(y_true, y_pred_label)
    
    # Normalize the confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Set up the matplotlib figure (with a specified figure size)
    plt.figure(figsize=(10, 7))
    
    combined_labels = np.unique(np.concatenate((y_true, y_pred_label)))
    n_classes = len(combined_labels)
    # Create a heatmap that displays both normalized and actual counts
    if n_classes  == 4:
        sns.heatmap(cm_normalized, annot=np.array([[f"{int(cm[i, j])}\n({cm_normalized[i, j]:.2f})" for j in range(len(cm))] for i in range(len(cm))]), 
                    fmt='', cmap='Blues', annot_kws={"size": 14}, cbar=cbar,
                    xticklabels=["Bad", "Poor", "Fair", "Good"], yticklabels=["Bad", "Poor", "Fair", "Good"])
    elif n_classes  == 2:
        sns.heatmap(cm_normalized, annot=np.array([[f"{int(cm[i, j])}\n({cm_normalized[i, j]:.2f})" for j in range(len(cm))] for i in range(len(cm))]), 
                    fmt='', cmap='Blues', annot_kws={"size": 14}, cbar=cbar,
                    xticklabels=["Bad/Poor", "Good/Fair"], yticklabels=["Bad/Poor", "Good/Fair"])
    # Set font sizes for tick labels
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    # Set labels and title with specified font sizes
    plt.xlabel('Predicted', fontsize=24)
    plt.ylabel('True', fontsize=24)
    plt.title(f'Confusion Matrix: {Name}', fontsize=24)
    
    # Save the figure to a file if model_name is provided
    if model_name:
        plt.savefig(f'./plot/{model_name}_confusion_matrix.png')
    
    # Adjust layout
    plt.tight_layout()
    
    # Display the plot
    plt.show()


def performance_check(model_path, data_path, Name=None, cbar=True):
    model = load_model(model_path)
    data = pd.read_csv(data_path)
    predictions = predict_model(model, data=data)
    # Placeholder for your actual function to calculate metrics
    metrics = calculate_metrics(predictions)
    metrics = {k: round(v, 3) for k, v in metrics.items()}

    print(f'Accuracy: {round(metrics["Accuracy"], 3) * 100}%')
    print(f'ROC-AUC Score: {round(metrics["AUC"], 3) * 100}%')
    print(f'RMSE: {round(metrics["RMSE"], 3)}')
    print(f'Recall: {round(metrics["Recall"], 3) * 100}%')
    print(f'Precision: {round(metrics["Precision"], 3) * 100}%')
    print(f'F1 Score: {round(metrics["F1 Score"], 3) * 100}%')
    print(f'Kappa: {round(metrics["Kappa"], 3)}')
    print(f'MCC: {round(metrics["MCC"], 3)}')

    # Plot confusion matrix
    plot_confusion_matrix(predictions['Label'], predictions['prediction_label'], model_name=os.path.basename(model_path), Name=Name, cbar=cbar) 