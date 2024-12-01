from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Function to evaluate and display results
def evaluate_models(model_a_outputs, model_b_outputs, ground_truth):
    # Accuracy scores
    acc_model_a = accuracy_score(ground_truth, model_a_outputs)
    acc_model_b = accuracy_score(ground_truth, model_b_outputs)
    
    # Classification reports
    report_model_a = classification_report(ground_truth, model_a_outputs, target_names=["No", "Yes"])
    report_model_b = classification_report(ground_truth, model_b_outputs, target_names=["No", "Yes"])
    
    # Confusion matrices
    cm_model_a = confusion_matrix(ground_truth, model_a_outputs)
    cm_model_b = confusion_matrix(ground_truth, model_b_outputs)
    
    # Display results
    print("Model A Evaluation:")
    print(f"Accuracy: {acc_model_a:.2f}")
    print("Classification Report:")
    print(report_model_a)
    
    print("\nModel B Evaluation:")
    print(f"Accuracy: {acc_model_b:.2f}")
    print("Classification Report:")
    print(report_model_b)
    
    # Plot confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(cm_model_a, annot=True, fmt="d", cmap="Blues", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"], ax=axes[0])
    axes[0].set_title("Model A Confusion Matrix")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    
    sns.heatmap(cm_model_b, annot=True, fmt="d", cmap="Blues", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"], ax=axes[1])
    axes[1].set_title("Model B Confusion Matrix")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")
    
    plt.tight_layout()
    plt.show()

# Example Inputs
ground_truth = ["Yes", "No", "Yes"]
model_a_outputs = ["Yes", "No", "Yes"]
model_b_outputs = ["Yes", "No", "No"]

# Evaluate Models
evaluate_models(model_a_outputs, model_b_outputs, ground_truth)
