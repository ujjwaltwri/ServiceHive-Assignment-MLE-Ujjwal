import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

print("--- Loading Model and Preparing Test Data ---")

# Load the saved model
model = tf.keras.models.load_model('models/efficientnet_v2.keras')

# If you have a separate test directory, use this:
test_dir = 'data/archive/seg_test/seg_test'  # Update path as needed

# Or use validation data (if no separate test set)
train_dir = 'data/archive/seg_train/seg_train'

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

try:
    # Option 1: Load test dataset (if available)
    test_dataset = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False  # Important: don't shuffle for confusion matrix
    )
    print("Using separate test dataset")
except:
    print("Test directory not found, using validation split from training data")
    # Option 2: Use validation split if no test set
    test_dataset = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False  # Important: don't shuffle for confusion matrix
    )

class_names = test_dataset.class_names
print(f"Classes: {class_names}")

# Prepare test dataset (no augmentation, just preprocessing)
test_ds = test_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

print("--- Making Predictions ---")

# Get predictions
predictions = model.predict(test_ds)
predicted_classes = np.argmax(predictions, axis=1)

# Get true labels
true_labels = []
for images, labels in test_ds:
    true_labels.extend(labels.numpy())

true_labels = np.array(true_labels)

print("--- Generating Confusion Matrix ---")

# Create confusion matrix
cm = confusion_matrix(true_labels, predicted_classes)

# Calculate accuracy
accuracy = np.sum(predicted_classes == true_labels) / len(true_labels)
print(f"Test Accuracy: {accuracy:.4f}")

# --- Visualization ---

# Create figure with subplots
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Plot 1: Confusion Matrix (counts)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names,
            ax=axes[0])
axes[0].set_title('Confusion Matrix (Counts)', fontsize=16, fontweight='bold')
axes[0].set_xlabel('Predicted Labels', fontsize=12)
axes[0].set_ylabel('True Labels', fontsize=12)
axes[0].tick_params(axis='x', rotation=45)
axes[0].tick_params(axis='y', rotation=0)

# Plot 2: Normalized Confusion Matrix (percentages)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            ax=axes[1])
axes[1].set_title('Normalized Confusion Matrix (Percentages)', fontsize=16, fontweight='bold')
axes[1].set_xlabel('Predicted Labels', fontsize=12)
axes[1].set_ylabel('True Labels', fontsize=12)
axes[1].tick_params(axis='x', rotation=45)
axes[1].tick_params(axis='y', rotation=0)

plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# --- Classification Report ---
print("\n--- Detailed Classification Report ---")
report = classification_report(true_labels, predicted_classes, 
                             target_names=class_names, digits=4)
print(report)

# --- Per-Class Accuracy ---
print("\n--- Per-Class Accuracy ---")
class_accuracies = cm.diagonal() / cm.sum(axis=1)
for i, class_name in enumerate(class_names):
    print(f"{class_name}: {class_accuracies[i]:.4f}")

# --- Additional Visualization: Per-Class Accuracy Bar Plot ---
plt.figure(figsize=(12, 6))
bars = plt.bar(class_names, class_accuracies, color='skyblue', edgecolor='navy', alpha=0.7)
plt.title('Per-Class Accuracy', fontsize=16, fontweight='bold')
plt.xlabel('Classes', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 1)

# Add value labels on bars
for bar, acc in zip(bars, class_accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('per_class_accuracy.png', dpi=300, bbox_inches='tight')
plt.show()

# --- Save Results ---
print("\n--- Saving Results ---")

# Save confusion matrix as CSV
import pandas as pd
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
cm_df.to_csv('confusion_matrix.csv')

# Save classification report to file
with open('classification_report.txt', 'w') as f:
    f.write(f"Test Accuracy: {accuracy:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)
    f.write("\n\nPer-Class Accuracy:\n")
    for i, class_name in enumerate(class_names):
        f.write(f"{class_name}: {class_accuracies[i]:.4f}\n")

print("Results saved:")
print("- confusion_matrix.png")
print("- per_class_accuracy.png") 
print("- confusion_matrix.csv")
print("- classification_report.txt")
print("\n--- Analysis Complete ---")