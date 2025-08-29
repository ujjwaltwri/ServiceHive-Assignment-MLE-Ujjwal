import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import sys

print("--- Starting Final Model Training ---")

# --- Data Preparation ---
train_dir = 'data/archive/seg_train/seg_train'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

try:
    # Create training and validation datasets
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        train_dir, validation_split=0.2, subset="training", seed=123,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE)
    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        train_dir, validation_split=0.2, subset="validation", seed=123,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE)
    class_names = train_dataset.class_names
    print(f"Found classes: {class_names}")
except Exception as e:
    print(f"Error loading data: {e}. Please ensure the dataset is unzipped correctly in the 'data' folder.")
    sys.exit(1)

# --- Data Augmentation Layer ---
# This applies random transformations to the training images to make the model more robust.
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
], name="data_augmentation")

# --- Preprocessing Pipeline ---
def prepare(ds, is_train=True):
    """Applies augmentation to the training set and optimizes data loading."""
    if is_train:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    # Use buffered prefetching to load data efficiently
    return ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

train_ds = prepare(train_dataset, is_train=True)
val_ds = prepare(validation_dataset, is_train=False)
print("--- Data preparation complete ---")

# --- Build the EfficientNetV2B0 Model ---
# Load the pre-trained base model without its top classification layer
base_model = tf.keras.applications.EfficientNetV2B0(
    input_shape=(224, 224, 3), include_top=False, weights='imagenet')
# Freeze the base model to preserve its learned weights
base_model.trainable = False

# Create a new classification head
inputs = layers.Input(shape=(224, 224, 3))
# EfficientNet has its own normalization, so we don't need a Rescaling layer
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x) # Helps stabilize training
x = layers.Dropout(0.3)(x) # Regularization to prevent overfitting
outputs = layers.Dense(len(class_names), activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)
print("--- Model built successfully ---")

# --- Compile and Train ---
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

# This callback stops training when validation accuracy stops improving, ensuring we get the best model
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

print("--- Starting training... ---")
model.fit(train_ds, epochs=25, validation_data=val_ds, callbacks=[early_stopping])

# --- SAVE THE MODEL ---
model.save('models/efficientnet_v2.keras')
print("\n--- FINAL MODEL SAVED SUCCESSFULLY to models/efficientnet_v2.keras ---")