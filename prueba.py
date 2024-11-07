import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
tf.random.set_seed(42)

class SEBlock(layers.Layer):
    def __init__(self, filters, reduction=16, **kwargs):
        super().__init__(**kwargs)
        self.global_avg_pool = layers.GlobalAveragePooling2D()
        self.dense1 = layers.Dense(filters // reduction, activation="relu")
        self.dense2 = layers.Dense(filters, activation="sigmoid")

    def call(self, inputs):
        se = self.global_avg_pool(inputs)
        se = self.dense1(se)
        se = self.dense2(se)
        se = layers.Reshape((1, 1, -1))(se)
        return layers.multiply([inputs, se])

class ResidualUnit(keras.layers.Layer):
    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = keras.activations.get(activation)
        self.main_layers = [
            keras.layers.SeparableConv2D(filters, 3, strides=strides, padding="same", use_bias=False),
            keras.layers.BatchNormalization(),
            self.activation,
            keras.layers.SeparableConv2D(filters, 3, strides=1, padding="same", use_bias=False),
            keras.layers.BatchNormalization()
        ]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                keras.layers.SeparableConv2D(filters, 1, strides=strides, padding="same", use_bias=False),
                keras.layers.BatchNormalization()
            ]
        
        self.se_block = SEBlock(filters)

    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)

# Define model architecture with custom residual units and adjusted input shape
model = keras.models.Sequential()
model.add(keras.layers.Input(shape=(600, 600, 3)))  # Adjust to (600, 600, 3)

# Data Augmentation and Preprocessing
model.add(keras.layers.RandomFlip("horizontal"))
model.add(keras.layers.RandomRotation(0.1))
model.add(keras.layers.RandomZoom(0.2))
model.add(keras.layers.Rescaling(1./255))

# Initial Conv Block
model.add(keras.layers.Conv2D(64, 7, strides=2, padding="same", use_bias=False))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation("relu"))
model.add(keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same"))

# Add Residual Blocks
prev_filters = 64
for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
    strides = 1 if filters == prev_filters else 2
    model.add(ResidualUnit(filters, strides=strides))
    prev_filters = filters

# Global Average Pooling and Output Layer
model.add(keras.layers.GlobalAvgPool2D())
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(1, activation="sigmoid"))

# Compile the model
model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.AdamW(learning_rate=1e-3), metrics=["accuracy"])

# Load the dataset and split into training and validation
from keras.utils import image_dataset_from_directory

# Directory where the two types of pistachios are stored
dataset_directory = "DataSet"  # Change to your dataset path

# Automatically split into training and validation sets
train_dataset = image_dataset_from_directory(
    dataset_directory,
    image_size=(600, 600),  # Match to your image size
    batch_size=32,
    validation_split=0.2,
    subset="training",
    seed=35
)

validation_dataset = image_dataset_from_directory(
    dataset_directory,
    image_size=(600, 600),
    batch_size=32,
    validation_split=0.2,
    subset="validation",
    seed=35
)

# Train the model
callbacks = [keras.callbacks.ModelCheckpoint("resnet_pistaches.keras", save_best_only=True, monitor="val_loss"),
             keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, min_lr=1e-5)]

history = model.fit(train_dataset, epochs=30, validation_data=validation_dataset, callbacks=callbacks)

# Plot training results
plt.figure(figsize=(16, 4))
plt.subplot(121)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.subplot(122)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Evaluate on validation set as test
test_loss, test_acc = model.evaluate(validation_dataset)
print(f"Test accuracy: {test_acc:.3f}")
