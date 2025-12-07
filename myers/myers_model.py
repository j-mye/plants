import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM
from tensorflow.keras import layers
import os
from load_images import create_image_generator

IMG_SIZE = 224
BATCH_SIZE = 32
train_dir = "datasets/dataset"
test_dir = "datasets/dataset-test"

class_names = sorted([d for d in os.listdir(train_dir) 
                      if os.path.isdir(os.path.join(train_dir, d))])
num_classes = len(class_names)
epochs = 20
batch_size = BATCH_SIZE

# Load the pre-trained MobileNetV2 model
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze the base model
base_model.trainable = False

# Adding LSTM on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = layers.Reshape((1, -1))(x)
x = LSTM(128)(x)  # LSTM layer
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Create generators
train_gen, train_samples = create_image_generator(
    train_dir, BATCH_SIZE, shuffle=True, augment=True
)
test_gen, test_samples = create_image_generator(
    test_dir, BATCH_SIZE, shuffle=False, augment=False
)

# Train the model using generators
history = model.fit(
    train_gen,
    steps_per_epoch=train_samples // BATCH_SIZE,
    epochs=epochs,
    validation_data=test_gen,
    validation_steps=test_samples // BATCH_SIZE,
    verbose=1
)

model_path = "comparison_model_lstm.h5"
model.save(model_path)
print(f"Model saved to {model_path}")
