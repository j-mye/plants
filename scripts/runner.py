# Building a model
# --data - subdirectory of images for training
# --batch_size - batch size to use for training
# --epochs - amount of epochs to use for training
# --main_dir - where to save produced models, defaults to working directory
# --augment_data - boolean indication for whether to use data augmentation
# --fine_tune - boolean indication for whether to use fine tuning

# Note:
#     - directory arguments must not be followed by a '/'
#         Good: home/ad.msoe.edu/username
#         Bad: home/ad.msoe.edu/username/
    
# Example:
    # python runner.py --data /datasets --batch_size 32 --epochs 10 --main_dir ${WORKDIR} --augment_data false --fine_tune true

from model import model, train_ds, test_ds

import argparse
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='')
    parser.add_argument('--batch_size', type=str, default='')
    parser.add_argument('--epochs', type=str, default='')
    parser.add_argument('--main_dir', type=str, default='')
    parser.add_argument('--augment_data', type=str, default='')
    parser.add_argument('--fine_tune', type=str, default='')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    model.build_model(args.data, args.batch_size, 
                      args.epochs, args.main_dir, 
                      args.augment_data, args.fine_tune
                      )
    history = model.fit(train_ds,
                        validation_data = test_ds,
                        batch_size=int(args.batch_size),
                        epochs=int(args.epochs),
                        verbose=1)
    model.save(f"{args.main_dir}/out/plant_model.h5")
    
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.savefig(f"{args.main_dir}/training_history.png")
    plt.show()
