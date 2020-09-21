import calendar
import os
import time
import json

from tensorflow import keras
import tensorflow as tf
import pickle
import argparse

from constants import PROJECT_ROOT

from tensorflow.python.lib.io import file_io
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np


def train(data_dir: str, epochs: str):
    # Training
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10)])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    with open(os.path.join(data_dir, 'train_images.pickle'), 'rb') as f:
        train_images = pickle.load(f)

    with open(os.path.join(data_dir, 'train_labels.pickle'), 'rb') as f:
        train_labels = pickle.load(f)

    model.fit(train_images, train_labels, epochs=int(epochs))

    with open(os.path.join(data_dir, 'test_images.pickle'), 'rb') as f:
        test_images = pickle.load(f)

    with open(os.path.join(data_dir, 'test_labels.pickle'), 'rb') as f:
        test_labels = pickle.load(f)

    # Evaluation
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

    print(f'Test Loss: {test_loss}')
    print(f'Test Acc: {test_acc}')

    # Save model
    ts = calendar.timegm(time.gmtime())
    model_path = os.path.join(PROJECT_ROOT, f'mnist-{ts}.h5')
    tf.saved_model.save(model, model_path)

    with open(os.path.join(PROJECT_ROOT, 'output.txt'), 'w') as f:
        f.write(model_path)
        print(f'Model written to: {model_path}')


    # Add pipeline metrics
    metrics = {
        'metrics': [{
            'name': 'accuracy',
            'numberValue':  float(test_acc),
            'format': "PERCENTAGE",
        },
        {
            'name': 'loss',
            'numberValue':  float(test_loss),
            'format': "PERCENTAGE",
        }]
    }
    with file_io.FileIO('/mlpipeline-metrics.json', 'w') as f:
        json.dump(metrics, f)
    
    # Add confusion matrix
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])
    predictions = np.argmax(probability_model.predict(test_images), axis=1)
    
              
    df = pd.DataFrame(
        test_labels.tolist(), columns=['target'])
    )
    df['predicted'] = predictions.tolist()

    vocab = list(df['target'].unique())
    cm = confusion_matrix(df['target'], df['predicted'], labels=vocab)
    data = []
    for target_index, target_row in enumerate(cm):
        for predicted_index, count in enumerate(target_row):
            data.append((vocab[target_index], vocab[predicted_index], count))
    
    df_cm = pd.DataFrame(data, columns=['target', 'predicted', 'count'])
    cm_file = os.path.join('/confusion_matrix.csv')
    with file_io.FileIO(cm_file, 'w') as f:
        df_cm.to_csv(f, columns=['target', 'predicted', 'count'], header=False, index=False)

    metadata = {
        'outputs' : [{
        'type': 'confusion_matrix',
        'format': 'csv',
        'schema': [
            {'name': 'target', 'type': 'CATEGORY'},
            {'name': 'predicted', 'type': 'CATEGORY'},
            {'name': 'count', 'type': 'NUMBER'},
        ],
        'source': cm_file,
        # Convert vocab to string because for bealean values we want "True|False" to match csv data.
        'labels': list(map(str, vocab)),
        }]
    }
    with file_io.FileIO('/mlpipeline-ui-metadata.json', 'w') as f:
        json.dump(metadata, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kubeflow FMNIST training script')
    parser.add_argument('--data_dir', help='path to images and labels.')
    parser.add_argument('--epochs', help='epochs')
    args = parser.parse_args()

    train(data_dir=args.data_dir, epochs=args.epochs)
