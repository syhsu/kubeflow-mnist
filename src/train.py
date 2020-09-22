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

def topk_accuracy(y_true, y_prob, n=5, normalize=True):
    num_obs, num_labels = y_prob.shape
    idx = num_labels - n - 1
    counter = 0
    argsorted = np.argsort(y_prob, axis=1)
    for i in range(num_obs):
        if y_true[i] in argsorted[i, idx+1:]:
            counter += 1
    
    if normalize:
        return counter * 1.0 / num_obs
    else:
        return counter


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

    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    y_predictions = probability_model.predict(test_images)
    y_pred = np.argmax(y_predictions, axis=1)
    topkAccuracy = topk_accuracy(test_labels, y_predictions)


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
            'format': "RAW",
        },
        #{
        #    'name': 'top5',
        #    'numberValue':  float(topkAccuracy),
        #    'format': "PERCENTAGE",
        #}
        ]
    }
    with file_io.FileIO('/mlpipeline-metrics.json', 'w') as f:
        json.dump(metrics, f)
    
    # Add confusion matrix
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    df = pd.DataFrame(
        test_labels.tolist(), columns=['target']
    )
    df['predicted'] = y_pred.tolist()

    vocab = sorted(list(df['target'].unique()), reverse=True)
    cm = confusion_matrix(df['target'], df['predicted'], labels=vocab)
    data = []
    for target_index, target_row in enumerate(cm):
        for predicted_index, count in enumerate(target_row):
            data.append((vocab[target_index], vocab[predicted_index], count))
    
    df_cm = pd.DataFrame(data, columns=['target', 'predicted', 'count'])
    df_cm = df_cm.sort_values(by='target', ascending=True)
    cm_file = '/confusion_matrix.csv'
    with file_io.FileIO(cm_file, 'w') as f:
        df_cm.to_csv(f, columns=['target', 'predicted', 'count'], header=False, index=False)
    
    rawCsv = ''
    for line in open(cm_file):
        rawCsv = rawCsv + line
    print(rawCsv)

    metadata = {
        'outputs' : [{
        'type': 'confusion_matrix',
        'format': 'csv',
        'storage': 'inline',
        'schema': [
            {'name': 'target', 'type': 'CATEGORY'},
            {'name': 'predicted', 'type': 'CATEGORY'},
            {'name': 'count', 'type': 'NUMBER'},
        ],
        'source': rawCsv,
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
