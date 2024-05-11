#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
import math

parser = argparse.ArgumentParser()

parser.add_argument("--classes", default=10, type=int, help="Počet tříd pro klasifikaci")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--epochs", default=50, type=int, help="Počet epoch na trénování Minibatch SGD")
parser.add_argument("--batch_size", default=10, type=int, help="Velikost batche")
parser.add_argument("--data_size", default=100, type=int, help="Velikost datasetu")
parser.add_argument("--test_size", default=0.5, type=float, help="Velikost testovací množiny")
parser.add_argument("--seed", default=42, type=int, help="Náhodný seed")

def softmax(predictions, classes):
    result = []
    for i in range(classes):
        summation = 0
        for j in range(classes):
            summation += math.exp(predictions[j])

        result.append(math.exp(predictions[i])/summation)
    
    return result

def main(args: argparse.Namespace):
    generator = np.random.RandomState(args.seed)

    data, target = sklearn.datasets.make_classification(
        n_samples=args.data_size, n_informative=args.classes,
        n_classes=args.classes, random_state=args.seed
    )

    data = np.concatenate([data, np.ones([args.data_size, 1])], axis=1)

    train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=args.test_size, random_state=args.seed)

    weights = generator.uniform(size=[args.classes, train_data.shape[1]], low=-0.1, high=0.1)

    for epoch in range(args.epochs):
        perm = generator.permutation(train_data.shape[0])

        batch_num = train_data.shape[0] // args.batch_size
        for batch in range(batch_num):
            batch_predictions = []
            batch_data = []
            batch_target = []
            for i in range(args.batch_size):
                p = perm[batch * args.batch_size + i]
                data = train_data[p]

                target = [0] * args.classes
                target[train_target[p]] = 1

                predictions = []
                for j in range(args.classes):
                    predictions.append(data.dot(weights[j]))
                predictions = softmax(predictions, args.classes)
                
                batch_predictions.append(predictions)
                batch_data.append(data)
                batch_target.append(target)

            batch_data = np.array(batch_data)
            batch_predictions = np.array(batch_predictions)
            batch_target = np.array(batch_target)

            gradient = np.matmul(np.transpose(batch_predictions-batch_target), batch_data)

            weights -= 1/args.batch_size * args.learning_rate * gradient

        train_predictions = []
        train_classified_predictions = []
        for data in train_data:
            predictions = []
            for j in range(args.classes):
                predictions.append(data.dot(weights[j]))
            predictions = softmax(predictions, args.classes)

            train_predictions.append(predictions)
            train_classified_predictions.append(np.argmax(predictions))
        test_predictions = []
        test_classified_predictions = []
        for data in test_data:
            predictions = []
            for j in range(args.classes):
                predictions.append(data.dot(weights[j]))
            predictions = softmax(predictions, args.classes)

            test_predictions.append(predictions)
            test_classified_predictions.append(np.argmax(predictions))
        train_accuracy, test_accuracy = accuracy_score(train_target, train_classified_predictions), accuracy_score(test_target, test_classified_predictions)

        train_log_loss, test_log_loss = log_loss(train_target, train_predictions), log_loss(test_target, test_predictions)

        print(f"Epoch {epoch+1}: train loss {train_log_loss:.6f} acc {train_accuracy*100:.2f}%, test loss {test_log_loss:.6f} acc {test_accuracy*100:.2f}%")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)