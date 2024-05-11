#!/usr/bin/env python3
import argparse
import numpy as np
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

parser = argparse.ArgumentParser()

parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--epochs", default=50, type=int, help="Počet epoch na trénování Minibatch SGD")
parser.add_argument("--batch_size", default=10, type=int, help="Velikost batche")
parser.add_argument("--data_size", default=100, type=int, help="Velikost datasetu")
parser.add_argument("--test_size", default=0.5, type=float, help="Velikost testovací množiny")
parser.add_argument("--seed", default=42, type=int, help="Náhodný seed")

def main(args: argparse.Namespace):
    generator = np.random.RandomState(args.seed)

    data, target = sklearn.datasets.make_classification(n_samples=args.data_size, random_state=args.seed)

    data = np.concatenate([data, np.ones([args.data_size, 1])], axis=1)

    train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=args.test_size, random_state=args.seed)

    weights = generator.uniform(size=train_data.shape[1], low=-0.1, high=0.1)

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
                target = train_target[p]
                prediction = sigmoid(data.dot(weights))
                
                batch_predictions.append(prediction)
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
            prediction = sigmoid(data.dot(weights))
            train_predictions.append(prediction)
            train_classified_predictions.append(1 if prediction >= 0.5 else 0)
        test_predictions = []
        test_classified_predictions = []
        for data in test_data:
            prediction = sigmoid(data.dot(weights))
            test_predictions.append(prediction)
            test_classified_predictions.append(1 if prediction >= 0.5 else 0)
        train_accuracy, test_accuracy = accuracy_score(train_target, train_classified_predictions), accuracy_score(test_target, test_classified_predictions)

        train_log_loss, test_log_loss = log_loss(train_target, train_predictions), log_loss(test_target, test_predictions)

        print(f"Epoch {epoch+1}: train loss {train_log_loss:.6f} acc {train_accuracy*100:.2f}, test loss {test_log_loss:.6f} acc {test_accuracy*100:.2f}")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
