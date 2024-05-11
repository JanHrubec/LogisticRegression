#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()

parser.add_argument("--classes", default=10, type=int, help="Počet epoch na trénování Minibatch SGD")
parser.add_argument("--data_size", default=100, type=int, help="Velikost datasetu")
parser.add_argument("--test_size", default=0.5, type=float, help="Velikost testovací množiny")
parser.add_argument("--seed", default=42, type=int, help="Náhodný seed")
parser.add_argument("--type", default="ovr", choices=["ovr", "ovo"], help="Typ klasifikátoru")
parser.add_argument("--test", action='store_true', help="Vypisovat testovací hlášky")

class OneVsRest:
    def __init__(self, classes, seed, test):
        self.classes = classes
        self.models = None
        self.seed = seed
        self.test = test

    def fit(self, data, target):
        self.models = [LogisticRegression(random_state=self.seed) for _ in range(self.classes)]

        for class_idx in range(self.classes):
            temp_target = np.copy(target)
            temp_target[temp_target != class_idx] = -1
            self.models[class_idx].fit(data, temp_target)

    def predict(self, data):
        predictions = []

        for idx, dato in enumerate(data):
            dato = dato[np.newaxis, :]

            probas = [0] * self.classes
            for model in self.models:

                probas[model.classes_[1]] = model.predict_proba(dato)[0][1]

            if self.test and idx < 5:
                print(" ".join(map(lambda x: f"{x:.4f}", probas)))

            predictions.append(np.argmax(probas))

        return predictions

class OneVsOne:
    def __init__(self, classes, seed, test):
        self.classes = classes
        self.models = None
        self.seed = seed
        self.test = test

    def fit(self, data, target):
        self.models = [LogisticRegression(random_state=self.seed) for _ in range(self.classes*(self.classes-1)//2)]
        i = 0
        for class1 in range(self.classes):
            for class2 in range(class1 + 1, self.classes):
                temp_target = target[(target==class1) + (target==class2)]
                temp_data = data[(target==class1) + (target==class2)]
                self.models[i].fit(temp_data, temp_target)
                i += 1

    def predict(self, data):
        predictions = []
        for idx, dato in enumerate(data):
            dato = dato[np.newaxis, :]

            votes = [0] * self.classes
            for model in self.models:
                votes[model.predict(dato)[0]] += 1

            if self.test and idx < 5:
                print(" ".join(map(str, votes)))

            predictions.append(np.argmax(votes))

        return predictions


def main(args: argparse.Namespace):
    generator = np.random.RandomState(args.seed)

    data, target = sklearn.datasets.make_classification(
        n_samples=args.data_size, n_informative=args.classes,
        n_classes=args.classes, random_state=args.seed
    )
    
    train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=args.test_size, random_state=args.seed)

    if args.type == "ovr":
        classifier = OneVsRest(args.classes, args.seed, args.test)
    else:
        classifier = OneVsOne(args.classes, args.seed, args.test)

    classifier.fit(train_data, train_target)
    predictions = classifier.predict(test_data)

    if args.test:
        print("Predikované třídy pro první 10 dat:")
        print(" ".join(map(str, predictions[:10])))


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
