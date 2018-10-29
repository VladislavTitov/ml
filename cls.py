file1 = open("data/m+.txt", 'w', encoding='utf-8')
file2 = open("data/m.txt", 'r', encoding='utf-8')

for line in file2:
    newline = line[:-1] + ",0\n"
    file1.write(newline)

file1 = open("data/f+.txt", 'w', encoding='utf-8')
file2 = open("data/f.txt", 'r', encoding='utf-8')

for line in file2:
    newline = line[:-1] + ",1\n"
    file1.write(newline)

import numpy as np

features = np.empty((0, 2))
sample = np.empty((0, 2))
with open("data/m+.txt", 'r', encoding='utf-8') as file1:
    data = file1.readlines()
    for line in data:
        words = line.split(",")
        sample = np.append(sample, [[words[0], words[1][0]]], axis=0)

with open("data/f+.txt", 'r', encoding='utf-8') as file1:
    data = file1.readlines()
    for line in data:
        words = line.split(",")
        sample = np.append(sample, [[words[0], words[1][0]]], axis=0)

dataset = np.random.permutation(sample)


features = [(feat[-1], int(label)) for feat, label in dataset]


def fit(dataset):
    classes, freq = {}, {}
    for feats, label in dataset:
        if label not in classes:
            classes[label] = 0
        classes[label] += 1
        for feat in feats:
            if (label, feat) not in freq:
                freq[(label, feat)] = 0
            freq[(label, feat)] += 1
    for label, feat in freq:
        freq[(label, feat)] /= classes[label]
    for c in classes:
        classes[c] /= len(dataset)

    return classes, freq


classifier = fit(features)

import math


def classify(classifier, features):
    classes, freq = classifier
    return min(classes.keys(), key=lambda cls: -math.log(classes[cls]) + sum(
        -math.log(freq.get((cls, feat), 10 ** (-7))) for feat in features))


print(classify(classifier, "я"))


def get_features(sample):
    features = sample[0] + sample[-2] + sample[-1]
    return features.lower()


features = [([get_features(feat)], int(label)) for feat, label in dataset]

classifier = fit(features)
print(classify(classifier, [get_features("Эльвира")]))

