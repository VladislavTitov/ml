dataset = [
        [['Зеленый', 3], 'Яблоко'],
        [['Желтый', 3], 'Яблоко'],
        [['Красный', 3], 'Виноград'],
        [['Красный', 3], 'Виноград'],
        [['Желтый', 3], 'Лимон'],
]

header = [['цвет', 'размер'], 'метка']

def unique_vals(rows, col):
    return set([row[0][col] for row in rows])

print(unique_vals(dataset, 0))

def class_counts(rows):
    counts = {}
    for row in rows:
        label = row[1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

print(class_counts(dataset))

def is_numeric(value):
    return isinstance(value, int) or isinstance(value, float)

class Question:
    def __init__( self, column, value ):
        self.column = column
        self.value = value

    def match(self, example):
        val = example[self.column]
        if is_numeric(val):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):
        condition = "=="
        if is_numeric(self.value):
            condition = ">="
        return "{} {} {}?".format(header[0][self.column], condition, str(self.value))

q = Question(0, 'Зеленый')
example = dataset[0][0]
q.match(example)

def partition(rows, question):
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row[0]):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows

print(partition(dataset, Question(0, 'Красный')))

def gini(rows):
    counts = class_counts(rows)
    impurity = 1
    for label in counts:
        prob_label = counts[label] / float(len(rows))
        impurity -= prob_label ** 2
    return impurity

print(gini(dataset))

def info_gain(left, right, current):
    p = float(len(left)) / (len(left) + len(right))
    return current - p * gini(left) - (1-p)*gini(right)

current = gini(dataset)
true_rows, false_rows = partition(dataset, Question(0, 'Зеленый'))
print(info_gain(true_rows, false_rows, current))

def find_best_split(rows):
    best_gain = 0
    best_question = None
    current = gini(rows)
    n_features = len(rows[0][0])
    for col in range(n_features):
        values = set([row[0][col] for row in rows])

        for val in values:
            question = Question(col, val)
            true_rows, false_rows = partition(rows, question)
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue
            gain = info_gain(true_rows, false_rows, current)

            if gain >= best_gain:
                best_gain, best_question = gain, question
    return best_gain, best_question

print(find_best_split(dataset))


