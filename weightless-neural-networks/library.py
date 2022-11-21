import os
import sys
import numpy as np
import pandas as pd
import seaborn as sn
from matplotlib import pyplot as plt


def get_path(folder, subfolder=None, file_name=None):
    base_path = os.path.dirname(os.getcwd())
    if file_name is None:
        return os.path.join(base_path, folder, subfolder)
    if subfolder is None:
        return os.path.join(base_path, folder, file_name)
    else:
        return os.path.join(base_path, folder, subfolder, file_name)


def load(f):
    return np.load(f)['arr_0']


def preprocess(df, threshold):
    columns = df.columns
    for column in columns:
        df[column] = np.where(df[column] >= threshold, 1, 0)
    return df


def cm2df(cm, labels):
    df = pd.DataFrame()
    # rows
    for i, row_label in enumerate(labels):
        rowdata = {}
        # columns
        for j, col_label in enumerate(labels):
            rowdata[col_label] = cm[i, j]
        df = df.append(pd.DataFrame.from_dict({row_label: rowdata}, orient='index'))
    return df


def save_accuracy(threshold, addressSize, train_accuracy_score, validation_accuracy_score, test_accuracy_score):
    train_accuracy_score_mean = np.mean(train_accuracy_score, axis=0)
    validation_accuracy_score_mean = np.mean(validation_accuracy_score, axis=0)
    test_accuracy_score_mean = np.mean(test_accuracy_score, axis=0)
    train_accuracy_score_std = np.std(train_accuracy_score)
    validation_accuracy_score_std = np.std(validation_accuracy_score)
    test_accuracy_score_std = np.std(test_accuracy_score)

    matrix = {'threshold': [threshold], 'addressSize': [addressSize],
              'train_accuracy_mean': [train_accuracy_score_mean],
              'validation_accuracy_mean': [validation_accuracy_score_mean],
              'test_accuracy_mean': [test_accuracy_score_mean], 'train_accuracy_std': [train_accuracy_score_std],
              'validation_accuracy_std': [validation_accuracy_score_std],
              'test_accuracy_std': [test_accuracy_score_std]}

    result = pd.DataFrame(matrix)
    with open(get_path(folder='results', file_name='accuracy.csv'), 'a') as file:
        result.to_csv(file, index=False, header=True)


def save_matrix(cm, filename):
    df = cm2df(cm, range(10))
    with open(filename, 'a') as file:
        df.to_csv(file)


def plot_heatmap(cm, title, filename):
    df_cm = pd.DataFrame(cm, index=range(10), columns=range(10))
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, cmap="YlGnBu", annot=True)
    plt.title(title)
    plt.ylabel('Target Value')
    plt.xlabel('Predicted Value')
    plt.savefig(get_path(folder='results', file_name=filename + '.jpg'))


def load_dataset(threshold: int):
    x_train = load(get_path(folder='data', file_name='kmnist-train-imgs.npz'))
    x_test = load(get_path(folder='data', file_name='kmnist-test-imgs.npz'))
    y_train = load(get_path(folder='data', file_name='kmnist-train-labels.npz'))
    y_test = load(get_path(folder='data', file_name='kmnist-test-labels.npz'))
    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)
    y_train = y_train.astype(str)
    y_test = y_test.astype(str)
    x_train = binary_encoder(x_train, threshold)
    x_test = binary_encoder(x_test, threshold)
    return x_train, x_test, y_train, y_test


def sample_digit(target, X, y):
    return next((digit for (digit, label) in zip(X, y) if label == np.array(str(target)))).reshape((28, 28))


def display_mnist_digits():
    fig, axs = plt.subplots(2, 5)
    for i in range(2):
        for j in range(5):
            axs[i, j].imshow(sample_digit(5 * i + j), cmap="gray")
            axs[i, j].axes.xaxis.set_visible(False)
            axs[i, j].axes.yaxis.set_visible(False)

    return fig


def binarize(image, threshold):
    return np.where(image > threshold, 1, 0).tolist()


def binary_encoder(images, threshold=127):
    return [binarize(image, threshold) for image in images]


def build_report(threshold, address_size,
                 confusion_matrix_test_scores,
                 confusion_matrix_train_scores, confusion_matrix_validation_scores):
    path = get_path(folder='results', subfolder='threshold_') + \
           str(threshold) + '/' + 'addressSize_' + str(address_size)
    if not os.path.exists(path):
        os.makedirs(path)
    plot_heatmap(confusion_matrix_train_scores, title='Training', filename=path + '/training_heatmap')
    plot_heatmap(confusion_matrix_validation_scores, title='Validation', filename=path + '/validation_heatmap')
    plot_heatmap(confusion_matrix_test_scores, title='Test', filename=path + '/test_heatmap')
    save_matrix(confusion_matrix_train_scores, path + '/training_confusion_matrix.csv')
    save_matrix(confusion_matrix_validation_scores, path + '/validation_confusion_matrix.csv')
    save_matrix(confusion_matrix_test_scores, path + '/test_confusion_matrix.csv')


class WizardParam:
    def __init__(self):
        self.threshold = None
        self.addressSize = None
        self.bleachingActivated = None
        self.ignoreZero = None
        self.completeAddressing = None
        self.verbose = None
        self.returnActivationDegree = None
        self.returnConfidence = None
        self.returnClassesDegrees = None

    def get_param(self):
        return self.__dict__


if __name__ == "__main__":
    sys.exit()
