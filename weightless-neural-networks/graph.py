import sys
import pandas as pd
import matplotlib.pyplot as plt
from library import get_path


def plot_line(df, column_x, xlabel, ylabel, title, filename):
    # Todo: Plot a line between executions
    plt.figure(figsize=(10, 7))
    plt.title(title)
    plt.ylim(0.5, 1.01)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.errorbar(df[column_x], df['train_accuracy_mean'], df['train_accuracy_std'],
                 linestyle='None', marker='^', color='red', label='Treinamento')
    plt.errorbar(df[column_x], df['validation_accuracy_mean'], df['validation_accuracy_std'],
                 linestyle='None', marker='^', color='green', label='Validaao')
    plt.errorbar(df[column_x], df['test_accuracy_mean'], df['test_accuracy_std'],
                 linestyle='None', marker='^', color='blue', label='Teste')
    plt.legend(loc='lower left')
    plt.savefig(get_path(folder='results', subfolder='graph') + '/' + '{}.jpg'.format(filename))


def main():
    # Todo: Change labels
    df = pd.read_csv(get_path(folder='results', file_name='accuracy.csv'))
    thresholds = df['threshold'].unique()
    for threshold in thresholds:
        print("Generating graph for threshold " + str(threshold))
        df_threshold = df[df['threshold'] == threshold]
        column_x = 'addressSize'
        xlabel = 'Numero de bits de enderecamento'
        ylabel = 'Acuracia'
        title = 'Resultados variando o numero de bits de enderecamento e threshold=' + str(threshold)
        filename = 'threshold_' + str(threshold)
        plot_line(df_threshold, column_x, xlabel, ylabel, title, filename)


if __name__ == "__main__":
    sys.exit(main())
