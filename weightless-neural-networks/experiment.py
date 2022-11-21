import os
import numpy as np
import pandas as pd
from pandas import compat
from library import load_dataset, save_accuracy, WizardParam, build_report
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score

compat.PY3 = True
pd.options.display.float_format = '{:.2f}'.format


def run(wisard_model, wisard_param: WizardParam, n_splits=None):
    confusion_matrix_train_scores = np.zeros((10, 10))
    confusion_matrix_validation_scores = np.zeros((10, 10))
    confusion_matrix_test_scores = np.zeros((10, 10))

    train_accuracy_score = []
    validation_accuracy_score = []
    test_accuracy_score = []

    # Define model
    wsd = wisard_model

    # Load data
    X_train, X_test, Y_train, Y_test = load_dataset(wisard_param.threshold)

    if n_splits:
        kf = KFold(n_splits=n_splits, shuffle=True)
        fold = 1
        for train_index, val_index in kf.split(X_train):
            print("FOLD:", fold)
            print("TRAIN: {} VALIDATION: {}".format(len(train_index), len(val_index)))
            x_train, x_val = [X_train[index] for index in train_index], [X_train[index] for index in val_index]
            y_train, y_val = [str(Y_train[index]) for index in train_index], [str(Y_train[index]) for index in val_index]

            # Train using the input data
            print("Training")
            wsd.train(x_train, y_train)

            # classify train data
            print("Train data classification")
            out_train = wsd.classify(x_train)

            cm_training = confusion_matrix(y_train, out_train)
            cm_training = cm_training / cm_training.astype(np.float).sum(axis=1)
            confusion_matrix_train_scores += cm_training
            train_accuracy_score.append(accuracy_score(y_train, out_train))

            # classify validation data
            print("Validation data classification")
            out_val = wsd.classify(x_val)

            cm_validation = confusion_matrix(y_val, out_val)
            cm_validation = cm_validation / cm_validation.astype(np.float).sum(axis=1)
            confusion_matrix_validation_scores += cm_validation
            validation_accuracy_score.append(accuracy_score(y_val, out_val))

            # classify test data
            print("Test data classification")
            out_test = wsd.classify(X_test)

            cm_test = confusion_matrix(Y_test, out_test)
            cm_test = cm_test / cm_test.astype(np.float).sum(axis=1)
            confusion_matrix_test_scores += cm_test
            test_accuracy_score.append(accuracy_score(Y_test, out_test))

            fold += 1
            print('\n')
    else:
        wsd.train(X_train, Y_train)
        # classify train data
        print("Train data classification")
        out_train = wsd.classify(X_train)
        cm_training = confusion_matrix(Y_train, out_train)
        cm_training = cm_training / cm_training.astype(np.float).sum(axis=1)
        confusion_matrix_train_scores += cm_training
        train_accuracy_score.append(accuracy_score(Y_train, out_train))

        # classify validation data
        print("Validation data classification")
        out_train = wsd.classify(X_train)
        cm_training = confusion_matrix(Y_train, out_train)
        cm_training = cm_training / cm_training.astype(np.float).sum(axis=1)
        confusion_matrix_train_scores += cm_training
        train_accuracy_score.append(accuracy_score(Y_train, out_train))

        # classify test data
        print("Test data classification")
        out_test = wsd.classify(X_test)

        cm_test = confusion_matrix(Y_test, out_test)
        cm_test = cm_test / cm_test.astype(np.float).sum(axis=1)
        confusion_matrix_test_scores += cm_test
        test_accuracy_score.append(accuracy_score(Y_test, out_test))
        n_splits = 1
        print('\n')

    confusion_matrix_train_scores = np.divide(confusion_matrix_train_scores, n_splits)
    confusion_matrix_validation_scores = np.divide(confusion_matrix_validation_scores, n_splits)
    confusion_matrix_test_scores = np.divide(confusion_matrix_test_scores, n_splits)

    save_accuracy(wisard_param.threshold,
                  wisard_param.addressSize,
                  train_accuracy_score,
                  validation_accuracy_score,
                  test_accuracy_score)

    build_report(wisard_param.threshold,
                 wisard_param.addressSize,
                 confusion_matrix_test_scores,
                 confusion_matrix_train_scores,
                 confusion_matrix_validation_scores)
