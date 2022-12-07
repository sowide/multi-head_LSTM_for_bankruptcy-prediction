import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

import os


def extract_labels(df, window):
    df = df.values.reshape(len(df) // window, window)
    return df[:, 1]


def extract_variable_train(df, name, window):
    var = []
    x = df[name]
    x = pd.DataFrame(x)
    x = x[x.columns[::2]]
    return x[name].values.reshape(len(df) // window, window, 1)


def lstm_for_single_variable(name, window):
    model = Sequential()
    model.add(LSTM(window, input_shape=(window, 1)))

    return model


def processing(df):
    y = df.pop('status')
    df.pop('status_label')
    df.pop('cik')
    df.pop('fyear')
    return df, y


def shuffle_group(df, k):
    len_group = k

    index_list = np.array(df.index)
    np.random.shuffle(np.reshape(index_list, (-1, len_group)))

    shuffled_df = df.loc[index_list, :]
    return shuffled_df

WINDOW_START = 3
WINDOW_END = 5
BAL_TEST = False

with open("results/test_results_Multihead_LSTM.csv", "a+") as out:
    out.write("WL,TP,FN,FP,TN,I_ERR,II_ERR,Precision,Recall,AUC,Accuracy,Bal_Train,Bal_Test\n")

for i in range(WINDOW_START, WINDOW_END + 1):
    accs = []
    tps = []
    fps = []
    tns = []
    fns = []
    aucs = []

    test = pd.read_csv("datasets_for_LSTM/" + str(i) + "_test_for_LSTM.csv")

    if BAL_TEST:
        test_bal = shuffle_group(test, i)
        test_grp = test_bal.groupby(['status', 'company_name'])

        test_postproc = pd.DataFrame()
        for key, item in test_grp:
            df_grp = test_grp.get_group(key).sort_values("fyear")
            test_postproc = pd.concat([test_postproc, df_grp])

        to_cut = len(test_bal[test_bal.status_label == 'alive']) - len(
            test_bal[test_bal.status_label == "failed"])
        test = test_postproc.drop(index=test_postproc.index[:to_cut])

    if i == 3:
        X_train = pd.read_csv("best_models/X-train - window_3.csv", index_col=0)
        y_train = pd.read_csv("best_models/Y-train - window_3.csv", index_col=0)
    elif i == 4:
        X_train = pd.read_csv("best_models/X-train - window_4.csv",index_col=0)
        y_train = pd.read_csv("best_models/Y-train - window_4.csv",index_col=0)
    else:
        X_train = pd.read_csv("best_models/X-train - window_5.csv",index_col=0)
        y_train = pd.read_csv("best_models/Y-train - window_5.csv",index_col=0)

    X_test, y_test = processing(test)

    avg = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    X_train = (X_train - avg) / std
    X_test = (X_test - avg) / std

    lstm_list = []
    input_list = []
    x_train_list = []
    x_test_list = []
    variables = ["current_assets", "total_assets", "cost_of_goods_sold", "total_long_term_debt",
                    "depreciation_and_amortization", "ebit", "ebitda", "gross_profit", "inventory",
                    "total_current_liabilities", "net_income", "retained_earnings", "total_receivables",
                    "total_revenue",
                    "market_value", "total_liabilities", "net_sales", "total_operating_expenses"]

    y_train = extract_labels(y_train, i)
    y_test = extract_labels(y_test, i)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    for var in variables:
        x_train_var = extract_variable_train(X_train, var, i)

        x_train_list.append(x_train_var)
        x_text_var = extract_variable_train(X_test, var, i)
        x_test_list.append(x_text_var)

        L = lstm_for_single_variable(var, i)
        lstm_list.append(L)

    if i == 3:
        model = keras.models.load_model(
            'best_models/best_model - window_3')
    elif i == 4:
        model = keras.models.load_model(
            'best_models/best_model - window_4')
    else:
        model = keras.models.load_model(
            'best_models/best_model - window_5')

    test_loss, test_acc, test_AUC = model.evaluate(x_test_list, y_test)
    y_pred = model.predict(x_test_list)
    test_acc = metrics.accuracy_score(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    tn, fp, fn, tp = matrix.ravel()

    tps.append(tp)
    fns.append(fn)
    fps.append(fp)
    tns.append(tn)

    if BAL_TEST:
        print("Accuracy", test_acc)
    else:
        print('AUC:', test_AUC)
    accs.append(test_acc)
    aucs.append(test_AUC)

    with open("results/test_results_Multihead_LSTM.csv", "a+") as out:
        prim_error = 100 * fp / (tn + fp)
        second_error = 100 * fn / (tp + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        out.write(str(i) + "," + str(tp) + "," + str(
            fn) + "," + str(fp) + "," + str(
            tn) + "," + str(prim_error) + "," + str(second_error) + "," + str(
            precision) + "," + str(
            recall) + "," + str(test_AUC) + "," + str(test_acc) + "," + str(True) + "," + str(
            BAL_TEST) + "\n")
