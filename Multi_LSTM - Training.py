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


os.environ["CUDA_VISIBLE_DEVICES"] = "-1" #Enable this flag if you don't want to use your GPU for training
EPS = 1000  # n° of epochs
END_RUN = 101
WINDOW_START = 3
WINDOW_END = 5
BAL_TRAIN = True
BAL_VALIDATION = False

with open("results/results_Multihead_LSTM.csv", "a+") as out:
    out.write("K,TP,FN,FP,TN,I_ERR,II_ERR,Precision,Recall,AUC,Accuracy,Bal_Train,Bal_Val \n")

with open("results/results_Multihead_LSTM_singlerun.csv", "a+") as out:
    out.write("Run,window,TP,FN,FP,TN,I_ERR,II_ERR,Precision,Recall,AUC,Accuracy,Bal_Train,Bal_Val \n")

for i in range(WINDOW_START, WINDOW_END + 1):
    accs = []
    tps = []
    fps = []
    tns = []
    fns = []
    aucs = []
    
    for RUN in range(1, END_RUN):
        print("Run: " + str(RUN))
        train = pd.read_csv("datasets_for_LSTM/" + str(i) + "_train_for_LSTM.csv")
        validation = pd.read_csv("datasets_for_LSTM/" + str(i) + "_validation_for_LSTM.csv")

        if BAL_TRAIN:
            train = shuffle_group(train, i)

            train_grp = train.groupby(['status', 'company_name'])

            train_postproc = pd.DataFrame()
            for key, item in train_grp:
                df_grp = train_grp.get_group(key).sort_values("fyear")
                train_postproc = pd.concat([train_postproc, df_grp])

            to_cut = len(train[train.status_label == 'alive']) - len(train[train.status_label == "failed"])
            train = train_postproc.drop(index=train_postproc.index[:to_cut])

        if BAL_VALIDATION:
            validation_bal = shuffle_group(validation, i)
            validation_grp = validation_bal.groupby(['status', 'company_name'])

            val_postproc = pd.DataFrame()
            for key, item in validation_grp:
                df_grp = validation_grp.get_group(key).sort_values("fyear")
                val_postproc = pd.concat([val_postproc, df_grp])

            to_cut = len(validation_bal[validation_bal.status_label == 'alive']) - len(
                validation_bal[validation_bal.status_label == "failed"])
            validation = val_postproc.drop(index=val_postproc.index[:to_cut])

        X_train, y_train = processing(train)
        X_val, y_val = processing(validation)

        avg = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0)
        X_train = (X_train - avg) / std
        X_val = (X_val - avg) / std

        lstm_list = []
        input_list = []
        x_train_list = []
        x_val_list = []

        variables = ["current_assets", "total_assets", "cost_of_goods_sold", "total_long_term_debt",
                     "depreciation_and_amortization", "ebit", "ebitda", "gross_profit", "inventory",
                     "total_current_liabilities", "net_income", "retained_earnings", "total_receivables",
                     "total_revenue",
                     "market_value", "total_liabilities", "net_sales", "total_operating_expenses"]

        y_train = extract_labels(y_train, i)
        y_val = extract_labels(y_val, i)

        y_train = to_categorical(y_train)
        y_val = to_categorical(y_val)

        for var in variables:
            x_train_var = extract_variable_train(X_train, var, i)

            x_train_list.append(x_train_var)
            x_val_var = extract_variable_train(X_val, var, i)
            x_val_list.append(x_val_var)

            L = lstm_for_single_variable(var, i)
            lstm_list.append(L)

        # Concatenate
        merged = Concatenate()([nn.output for nn in lstm_list])
        dense1 = Dense(20, activation='relu')(merged)
        output_layer = Dense(2, activation='softmax')(dense1)

        model = Model(inputs=[nn.input for nn in lstm_list], outputs=output_layer)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(optimizer=optimizer, loss='binary_crossentropy',
                        metrics=["accuracy", tf.keras.metrics.AUC()])
        es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=int(EPS / 10))
        history = model.fit(x_train_list, y_train, epochs=EPS, validation_data=(x_val_list, y_val),
                            verbose=1, callbacks=es)

        val_loss, val_acc, val_AUC = model.evaluate(x_val_list, y_val)
        y_pred = model.predict(x_val_list)
        test_acc = metrics.accuracy_score(y_val.argmax(axis=1), y_pred.argmax(axis=1))
        matrix = confusion_matrix(y_val.argmax(axis=1), y_pred.argmax(axis=1))
        tn, fp, fn, tp = matrix.ravel()

        tps.append(tp)
        fns.append(fn)
        fps.append(fp)
        tns.append(tn)

        if BAL_VALIDATION:
            print('Accuracy:', val_acc)
        else:
            print('AUC:', val_AUC)
        accs.append(test_acc)
        aucs.append(val_AUC)

        with open("results/results_Multihead_LSTM_singlerun.csv", "a+") as out:
            prim_error = 100 * fp / (tn + fp)
            second_error = 100 * fn / (tp + fn)
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)

            out.write(str(RUN) + "," + str(i) + "," + str(tp) + "," + str(
                fn) + "," + str(fp) + "," + str(
                tn) + "," + str(prim_error) + "," + str(second_error) + "," + str(
                precision) + "," + str(
                recall) + "," + str(val_AUC) + "," + str(val_acc) + "," + str(BAL_TRAIN) + "," + str(
                False) + "\n")

    mean = np.average(np.array(accs))
    mean_auc = np.average(np.array(aucs))
    with open("results/results_Multihead_LSTM.csv", "a+") as out:
        prim_error = 100 * np.average(np.array(fps)) / (np.average(np.array(tns)) + np.average(np.array(fps)))
        second_error = 100 * np.average(np.array(fns)) / (np.average(np.array(tps)) + np.average(np.array(fns)))
        precision = np.average(np.array(tps)) / ((np.average(np.array(tps)) + np.average(np.array(fps))))
        recall = np.average(np.array(tps)) / ((np.average(np.array(tps)) + np.average(np.array(fns))))

        out.write(str(i) + "," + str(np.average(np.array(tps))) + "," + str(
            np.average(np.array(fns))) + "," + str(np.average(np.array(fps))) + "," + str(
            np.average(np.array(tns))) + "," + str(prim_error) + "," + str(second_error) + "," + str(
            precision) + "," + str(
            recall) + "," + str(mean_auc) + ","  + str(mean) + "," + str(BAL_TRAIN) + "," + str(
            False) + "\n")
        if i == 5:
            out.write("\n")
