import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, plot_roc_curve, confusion_matrix
from sklearn.model_selection import KFold

import mlflow
import mlflow.sklearn


DATA_PATH = r"data/creditcard.csv"
ANOMALY_WEIGHTS = [10, 50, 100, 150, 200]
NUM_FOLDS = 5


def train(sk_model, x_train, y_train):
    sk_model = sk_model.fit(x_train, y_train)
    train_acc = sk_model.score(x_train, y_train)

    mlflow.log_metric("train_acc", train_acc)
    print(r'Train accuracy: {0}'.format(train_acc))


def evaluate(sk_model, x_test, y_test):
    eval_acc = sk_model.score(x_test, y_test)
    preds = sk_model.predict(x_test)
    auc_score = roc_auc_score(y_test, preds)

    mlflow.log_metric("eval_acc", eval_acc)
    mlflow.log_metric("auc_score", auc_score)

    print(r'AUC Score: {0}'.format(auc_score))
    print(r'Eval Accuracy: {0}'.format(eval_acc))

    roc_plot = plot_roc_curve(sk_model, x_test, y_test, name='scikit-learn ROC Curve')
    plt.savefig(r'results/sklearn_roc_plot.png')
    plt.clf()

    conf_matrix = confusion_matrix(y_test, preds)
    ax = sns.heatmap(conf_matrix, annot=True, fmt='g')
    ax.invert_xaxis()
    ax.invert_yaxis()
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig(r'results/sklearn_confusion_matrix')

    mlflow.log_artifact(r'results/sklearn_roc_plot.png')
    mlflow.log_artifact(r'results/sklearn_confusion_matrix.png')


if __name__ == '__main__':
    df = pd.read_csv(DATA_PATH)
    df = df.drop("Time", axis=1)

    normal = df[df.Class == 0].sample(frac=0.5, random_state=2020).reset_index(drop=True)
    anomaly = df[df.Class == 1]

    normal_train, normal_test = train_test_split(normal, test_size=0.2, random_state=2020)
    anomaly_train, anomaly_test = train_test_split(anomaly, test_size=0.2, random_state=2020)
    normal_train, normal_validate = train_test_split(normal_train, test_size=0.25, random_state=2020)
    anomaly_train, anomaly_validate = train_test_split(anomaly_train, test_size=0.25, random_state=2020)

    x_train = pd.concat((normal_train, anomaly_train))
    x_test = pd.concat((normal_test, anomaly_test))
    x_validate = pd.concat((normal_validate, anomaly_validate))

    y_train = np.array(x_train["Class"])
    y_test = np.array(x_test["Class"])
    y_validate = np.array(x_validate["Class"])

    x_train = x_train.drop("Class", axis=1)
    x_test = x_test.drop("Class", axis=1)
    x_validate = x_validate.drop("Class", axis=1)

    scaler = StandardScaler()
    scaler.fit(pd.concat((normal, anomaly)).drop("Class", axis=1))
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    x_validate = scaler.transform(x_validate)

    kfold = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=2020)

    mlflow.set_experiment("sklearn_creditcard_guided_search")
    logs = []
    for f in range(len(ANOMALY_WEIGHTS)):
        fold = 1
        accuracies = []
        auc_scores = []

        for train, test in kfold.split(x_validate, y_validate):
            with mlflow.start_run():
                weight = ANOMALY_WEIGHTS[f]
                mlflow.log_param("anomaly_weight", weight)

                class_weight = {
                    0: 1,
                    1: weight
                }
                sk_model = LogisticRegression(random_state=None, max_iter=400, solver='newton-cg',
                                              class_weight=class_weight).fit(x_validate[train], y_validate[train])

                for h in range(40):
                    print('-', end="")

                print(r'\nfold: {0}\nAnomaly weight: {1}'.format(fold, weight))
                train_acc = sk_model.score(x_validate[train], y_validate[train])
                mlflow.log_metric("train_acc", train_acc)

                eval_acc = sk_model.score(x_validate[test], y_validate[test])
                preds = sk_model.predict(x_validate[test])
                mlflow.log_metric("eval_acc", eval_acc)

                try:
                    auc_score = roc_auc_score(y_validate[test], preds)

                except:
                    auc_score = -1

                mlflow.log_metric("auc_score", auc_score)
                print(r'AUC: {0}\nEval_acc: {1}'.format(auc_score, eval_acc))

                accuracies.append(eval_acc)
                auc_scores.append(auc_score)

                log = [sk_model, x_validate[test], y_validate[test], preds]
                logs.append(log)

                mlflow.sklearn.log_model(sk_model, r'anomaly_weight_{0}_fold_{1}'.format(weight, fold))
                fold = fold + 1

                mlflow.end_run()

        print(r'\nAverages: ')
        print(r'Accuracy: {0}'.format(np.mean(accuracies)))
        print(r'AUC: {0}'.format(np.mean(auc_scores)))

        print(r'Best: ')
        print(r'Accuracy: {0}'.format(np.max(accuracies)))
        print(r'AUC: {0}'.format(np.max(auc_scores)))







