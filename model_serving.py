import pandas as pd
import mlflow
import mlflow.sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, plot_roc_curve, confusion_matrix, accuracy_score
from sklearn.model_selection import KFold
import numpy as np
import subprocess
import json


DATA_PATH = r"data/creditcard.csv"


if __name__ == '__main__':
    df = pd.read_csv(DATA_PATH)

    normal = df[df.Class == 0].sample(frac=0.5, random_state=2020).reset_index(drop=True)
    anomaly = df[df.Class == 1]

    normal_train, normal_test = train_test_split(normal, test_size=0.2, random_state=2020)
    anomaly_train, anomaly_test = train_test_split(anomaly, test_size=0.2, random_state=2020)

    scaler = StandardScaler()
    scaler.fit(pd.concat((normal, anomaly)).drop(["Time", "Class"], axis=1))

    scaled_selection = scaler.transform(df.iloc[:80].drop(["Time", "Class"], axis=1))
    input_json = pd.DataFrame(scaled_selection).to_json(orient="split")

    test = df.iloc[:8000]
    true = test.Class

    test = scaler.transform(test.drop(["Time", "Class"], axis=1))
    preds = []
    batch_size = 80

    for f in range(100):
        sample = pd.DataFrame(test[f*batch_size:(f+1)*batch_size]).to_json(orient="split")

        proc = subprocess.run(["curl", "-X", "POST", "-H", "Content-Type:application/json; format=pandas-split",
                               "--data", sample, "http://127.0.0.1:1235/invocations"],
                              stdout=subprocess.PIPE, encoding='utf-8')

        output = proc.stdout
        resp = pd.DataFrame([json.loads(output)])

        preds = np.concatenate((preds, resp.values[0]))

    eval_acc = accuracy_score(true, preds)
    eval_auc = roc_auc_score(true, preds)

    print("Eval Acc: {0}".format(eval_acc))
    print("Eval AUC: {0}".format(eval_auc))
