import mlflow
import mlflow.sklearn


def test_loaded_model(model, x_test, y_test):
    print(r'Accuracy score: {0}'.format(model.score(x_test, y_test)))


if __name__ == '__main__':
    loaded_model = mlflow.sklearn.load_model("runs:/05c14a2aea30455d81e7f77ea9dd01a7/log_reg_model")

    