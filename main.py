import pandas as pd
import numpy as np
import sklearn.model_selection
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style


class StudentLinearRegression:
    def prepare_data_set(self):
        data = pd.read_csv("student-mat.csv", sep=";")
        data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
        return data

    def prepare_training_and_test_data(self, data):
        predict = "G3"
        X = np.array(data.drop([predict], 1))
        y = np.array(data[predict])
        return sklearn.model_selection.train_test_split(X, y, test_size=.1)

    def train_model(self, x_train, x_test, y_train, y_test):
        best = 0
        for _ in range(30):
            linear_model = LinearRegression()
            linear_model.fit(x_train, y_train)
            model_accuracy = linear_model.score(x_test, y_test)
            if model_accuracy > best:
                best = model_accuracy
                with open("student-model.pickle", "wb") as model_file:
                    pickle.dump(linear_model, model_file)

    def load_model(self):
        pickle_in = open("student-model.pickle", "rb")
        linear_model = pickle.load(pickle_in)
        return linear_model

    def predict(self,linear_model, x_test, y_test):
        predictions = linear_model.predict(x_test)
        for x in range(len(predictions)):
            print(predictions[x], x_test[x], y_test[x])

    def visualize(self, data):
        # visualization
        p = "G1"
        style.use("seaborn-paper")
        pyplot.scatter(data[p], data["G3"])
        pyplot.xlabel(p)
        pyplot.ylabel("Final Grade")
        pyplot.show()

    def start(self):
        data = self.prepare_data_set()
        x_train, x_test, y_train, y_test = self.prepare_training_and_test_data(data)
        self.train_model(x_train, x_test, y_train, y_test)
        linear_model = self.load_model()
        self.predict(linear_model, x_test, y_test)
        self.visualize(data)


def main():
    student_linear_regression = StudentLinearRegression()
    student_linear_regression.start()


if __name__ == "__main__":
    main()

