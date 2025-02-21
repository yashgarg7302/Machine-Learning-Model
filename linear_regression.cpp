#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>

using namespace std;

struct DataPoint {
    vector<double> features;
    double target;
};

class LinearRegression {
private:
    vector<double> weights;
    double learningRate;
    int epochs;

    double predict(const vector<double>& features) {
        double result = weights[0];
        for (size_t i = 0; i < features.size(); ++i) {
            result += weights[i + 1] * features[i];
        }
        return result;
    }

public:
    LinearRegression(double lr, int ep) : learningRate(lr), epochs(ep) {}

    void train(const vector<DataPoint>& data) {
        int n = data[0].features.size();
        weights.assign(n + 1, 0.0);

        for (int epoch = 0; epoch < epochs; ++epoch) {
            vector<double> gradient(n + 1, 0.0);
            for (const auto& dp : data) {
                double error = predict(dp.features) - dp.target;
                gradient[0] += error;
                for (size_t i = 0; i < dp.features.size(); ++i) {
                    gradient[i + 1] += error * dp.features[i];
                }
            }
            for (size_t i = 0; i < weights.size(); ++i) {
                weights[i] -= (learningRate / data.size()) * gradient[i];
            }
        }
    }

    double evaluate(const vector<DataPoint>& data) {
        double mse = 0.0;
        for (const auto& dp : data) {
            double error = predict(dp.features) - dp.target;
            mse += error * error;
        }
        return mse / data.size();
    }

    void printModel() {
        cout << "Linear Model: y = " << weights[0];
        for (size_t i = 1; i < weights.size(); ++i) {
            cout << " + " << weights[i] << "*x" << i;
        }
        cout << endl;
    }
};

vector<DataPoint> loadDataset(const string& filename) {
    vector<DataPoint> data;
    ifstream file(filename);
    string line;
    while (getline(file, line)) {
        stringstream ss(line);
        vector<double> features;
        double value;
        while (ss >> value) {
            features.push_back(value);
            if (ss.peek() == ',') ss.ignore();
        }
        double target = features.back();
        features.pop_back();
        data.push_back({features, target});
    }
    return data;
}

int main() {
    string filename = "dataset.csv";
    vector<DataPoint> trainingData = loadDataset(filename);
    
    LinearRegression model(0.01, 1000);
    model.train(trainingData);
    model.printModel();
    
    cout << "Model Evaluation (MSE): " << model.evaluate(trainingData) << endl;
    return 0;
}
