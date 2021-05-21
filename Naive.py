from math import log
import operator

class BayesCla:
    prior = []
    predictor_class = {}
    total_row = -1
    def fit(self, X , y):
        # Cek semua jenis label
        # X = array 2 dimensi yg berisi baris-baris variabel
        # y = nilai dari masing-masing baris.
        # prior[0]["Depok"]["A"] -> suatu value
        self.total_row = len(X)
        self.predictor_class = {i: 0 for i in set(y)}
        self.prior = [{} for i in range(len(X[0]))]
        for i in range(len(X)):
            self.predictor_class[y[i]] += 1
            for j in range(len(X[i])):
                label = X[i][j]
                if(label not in self.prior[j]):
                    self.prior[j][label] = {predicts: 0 for predicts in set(y)}
                self.prior[j][label][y[i]] += 1

    def predict(self, X):
        hasil = []
        for i in range(len(X)):
            results = {j: (self.predictor_class[j] / self.total_row) for j in self.predictor_class.keys()}
            for j in range(len(X[i])):
                label_probabilities = self.prior[j][X[i][j]]
                for k in label_probabilities.keys():
                    curr_value_probability = label_probabilities[k] / self.predictor_class[k]
                    results[k] *= curr_value_probability
            hasil.append(max(results, key=results.get))
        return hasil
        # pass
        #Prediksi hasil