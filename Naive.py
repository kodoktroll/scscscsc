from math import log
import operator

class BayesCla:
    # Prior itu list of columns, masing-masing column adalah
    # dictionary dengan key nilai pada kolom tsb dan value dictionary jumlah kemunculan
    # Misal kolom 0 merupakan kolom kota, maka prior[0]["Depok"]["A"] 
    # bakal ngembaliin jumlah kemunculan hasil "A" saat kolom 0 berisi "Depok" dari training dataset.

    # predictor_class itu dictionary yg keynya hasil dan valuenya jumlah kemunculan hasil dari keseluruhan dataset.
    # misal {"A": 322, "B": 235, "C": 17} dll

    # total row = jumlah baris yang ada pada dataset, gunanya buat jadi pembagi pas prediction
    prior = []
    predictor_class = {}
    total_row = -1
    def fit(self, X , y):
        # method ini pada dasarnya ngebaca training dataset terus nge-tally kemunculan
        # X berisi kolom-kolom
        # y berisi hasil
        # Cek semua jenis label
        self.total_row = len(X)
        self.predictor_class = {i: 0 for i in set(y)}
        self.prior = [{} for i in range(len(X[0]))]
        for i in range(len(X)):
            self.predictor_class[y[i]] += 1
            for j in range(len(X[i])):
                label = X[i][j]
                if(label not in self.prior[j]):
                    self.prior[j][label] = {pred_class: 0 for pred_class in set(y)}
                self.prior[j][label][y[i]] += 1

    def predict(self, X):
        # method ini bakal minta list of kolom-kolom baru yg mau diprediksi
        # dan bakal ngembaliin list of prediksi dengan mencari probability yang paling tinggi
        # di antara hasil-hasil yg ada
        hasil = []
        for i in range(len(X)):
            # misal kemungkinan hasil adalah "A", "B", "C"
            # isi results pertama-tama adalah P("A"), P("B"), P("C")
            # dengan P("A") = jumlah kemunculan A / jumlah row keseluruhan dst
            results = {j: (self.predictor_class[j] / self.total_row) for j in self.predictor_class.keys()}
            for j in range(len(X[i])):
                # ngambil jumlah kemunculan "A", "B", "C"
                # pada kolom j dengan nilai X[i][j]
                # misalkan kolom 0 (kota) labelnya "Depok"
                # isi label_probabilities = {"A": 32, "B": 57, "C": 12} (contoh)
                label_probabilities = self.prior[j][X[i][j]] 
                for k in label_probabilities.keys():
                    # Loop ini ngitung probabilitas label dengan suatu hasil
                    # Terus ngaliin ke result
                    # contoh P("Depok" dan "A") = jumlah kemunculan "Depok" dan hasil "A" / jumlah keseluruhan "A"
                    # terus hasilnya dikaliin deh 
                    curr_value_probability = label_probabilities[k] / self.predictor_class[k] 
                    results[k] *= curr_value_probability
            # dari yang udah diitung tadi, cari result dengan value paling gede, itulah yang jadi prediksinya
            hasil.append(max(results, key=results.get))
        return hasil
        # pass
        #Prediksi hasil