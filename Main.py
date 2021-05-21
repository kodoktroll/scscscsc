from Naive import BayesCla

model = BayesCla()
ContohX_training = [ 
['Depok' , 'Android'] ,
['Depok' , 'Iphone'] ,
['Depok' , 'BuWei'] ,
['Jakarta' , 'Android'] ,
['Jakarta' , 'Iphone'] ,
['Surabaya' , 'Android'] ,
['Surabaya' , 'Iphone'] ,
['Surabaya' , 'BuWei'] ,
]

ContohY_training = ['A' , 'B' , 'C' , 'A' , 'B' , 'C' , 'A' , 'B']

ContohX_test = [ 
['Jakarta' , 'BuWei'] ,
['Depok' , 'Android'] , 
]

model.fit(ContohX_training  , ContohY_training)

prediction = model.predict(ContohX_test)

assert len(prediction) == 2 , "Harusnya jumlah hasil prediction adalah 2"
assert prediction[0] == 'B' , "Jika kodingannya benar, harusnya dia akan meng-output B"
assert prediction[1] == 'A' , "Jika kodingannya benar, harusnya dia akan meng-output A"

file = open("train.csv")
data = file.readlines()

X_train = []
y_train =[]

for row in data[1:]:
    masukan = row.strip().split(" ")
    X_train.append(masukan[:-1])
    y_train.append(masukan[-1])

file = open("test.csv")
data = file.readlines()

X_test = []
y_test =[]

for row in data[1:]:
    masukan = row.strip().split(" ")
    X_test.append(masukan[:-1])
    y_test.append(masukan[-1])

model = BayesCla()

model.fit(X_train  , y_train)

prediction = model.predict(X_test)

#TODO : Hitung akurasi per kelas

def generate_confusion_matrix(classes, predicted, actual):
    #asumsi len predicted = len actual
    # row = predicted, column = actual
    result_dict = {i: {j: 0 for j in classes} for i in classes}
    for i in range(len(predicted)):
        result_dict[predicted[i]][actual[i]] += 1
    return result_dict

def calculate_precision(predict_class, matrix):
    return matrix[predict_class][predict_class] / sum([matrix[predict_class][i] for i in matrix.keys()])

def calculate_recall(predict_class, matrix):
    return matrix[predict_class][predict_class] / sum([matrix[i][predict_class] for i in matrix.keys()])

def calculate_accuracy(matrix):
    correct = sum([matrix[i][i] for i in matrix.keys()])
    total = sum([matrix[i][j] for i in matrix.keys() for j in matrix[i].keys()])
    return correct / total
    
def pretty_print_matrix(matrix):
    first_line = ""
    first_line += "{:5.5}".format("     ")
    for i in matrix.keys():
        first_line += " {:5.5}".format(i)
    rest = []
    for i in matrix.keys():
        curr_line = "{:5.5}".format(i)
        for j in matrix[i].keys():
            curr_line += " {:5.5}".format(str(matrix[i][j]))
        rest.append(curr_line)
    print(first_line)
    for i in rest:
        print(i)

my_matrix = generate_confusion_matrix(model.predictor_class, prediction, y_test)

