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
