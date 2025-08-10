from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
from django.conf import settings
import os
import io
import base64
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import pymysql
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, load_model, Model
import pickle
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from keras.layers import  MaxPooling2D
from keras.layers import Convolution2D
import timeit

global username
global X_train, X_test, y_train, y_test, X, Y
accuracy = []
precision = []
recall = [] 
fscore = []
path = "Dataset"

#function to calculate all metrics
def calculateMetrics(algorithm, y_test, predict):
    a = (accuracy_score(y_test,predict)*100)
    p = (precision_score(y_test, predict,average='macro') * 100)
    r = (recall_score(y_test, predict,average='macro') * 100)
    f = (f1_score(y_test, predict,average='macro') * 100)
    a = round(a, 3)
    p = round(p, 3)
    r = round(r, 3)
    f = round(f, 3)
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    return algorithm

dataset = pd.read_csv("Dataset/5g_ddos.csv")
dataset.fillna(0, inplace=True)#remove missing values

dataset['ema'] = dataset['Total Fwd Packet'].ewm(span=3, adjust=False).mean()

Y = dataset['label'].ravel()
dataset.drop(['label', 'Slice'], axis = 1,inplace=True)

X = dataset.values
scaler = StandardScaler()
X = scaler.fit_transform(X)

indices = np.arange(X.shape[0])
np.random.shuffle(indices) #shuffle dataset
X = X[indices]
Y = Y[indices]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
data = np.load("model/data.npy", allow_pickle=True)
X_train, X_test, y_train, y_test = data

lr = LogisticRegression(solver="liblinear", tol=0.1)
lr.fit(X_train, y_train)
start = timeit.default_timer()
predict = lr.predict(X_test)
end = timeit.default_timer()
ema_time = end - start
calculateMetrics("EMA", y_test, predict)

mlp = MLPClassifier(tol=0.1)
mlp.fit(X_train, y_train)
start = timeit.default_timer()
predict = mlp.predict(X_test)
end = timeit.default_timer()
mlp_time = end - start
calculateMetrics("MLP", y_test, predict)

X_train1 = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test1 = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
y_train1 = to_categorical(y_train)
y_test1 = to_categorical(y_test)
cnn1d = Sequential()
#defining CNN layers with number of neuron as 32 to filter dataset features
cnn1d.add(Conv1D(filters=32, kernel_size = 1, input_shape = (X_train1.shape[1], X_train1.shape[2])))
#adding maxpool layer
cnn1d.add(MaxPooling1D(pool_size = 2))
cnn1d.add(Conv1D(filters=32, kernel_size = 1, activation='relu'))
cnn1d.add(MaxPooling1D(pool_size = 2))
cnn1d.add(Flatten())
#defining output layer
cnn1d.add(Dense(units = 256))
cnn1d.add(Dense(units = y_train1.shape[1], activation = 'softmax'))
#compile and train the model
cnn1d.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
if os.path.exists("model/cnn1d_weights.hdf5") == False:
    model_check_point = ModelCheckpoint(filepath='model/cnn1d_weights.hdf5', verbose = 1, save_best_only = True)
    hist = cnn1d.fit(X_train1, y_train1, batch_size = 16, epochs = 20, validation_data=(X_test1, y_test1), callbacks=[model_check_point], verbose=1)
    f = open('model/cnn1d_history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()    
else:
    cnn1d.load_weights("model/cnn1d_weights.hdf5")

#perform prediction on test data
start = timeit.default_timer()
predict = cnn1d.predict(X_test1)
predict = np.argmax(predict, axis=1)
y_test2 = np.argmax(y_test1, axis=1)
end = timeit.default_timer()
ai_time = end - start
calculateMetrics("AI", y_test2, predict)

X_train1 = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1, 1))
X_test1 = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1, 1))
y_train1 = to_categorical(y_train)
y_test1 = to_categorical(y_test)

cnn2d = Sequential()
cnn2d.add(Convolution2D(32, (1, 1), input_shape = (X_train1.shape[1], X_train1.shape[2], X_train1.shape[3]), activation = 'relu'))
cnn2d.add(MaxPooling2D(pool_size = (1, 1)))
cnn2d.add(Convolution2D(32, (1, 1), activation = 'relu'))
cnn2d.add(MaxPooling2D(pool_size = (1, 1)))
cnn2d.add(Flatten())
cnn2d.add(Dense(units = 256, activation = 'relu'))
cnn2d.add(Dense(units = y_train1.shape[1], activation = 'softmax'))
cnn2d.compile(optimizer = "adam", loss = 'categorical_crossentropy', metrics = ['accuracy'])
if os.path.exists("model/cnn2d_weights.hdf5") == False:
    model_check_point = ModelCheckpoint(filepath='model/cnn2d_weights.hdf5', verbose = 1, save_best_only = True)
    hist = cnn2d.fit(X_train1, y_train1, batch_size = 16, epochs = 20, validation_data=(X_test1, y_test1), callbacks=[model_check_point], verbose=1)
    f = open('model/cnn2d_history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()    
else:
    cnn2d.load_weights("model/cnn2d_weights.hdf5")

#perform prediction on test data
start = timeit.default_timer()
predict = cnn2d.predict(X_test1)
end = timeit.default_timer()
ext_time = end - start
predict = np.argmax(predict, axis=1)
y_test2 = np.argmax(y_test1, axis=1)
calculateMetrics("Extension", y_test2, predict)

def Predict(request):
    if request.method == 'GET':
        return render(request, 'Predict.html', {})

def PredictAction(request):
    if request.method == 'POST':
        global scaler
        cnn2d = load_model("model/cnn2d_weights.hdf5")
        myfile = request.FILES['t1'].read()
        filename = request.FILES['t1'].name
        if os.path.exists('SDNApp/static/'+filename):
            os.remove('SDNApp/static/'+filename)
        with open('SDNApp/static/'+filename, "wb") as file:
            file.write(myfile)
        file.close()
        testData = pd.read_csv('SDNApp/static/'+filename)
        testData.fillna(0, inplace=True)#remove missing values
        testData['ema'] = testData['Total Fwd Packet'].ewm(span=3, adjust=False).mean()
        data = testData.values
        testData = testData.values
        testData = scaler.transform(testData)
        testData = np.reshape(testData, (testData.shape[0], testData.shape[1], 1, 1))
        predict = cnn2d.predict(testData)
        predict = np.argmax(predict, axis=1)
        labels = ['Normal', 'DDoS']
        color = ['green', 'red']
        output='<table border=1 align=center width=100%><tr><th><font size="3" color="black">Network Monitor Test Data</th>'
        output += '<th><font size="3" color="black">Detected Status</th></tr>'
        for i in range(len(predict)):
            output += '<tr><td><font size="3" color="black">'+str(data[i])+'</td><td><font size="3" color="'+color[predict[i]]+'">'+str(labels[predict[i]])+'</td></tr>'
        output + "</table><br/><br/><br/><br/><br/>"
        context= {'data':output}
        return render(request, 'UserScreen.html', context)

def TrainModel(request):
    if request.method == 'GET':
        global accuracy, precision, recall, fscore, ema_time, mlp_time, ai_time, ext_time
        labels = ['Normal', 'DDoS']
        output='<table border=1 align=center width=100%><tr><th><font size="3" color="black">Algorithm Name</th><th><font size="3" color="black">Accuracy</th>'
        output += '<th><font size="3" color="black">Precision</th><th><font size="3" color="black">Recall</th><th><font size="3" color="black">FSCORE</th>'
        output += '<th><font size="3" color="black">Execution Time</th></tr>'
        algorithms = ['EMA', 'MLP', 'AI CNN1D', 'Extension AI CNN2D']
        execution_time = [ema_time, mlp_time, ai_time, ext_time]
        for i in range(len(algorithms)):
            output += '<td><font size="3" color="black">'+algorithms[i]+'</td><td><font size="3" color="black">'+str(accuracy[i])+'</td><td><font size="3" color="black">'+str(precision[i])+'</td>'
            output += '<td><font size="3" color="black">'+str(recall[i])+'</td><td><font size="3" color="black">'+str(fscore[i])+'</td>'
            output += '<td><font size="3" color="black">'+str(execution_time[i])+'</td></tr>'
        output+= "</table></br>"
        figure, axis = plt.subplots(nrows=1, ncols=2,figsize=(10, 3))#display original and predicted segmented image
        axis[0].set_title("Execution Time Comparison Graph")
        axis[0].tick_params(axis='x', labelrotation=70)
        axis[1].set_title("Attack Detection Accuracy Comparison Graph")
        bars = ['EMA', 'MLP', 'AI CNN1D', 'Extension AI CNN2D']
        axis[0].bar(bars, execution_time, color='skyblue')
        axis[0].set_ylabel('Execution Time')
        df = pd.DataFrame([['EMA','Precision',precision[0]],['EMA','Recall',recall[0]],['EMA','F1 Score',fscore[0]],['EMA','Accuracy',accuracy[0]],
                           ['MLP','Precision',precision[1]],['MLP','Recall',recall[1]],['MLP','F1 Score',fscore[1]],['MLP','Accuracy',accuracy[1]],
                           ['AI CNN1D','Precision',precision[2]],['AI CNN1D','Recall',recall[2]],['AI CNN1D','F1 Score',fscore[2]],['AI CNN1D','Accuracy',accuracy[2]],
                           ['Extension AI CNN2D','Precision',precision[3]],['Extension AI CNN2D','Recall',recall[3]],['Extension AI CNN2D','F1 Score',fscore[3]],['Extension AI CNN2D','Accuracy',accuracy[3]],
                          ],columns=['Parameters','Algorithms','Value'])
        df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar', ax=axis[1])
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        #plt.close()
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        plt.clf()
        plt.cla()
        context= {'data':output, 'img': img_b64}
        return render(request, 'UserScreen.html', context)

def LoadDataset(request):    
    if request.method == 'GET':
        global X_train, X_test, y_train, y_test, X, Y, labels
        labels = ['Normal', 'DDoS']
        output = '<font size="3" color="black">Cloud 5G Network SDN Simulation Attack Dataset Loaded</font><br/>'
        output += '<font size="3" color="blue">Total records found in Dataset = '+str(X.shape[0])+'</font><br/>'
        output += '<font size="3" color="blue">Total features found in Dataset = '+str(X.shape[1])+'</font><br/>'
        output += '<font size="3" color="blue">Different Class Labels found in Dataset = '+str(labels)+'</font><br/><br/>'
        output += '<font size="3" color="black">Dataset Train & Test Split details</font><br/>'
        output += '<font size="3" color="blue">80% dataset records used to train AI = '+str(X_train.shape[0])+'</font><br/>'
        output += '<font size="3" color="blue">20% dataset records used to test AI = '+str(X_test.shape[0])+'</font><br/>'
        context= {'data':output}
        return render(request, 'UserScreen.html', context)

def RegisterAction(request):
    if request.method == 'POST':
        global username
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        contact = request.POST.get('t3', False)
        email = request.POST.get('t4', False)
        address = request.POST.get('t5', False)
        
        output = "none"
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'sdn',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select username FROM register")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == username:
                    output = username+" Username already exists"
                    break                
        if output == "none":
            db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'sdn',charset='utf8')
            db_cursor = db_connection.cursor()
            student_sql_query = "INSERT INTO register(username,password,contact,email,address) VALUES('"+username+"','"+password+"','"+contact+"','"+email+"','"+address+"')"
            db_cursor.execute(student_sql_query)
            db_connection.commit()
            print(db_cursor.rowcount, "Record Inserted")
            if db_cursor.rowcount == 1:
                output = "Signup process completed. Login to perform DDOS Attack Detection"
        context= {'data':output}
        return render(request, 'Register.html', context)    

def UserLoginAction(request):
    global username
    if request.method == 'POST':
        global username
        status = "none"
        users = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'sdn',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select username,password FROM register")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == users and row[1] == password:
                    username = users
                    status = "success"
                    break
        if status == 'success':
            context= {'data':'Welcome '+username}
            return render(request, "UserScreen.html", context)
        else:
            context= {'data':'Invalid username'}
            return render(request, 'UserLogin.html', context)

def UserLogin(request):
    if request.method == 'GET':
       return render(request, 'UserLogin.html', {})

def index(request):
    if request.method == 'GET':
       return render(request, 'index.html', {})

def Register(request):
    if request.method == 'GET':
       return render(request, 'Register.html', {})
