import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score

import network as net
import preprocessing as prep

import numpy as np
from scipy.stats import spearmanr

def rmse( modeldata , targetdata ) :
    return np.sqrt( np.mean( (modeldata.flatten() - targetdata.flatten()) ** 2 ) )

def bias( modeldata , targetdata ) :
    return np.mean( modeldata.flatten() - targetdata.flatten() )

def corr_P( modeldata , targetdata ) :
    return np.corrcoef( modeldata.flatten() , targetdata.flatten() )[0,1]

def corr_S( modeldata , targetdata ) :
    return spearmanr( modeldata.flatten() , targetdata.flatten() )[0]


def define_seed(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
#-------------------------------------------------------------------------------------------------

df = pd.read_csv('../data/diabetes_prediction_dataset_train-labeled.csv')

cat_cols = ['gender', 'smoking_history']
num_cols = [x for x in df.columns if x not in cat_cols and x not in ['patient', 'diabetes']]
# En las columnas numéricas quitamos la columna "patient" que contiene el id de los pacientes y "diabetes" que es la variable target

X = df.drop(columns=['patient', 'diabetes'])
y = df['diabetes']
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state = 8)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, train_size=0.5, random_state = 22)


#Cargo el pipeline:
pipeline = joblib.load('../preproc_pipeline.pkl')
# Fiteo el pipeline
x_train_transformed = pipeline.fit_transform(x_train)
x_test_transformed = pipeline.transform(x_test)
x_val_transformed = pipeline.transform(x_val)

#-------------------------------------------------------------------------------------------------
#fijo semilla
define_seed(seed=100)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

n_layers = 3    #numero de capas ocultas del modelo

dropout = 0.5   #probabilidad de desactivar neuronas en el entrenamiento

#inicializo el modelo
if n_layers == 3:
    nlayers = [13,104,208,52,2]
    model = net.LinearNN3(nlayers,dropout)
elif n_layers == 2:
    nlayers = [13,130,20,2]
    model = net.LinearNN2(nlayers,dropout)
elif n_layers == 1:
    nlayers = [13,1000,2]
    model = net.LinearNN1(nlayers,dropout)
else: quit('not implemented yet')

#Guardamos al modelo que definimos previamente
model.to(device) #Cargamos en memoria

#----------------------------------------------------------------------------------------------------------------------------------
#Hiperparametros
batch_size= len(y_test)
max_epochs = 300
learning_rate = 5e-4

#cargo los datos
train_data = {'x':x_train_transformed, 'y':y_train}
test_data = {'x':x_test_transformed, 'y':y_test}
val_data = {'x':x_val_transformed, 'y':y_val}

ratio_1_0_train = len([i for i in y_train if i==1])/len(y_train)
ratio_1_0_test = len([i for i in y_test if i==1])/len(y_test)

train_subset = prep.set_up_data(train_data, scaling='01')
test_subset = prep.set_up_data(test_data, scaling='01')
val_subset = prep.set_up_data(val_data, scaling='01')

dataloader_train = DataLoader(train_subset, batch_size = batch_size, shuffle=False)
dataloader_test  = DataLoader(test_subset , batch_size=len(y_test), shuffle=False)
dataloader_val  = DataLoader(val_subset , batch_size=len(y_val), shuffle=False)


#----------------------------------------------------------------------------------------------------------------------------------
count_0 = len([i for i in y_train if i==0])
count_1 = len([i for i in y_train if i==1])

counts = [count_0,count_1]

#La función de costo va aestar ponderada por el inverso de la cantidad de elementos de cada clase,
#para favorecer la diabetes tipo 1, que es mucho menos recurrente
ww = 1./np.array(counts)    #pesos ponderados
ww = np.array([1,1])        #pesos no ponderados
ww_norm = ww/np.sqrt(np.sum(ww)**2) #normalizo

#Defino la función de costo, en este caso es la CrossEntropyLoss, para problemas de clasificación
Loss = nn.CrossEntropyLoss(weight=torch.as_tensor(ww_norm)).to(device)

#Definimos el optimizador
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#----------------------------------------------------------------------------------------------------------------------------------

#Listas donde guardamos loss de entrenamiento, y para el de validación la loss y las métricas de evaluación.
RMSE, BIAS, Corr_P, Corr_S = [], [], [], []
loss_train = []
loss_val = []

for epoch in range(max_epochs):
    #print('Epoca: '+ str(epoch+1) + ' de ' + str(max_epochs) )

    #Entrenamiento del modelo
    model.train()  #Esto le dice al modelo que se comporte en modo entrenamiento.

    sum_loss = 0.0
    batch_counter = 0

    # Iteramos sobre los minibatches.
    for inputs, target in dataloader_train :
        #Enviamos los datos a la memoria.
        inputs, target = inputs.to(device), target.to(device)

        optimizer.zero_grad()

        outputs = model(inputs).squeeze()

        loss = Loss(outputs.float(), target.float()).to(device)

        loss.backward()
        optimizer.step()

        batch_counter += 1
        sum_loss = sum_loss + loss.item()

    #Calculamos la loss media sobre todos los minibatches
    loss_train.append( sum_loss / batch_counter )

    model.eval()   #Esto le dice al modelo que lo usaremos para evaluarlo (no para entrenamiento)

    #Calculamos la función de costo para la muestra de validación.
    input_val, target_val = next(iter(dataloader_val))
    input_val, target_val = input_val.to(device) , target_val.to(device)

    with torch.no_grad():
        output_val = model(input_val).squeeze()

    #Calculo de la loss de validacion
    loss = Loss(output_val.float(), target_val.float()).cpu()
    loss_val.append(loss.numpy())

    print('Loss Train: ', str(loss_train[epoch]))
    print('Loss Val:   ', str(loss_val[epoch]))
    print('-'*50)
    ###################################

    #Calculo de metricas RMSE, BIAS, Correlacion de Pearson y Spearman
    Corr_P.append(corr_P(output_val.cpu(), target_val.cpu()))
    Corr_S.append(corr_S(output_val.cpu(), target_val.cpu()))

plt.plot(loss_train, label='train')
plt.plot(loss_val, label='val')
plt.legend()
plt.savefig('tmp/loss.png',dpi=300)

#----------------------------------------------------------------------------------------------------------------------------------

#hago la prediccion sobre el conjunto de testing
model.eval()   #Esto le dice al modelo que lo usaremos para evaluarlo (no para entrenamiento)

input_test, target_test = next(iter(dataloader_test))
input_test, target_test = input_test.detach().to(device) , target_test.detach().to(device)

with torch.no_grad():
    output_test = model( input_test )

#transformo outputs probabilisticos a etiquetas. La mayor probabilidad tiene un 1, el resto cero
output_test = np.array(output_test.cpu())

target_test = target_test.cpu()
output_test0 = np.zeros(output_test.shape)
for i in range(output_test.shape[0]):
    max = np.argmax(output_test[i])
    output_test0[i,max] = 1


import sklearn
from sklearn import calibration as cal
#from sklearn.calibration import CalibrationDisplay

#accuracy
acc = accuracy_score(np.argmax(target_test,axis=1) , np.argmax(output_test,axis=1))
print('accuracy: ',acc)
#Matriz de confusion
plt.clf()
cmatrix=sklearn.metrics.confusion_matrix( np.argmax(target_test,axis=1) , np.argmax(output_test,axis=1) , normalize="true")
disp = sklearn.metrics.ConfusionMatrixDisplay(cmatrix)
disp.plot()
plt.savefig("tmp/conf_mat.png",dpi=300)

fig ,ax = plt.subplots(1,1)
disp = cal.CalibrationDisplay.from_predictions( target_test[:,0], output_test[:,0], n_bins=10, ax=ax )
ax.set_title('Diagrama de confiabilidad')
ax.set_xlabel('Probabilidad de diabetes tipo 0')
ax.set_ylabel('Frecuencia observada - diabetes tipo 0')
fig.savefig("tmp/diag_conf_0.png",dpi=300)

fig ,ax = plt.subplots(1,1)
disp = cal.CalibrationDisplay.from_predictions( target_test[:,1], output_test[:,1], n_bins=10, ax=ax )
ax.set_title('Diagrama de confiabilidad')
ax.set_xlabel('Probabilidad de diabetes tipo 1')
ax.set_ylabel('Frecuencia observada - diabetes tipo 1')
fig.savefig("tmp/diag_conf_1.png",dpi=300)
