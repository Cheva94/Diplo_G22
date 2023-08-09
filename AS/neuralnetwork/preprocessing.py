import torch
from torch.utils.data import Dataset
import numpy as np

class set_up_data(Dataset):
    "Para utilizarse con el DataLoader de PyTorch"
    #nota: las series de pandas pueden tirar error en las keys, usar arrays de numpy o tensores de torch
    def __init__(self,data,scaling='norm'):
        self.x_data = np.array(data['x']).T
        self.y_data = np.array(data['y'])
        self.xmin, self.xmax = np.amin(self.x_data,axis=1), np.amax(self.x_data,axis=1)

        if scaling=='norm':
            self.scaling_mtd = self.Norm
        elif scaling=='01':
            self.scaling_mtd = self.ScaleTo01

        self.x_data = self.scaling_mtd( self.x_data, self.xmin, self.xmax)
        self.y_data = self.class_data(self.y_data)


    def __getitem__(self,index):
        x = torch.tensor(self.x_data[:,index], dtype=torch.float)
        y = torch.tensor(self.y_data[:,index], dtype=torch.float)
        return x, y

    def __len__(self):
        "Denoto el numero total de muestras"
        return self.y_data.shape[1]

    def Norm(self, data, datamin, datamax):
        #Normalizacion [0,1]
        return (data-datamin[:,np.newaxis])/(datamax[:,np.newaxis]-datamin[:,np.newaxis])

    def ScaleTo01(self, data, datamin, datamax):
        i=0
        for item in datamax:
            if item == 0.:
                datamax[i]=1.
            i+=1
        return data/datamax[:,np.newaxis]


    def denorm(self, data, datamin, datamax):
        return (data)*(datamax[:,np.newaxis]-datamin[:,np.newaxis])+datamin[:,np.newaxis]


    def class_data(self,arr):

        """
        transforma los target a arrays de dimensión (ndat,2)
        donde y[:,0]=1 si es diabetes de tipo 0 y 0 si no,
        y y[:,1]=1 si es diabetes de tipo 1 y 0 si no.
        """

        arr_out = np.zeros((2,len(arr)))

        for i in range(len(arr)):
            if arr[i] == 0:
                arr_out[0,i] = 1
            else:
                arr_out[1,i] = 1

        return arr_out
