import torch
import torch.nn as nn

class LinearNN3(nn.Module):

    def __init__(self,dims,p_drop):
        super().__init__()
        self.linear1 = nn.Linear(in_features = dims[0], out_features = dims[1], bias = True)
        self.linear2 = nn.Linear(in_features = dims[1], out_features = dims[2], bias = True)
        self.linear3 = nn.Linear(in_features = dims[2], out_features = dims[3], bias = True)
        self.linear4 = nn.Linear(in_features = dims[3], out_features = dims[4], bias = True)

        self.dims = dims
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax()

        self.dropout = nn.Dropout(p=p_drop)

    def forward(self,x):
        x = x.view(-1,self.dims[0])
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = x.view(-1,self.dims[1])
        x = self.linear2(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = x.view(-1,self.dims[2])
        x = self.linear3(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = x.view(-1,self.dims[3])
        x = self.linear4(x)
        x = self.softmax(x)
        x = self.dropout(x)

        return x


        def initialize_weights(self):
            for m in self.modules():
                    if m.bias is not None:
                        torch.nn.init.xavier_normal_(m.weight)
                        torch.nn.init.constant_(m.bias, 1)
#--------------------------------------------------------------------------------------------------------

class LinearNN2(nn.Module):

    def __init__(self,dims):
        super().__init__()
        self.linear1 = nn.Linear(in_features = dims[0], out_features = dims[1], bias = True)
        self.linear2 = nn.Linear(in_features = dims[1], out_features = dims[2], bias = True)
        self.linear3 = nn.Linear(in_features = dims[2], out_features = dims[3], bias = True)

        self.dims = dims
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax()

        self.dropout = nn.Dropout(p=p_drop)

    def forward(self,x):
        x = x.view(-1,self.dims[0])
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = x.view(-1,self.dims[1])
        x = self.linear2(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = x.view(-1,self.dims[2])
        x = self.linear3(x)
        x = self.softmax(x)
        x = self.dropout(x)

        self.dropout = nn.Dropout(p=p_drop)

        return x


        def initialize_weights(self):
            for m in self.modules():
                    if m.bias is not None:
                        torch.nn.init.xavier_normal_(m.weight)
                        torch.nn.init.constant_(m.bias, 1)
#--------------------------------------------------------------------------------------------------------

class LinearNN1(nn.Module):

    def __init__(self,dims):
        super().__init__()
        self.linear1 = nn.Linear(in_features = dims[0], out_features = dims[1], bias = True)
        self.linear2 = nn.Linear(in_features = dims[1], out_features = dims[2], bias = True)

        self.dims = dims
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self,x):
        x = x.view(-1,self.dims[0])
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = x.view(-1,self.dims[1])
        x = self.linear2(x)
        x = self.softmax(x)
        x = self.dropout(x)

        return x


    def initialize_weights(self):
        for m in self.modules():
                if m.bias is not None:
                    torch.nn.init.xavier_normal_(m.weight)
                    torch.nn.init.constant_(m.bias, 1)
