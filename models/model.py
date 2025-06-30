import torch.nn as nn
import torch 
from sklearn.preprocessing import MinMaxScaler 
import numpy as np
class LSTM_with_Attention(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim,dropout_rate,lstm_layers):
        super().__init__()
        
        self.in_stack = nn.Sequential(
            nn.Linear(in_features=input_dim,out_features=hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU()
        )

        self.lstm = nn.LSTM(input_size=hidden_dim,num_layers=lstm_layers,batch_first=True,hidden_size=hidden_dim,dropout=dropout_rate)


        self.mid_stack = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_dim,out_features=hidden_dim)
        )    

        self.out_stack = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_dim,out_features=output_dim)
        )

        self.att = nn.Linear(hidden_dim,1)
    


    def forward(self,x):
        x = self.in_stack(x)

        x,_ = self.lstm(x)
        
        x = self.mid_stack(x)
        # attention mechanisms
        w = torch.softmax(self.att(x),dim=1)
        x = torch.sum(w * x,dim=1)

        return self.out_stack(x)

    


def min_max_scale_3d(x_train,x_test):

    train_shape = x_train.shape
    test_shape = x_test.shape
    reshaped_train = x_train.reshape(-1, train_shape[-1])
    reshaped_test = x_test.reshape(-1, test_shape[-1])
    scaler = MinMaxScaler()
    scaled_train = scaler.fit_transform(reshaped_train)
    scaled_test = scaler.transform(reshaped_test)
    return scaled_train.reshape(train_shape),scaled_test.reshape(test_shape)




def create_sequences(data, targets, lookback:int):
    x_seq = []
    y_seq = []
    for i in range(len(data) - lookback):
        data_seq = data.iloc[i:i + lookback]
        target_seq = targets.iloc[i+lookback]
        x_seq.append(data_seq)
        y_seq.append(target_seq)
    return torch.tensor(np.array(x_seq)).to(torch.float),torch.tensor(np.array(y_seq)).to(torch.long)
