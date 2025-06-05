import pandas as pd
from model import LSTM_with_Attention
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
import torch 
import torch.optim as optim 
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report
from model import min_max_scale_3d
import  torch.nn.utils.prune as prune
from torch.utils.data import TensorDataset, DataLoader
import gc
from model import create_sequences
import matplotlib.pyplot as plt 
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
df = pd.read_parquet('FDMS_well_WELL_1.parquet',engine='fastparquet')
col = ['Oil_Water_Ratio'
       ,'Emulsion_Stability'
       ,'Formation_Damage_Index'
       ,'Emulsion_Risk',
       'timestamp']
df = df[col]

def make_binary(df,threshold=0.5):
    df['Emulsion_Risk_binary'] = np.where(df['Emulsion_Risk'] > threshold,1,0)
    return df.drop(columns=['Emulsion_Risk_binary']),df['Emulsion_Risk_binary']

def prune_model(model,amount=0.2):
    for name,_ in model.named_modules():
        try:
            prune.l1_unstructured(name,name='weight',amount=amount)
            prune.remove(name,name='weight')
        except:
            continue
def future_eng(df):
  df.set_index(['timestamp'],inplace=True)
  er = df['Emulsion_Risk']
  df.drop(columns=['Emulsion_Risk'],inplace=True)
  for col1 in df.columns:
    df[f'{col1}_dy_dx'] = np.gradient(df[col1])  
    df[f'{col1}_std_10s'] = df[f'{col1}'].groupby(pd.Grouper(freq='2s')).std()
    df[f'{col1}_mean_10s'] = df[f'{col1}'].groupby(pd.Grouper(freq='2s')).mean()
    df[f'{col1}_tanh'] = np.tanh(df[col1])
    df[f'{col1}/Emulsion_Risk_shift_1s'] = df[col1] / er.shift(1)
    df[f'{col1}_ma'] = df[col1].rolling(window=20).mean()
    df[f'{col1}_ema'] = df[col1].ewm(span=20,adjust=False).mean()
    df[f"{col1}_diff"] = df[col1].diff()
    df[f"{col1}_to_ES_ratio"] = df[col1].shift(1) / (df["Emulsion_Stability"].shift(1) + 1e-6)
  df["OWR_rolling_std_10"] = df["Oil_Water_Ratio"].rolling(window=10).std()
  df['Emulsion_Risk_shift_1s'] = er.shift(1)
  return df

    x,y = make_binary(df)
df_for_lstm = future_eng(x)
df_for_lstm.reset_index(inplace=True)
x.drop(columns=['timestamp'],inplace=True)
df_for_lstm = make_binary(df_for_lstm)

gc.collect()
X,y = df_for_lstm.iloc[:20_000],y.iloc[:20_000]
X.fillna(0,inplace=True)
tss = TimeSeriesSplit(n_splits=6)

X,y = create_sequences(data=X,targets=y,lookback=20)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = LSTM_with_Attention(input_dim=X.shape[2],hidden_dim=128,output_dim=2,dropout_rate=0.3,lstm_layers=2).to(device=device) 




optims = optim.Adam(model.parameters(), lr=0.001)
lr_scheduler = CosineAnnealingWarmRestarts(optimizer=optims,T_0=3,T_mult=2)
loss_fn = nn.CrossEntropyLoss().to(device=device) 

BATCH_SIZE = 100
EPOCHS = 100 
for epoch in range(EPOCHS):
    print(f'epoch:{epoch}\n')
    for fold,(train_index,test_index) in enumerate(tss.split(X)):
        print(f'fold:{fold}')
        x_train,y_train = X[train_index],y[train_index]
        x_test,y_test = X[test_index],y[test_index]

        x_train_scaled,x_test_scaled = min_max_scale_3d(x_train,x_test)

        x_train_tensor,x_test_tensor = torch.FloatTensor(x_train_scaled),torch.FloatTensor(x_test_scaled)
        y_train_tensor,y_test_tensor = torch.tensor(y_train),torch.tensor(y_test)

        
        train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False,drop_last=True) 

        test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,drop_last=True)

        model.train()
    
        for batch_x, batch_y in train_loader:
            y_pred = model(batch_x.to(device)).squeeze(1)
            loss_train = loss_fn(y_pred, batch_y.to(device=device))

            optims.zero_grad()
            loss_train.backward()
            optims.step()

            lr_scheduler.step()
            prune_model(model,amount=0.2)
        torch.cuda.empty_cache()

        model.eval()

        with torch.inference_mode():
            for batch_x_test, batch_y_test in test_loader:
                y_pred_test = model(batch_x_test.to(device)).squeeze(1)
                loss_test_batch = loss_fn(y_pred_test,batch_y_test.to(device=device))

            


        print(f"Train Loss: {loss_train} | Test Loss: {loss_test_batch},\nclassification_report:{classification_report(y_true=y_test[:len(y_pred_test)],y_pred=y_pred_test.argmax(dim=1).cpu().numpy(),zero_division=0)}")



df2 = pd.read_parquet('FDMS_well_WELL_2.parquet',engine='fastparquet')
df2 = df2.iloc[:10_000]

df_for_lstm2 = df2[col]
X2,y2 = make_binary(df_for_lstm2)

X2 = future_eng(X2)
X2.reset_index(inplace=True)
X2.drop(columns=['timestamp'],inplace=True)
X2,y2 = create_sequences(data=X2,targets=y2,lookback=20)
X2 = np.nan_to_num(X2)
x_test_scaled2,_ = min_max_scale_3d(X2,X2)



x_test_tensor2 = torch.FloatTensor(x_test_scaled2)
y_test_tensor2 = torch.tensor(y,dtype=torch.long)

with torch.inference_mode():
        y_pred_test2 = model(x_test_tensor2[:128].to(device)).squeeze(1)


print(f"\nclassification_report:{classification_report(y_true=y_test_tensor2[:len(y_pred_test2)],y_pred=y_pred_test2.argmax(dim=1).cpu().numpy(),zero_division=0)}")


y_score = y_pred_test2[:, 1].cpu().numpy()
y_true = y_test_tensor2[:len(y_pred_test2)].cpu().numpy()

y_pred_labels = y_pred_test2.argmax(dim=1).cpu().numpy()
y_true = y_test_tensor2[:len(y_pred_test2)].cpu().numpy()
roc_auc = roc_auc_score(y_true, y_score,average='samples')
cm = confusion_matrix(y_true, y_pred_labels)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)

fpr, tpr, _ = roc_curve(y_true, y_score,)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()



