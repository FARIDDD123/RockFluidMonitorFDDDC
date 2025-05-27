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
import torch.nn.utils.prune as prune
from torch.utils.data import TensorDataset, DataLoader
import random 
import matplotlib.pyplot as plt 
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import gc
df = pd.read_parquet('your_data.parquet')
col = ['Emulsion_Risk'                  
,'Oil_Water_Ratio'                
,'Solid_Content_%'                
,'Formation_Damage_Index'         
,'LONG'                           
,'LAT'                            
,'Mud_Weight_ppg'                 
,'Fracture_Gradient_ppg'          
,'Depth_m'
,'pH_Level'
,'Plastic_Viscosity'
,'Viscosity_cP']

df_for_lstm = df[col]

def make_binary(df,threshold=0.5):
    df['Emulsion_Risk_binary'] = np.where(df['Emulsion_Risk'] > threshold,1,0)
    return df

def prune_model(model,amount=0.2):
    for name,_ in model.named_modules():
        try:
            prune.l1_unstructured(name,name='weight',amount=amount)
            prune.remove(name,name='weight')
        except:
            continue 

def balanced_time_series_windows(X, y, window_size=10, threshold=0.5, seed=42):
    np.random.seed(seed)
    random.seed(seed)
    
    pos_windows = []
    neg_windows = []

    for start in range(0, len(X) - window_size + 1):
        end = start + window_size
        x_win = X.iloc[start:end].values
        y_win = y.iloc[start:end].values

        pos_ratio = np.mean(y_win)

        if pos_ratio >= threshold:
            pos_windows.append((x_win, 1))
        elif pos_ratio <= (1 - threshold):
            neg_windows.append((x_win, 0))
        gc.collect()

    num_pos = len(pos_windows)
    num_neg = len(neg_windows)

    if num_neg == 0 or num_pos == 0:
        raise ValueError("No valid windows found for one of the classes. Try adjusting window_size or threshold.")

    print(f"Positive windows: {num_pos} | Negative windows: {num_neg}")

    # smarter sampling
    neg_sampled = neg_windows
    pos_sampled = random.sample(pos_windows, k=len(neg_sampled))
    gc.collect()
    combined = pos_sampled + neg_sampled
    random.shuffle(combined)

    x = np.array([item[0] for item in combined])
    y = np.array([item[1] for item in combined])

    return x, y

df_for_lstm = make_binary(df_for_lstm)

X,y = balanced_time_series_windows(X=df_for_lstm.drop(columns=['Emulsion_Risk_binary','Emulsion_Risk']),y=df_for_lstm['Emulsion_Risk_binary'])

X = np.nan_to_num(X)

tss = TimeSeriesSplit(n_splits=6)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = LSTM_with_Attention(input_dim=X.shape[2],hidden_dim=64,output_dim=2,dropout_rate=0.4,lstm_layers=2).to(device=device) 




optims = optim.Adam(model.parameters(),lr=0.0002)
lr_scheduler = CosineAnnealingWarmRestarts(optimizer=optims,T_0=1,T_mult=2)
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



df2 = pd.read_parquet('your_data.parquet')

df_for_lstm2 = df2[col]
df_for_lstm2 = make_binary(df_for_lstm2)

X2,y2 = balanced_time_series_windows(X=df_for_lstm2.drop(columns=['Emulsion_Risk_binary','Emulsion_Risk']),y=df_for_lstm2['Emulsion_Risk_binary'])

X2 = np.nan_to_num(X2)
x_test_scaled2,_ = min_max_scale_3d(X2,X2)

x_test_tensor2 = torch.FloatTensor(x_test_scaled2)
y_test_tensor2 = torch.tensor(y2)

with torch.inference_mode():
        y_pred_test2 = model(x_test_tensor2[:100].to(device)).squeeze(1)


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



