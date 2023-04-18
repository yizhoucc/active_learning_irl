import torch
forcemaxlen=50
from torch import nn
device='cpu'
from copy import deepcopy
from plot_ult import *


def vis(true, pred,loss=0, ind=None):
    if ind is None:
        ind=torch.randint(low=0, high=len(true), size=(1,))
    maxlen=len(true[0])
    ts=np.linspace(0,maxlen/10, maxlen)
    plt.plot(ts,true.clone().detach()[0], color='k', label='actual data')
    plt.plot(ts,pred.clone().detach()[0], color='r', label='LSTM AE reconstructed')
    plt.xlabel('time [s]')
    plt.ylabel('control [a.u.]')
    plt.ylim(-1.1,1.1)
    quickleg(ax=plt.gca(), bbox_to_anchor=(-0.2,0))
    quickspine(ax=plt.gca())
    plt.title('loss={:.4f}'.format(loss))
    plt.show()
    
def forward(x,t=50):
    out=torch.zeros(x.size(0),t, x.size(1))
    out.shape
    # for tt in range(t):
    #     x=0.77*x+torch.rand(x.size())*0.002
    #     out[:,tt,:]=x
    # out[:,:,1]=out[:,:,1]-1 #+ torch.sin(torch.linspace(0,6,t))*0.3
    out[:,:,0]=out[:,:,0] + torch.sin(torch.linspace(0,6,t))*0.4

    return out

x=torch.rand(50,1)


X=forward(x,t=forcemaxlen)-0.5
vis(X,X)

# seqlen=torch.randint(30,40,(50,))
seqlen=[30]*50

# plt.plot(X[:,:,0].T)

class Encoder(nn.Module):
  def __init__(self, seq_len, n_features, embedding_dim=64):
    super(Encoder, self).__init__()
    self.seq_len, self.n_features = seq_len, n_features
    self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim
    self.rnn1 = nn.LSTM(
      input_size=n_features,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True
    )
    self.rnn2 = nn.LSTM(
      input_size=self.hidden_dim,
      hidden_size=embedding_dim,
      num_layers=1,
      batch_first=True
    )
  def forward(self, x):
    x = x.reshape((1, self.seq_len, self.n_features))
    x, (_, _) = self.rnn1(x)
    x, (hidden_n, _) = self.rnn2(x)
    return hidden_n.reshape((self.n_features, self.embedding_dim))
  
class Decoder(nn.Module):
  def __init__(self, seq_len, input_dim=64, n_features=1):
    super(Decoder, self).__init__()
    self.seq_len, self.input_dim = seq_len, input_dim
    self.hidden_dim, self.n_features = 2 * input_dim, n_features
    self.rnn1 = nn.LSTM(
      input_size=input_dim,
      hidden_size=input_dim,
      num_layers=1,
      batch_first=True
    )
    self.rnn2 = nn.LSTM(
      input_size=input_dim,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True
    )
    self.output_layer = nn.Linear(self.hidden_dim, n_features)
  def forward(self, x):
    x = x.repeat(self.seq_len, self.n_features)
    x = x.reshape((self.n_features, self.seq_len, self.input_dim))
    x, (hidden_n, cell_n) = self.rnn1(x)
    x, (hidden_n, cell_n) = self.rnn2(x)
    x = x.reshape((self.seq_len, self.hidden_dim))
    return self.output_layer(x)
  
class RecurrentAutoencoder(nn.Module):
  def __init__(self, seq_len, n_features, embedding_dim=64):
    super(RecurrentAutoencoder, self).__init__()
    self.encoder = Encoder(seq_len, n_features, embedding_dim).to(device)
    self.decoder = Decoder(seq_len, embedding_dim, n_features).to(device)
  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x
  

model = RecurrentAutoencoder(seq_len=forcemaxlen, n_features=1, embedding_dim=8)

train_dataset=X
val_dataset=X
n_epochs=50
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
criterion = nn.L1Loss(reduction='sum').to(device)
history = dict(train=[], val=[])
best_model_wts = deepcopy(model.state_dict())
best_loss = 10000.0
for epoch in range(1, n_epochs + 1):
  model = model.train()
  train_losses = []
  for seq_true,thelen in zip(train_dataset,seqlen):
    seq_true=seq_true.view(1,forcemaxlen,1)
    optimizer.zero_grad()
    seq_true = seq_true.to(device)
    seq_pred = model(seq_true)
    loss = criterion(torch.flip(seq_pred, dims=(0,1))[:thelen,:], seq_true[:,:thelen,:])
    loss.backward()
    max_norm = 1 # Example value for maximum norm
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

    optimizer.step()
    train_losses.append(loss.item())

  val_losses = []
  model = model.eval()
  with torch.no_grad():
    for seq_true in val_dataset:
      seq_true = seq_true.to(device)
      seq_pred = model(seq_true)
      loss = criterion(seq_pred, seq_true)
      val_losses.append(loss.item())

    ind=torch.randint(0,len(val_dataset), size=(1,))
    seq_pred = model(val_dataset[ind])
    vis((val_dataset[ind]),torch.flip(seq_pred.unsqueeze(0), dims=(0,1)),train_loss)
    vis((val_dataset[ind][:,:thelen,:]),torch.flip(seq_pred[:thelen,:].unsqueeze(0), dims=(0,1)),train_loss)


  train_loss = np.mean(train_losses)
  val_loss = np.mean(val_losses)
  history['train'].append(train_loss)
  history['val'].append(val_loss)
  if val_loss < best_loss:
    best_loss = val_loss
    best_model_wts = deepcopy(model.state_dict())
  print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')
model.load_state_dict(best_model_wts)






with torch.no_grad():
  rnn = nn.LSTM(1, 1, 1)
  input = torch.randn(1, 1, 1)
  h0 = torch.randn(1, 1, 1)
  c0 = torch.randn(1, 1, 1)
  res=[]
  output, (hn, cn) = rnn(input, (h0, c0))
  for _ in range(100):
    output, (hn, cn) = rnn(input, (hn, cn))
    res.append(output)
plt.plot(res)


