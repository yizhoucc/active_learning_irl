import pickle
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from notification import notify


note = 'data'
agent_name = 'ppo_baseline_0331_5cost'
with open('data/{}_{}'.format(agent_name, note), 'rb') as f:
    x_data, ys = pickle.load(f)
y_data = [torch.tensor(y).view(-1) for y in ys]

# use a small subset for testing
# x_data, y_data = x_data[:100], y_data[:100]

y_data = torch.stack(y_data)
y_data=y_data[:,[3,9,11]]
y_data=y_data/(torch.max(y_data,axis=0)[0])



# Find the maximum length of your time series data
max_length = max([len(d) for d in x_data])

# Pad your time series data with zeros at the front
x_data = [torch.tensor(x) for x in x_data]
padded_data = pad_sequence(x_data, batch_first=True, padding_value=0)
padded_data.shape  # ntrial, ts, input feature

################# normalize
# x_data[0].shape
# plt.hist(x_data[0][:,4])

# means = torch.mean(input_data, dim=0)
# stds = torch.std(input_data, dim=0)
# normalized = (input_data - means) / stds

######################

dataset = TensorDataset(padded_data)

# DataLoader from dataset
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size)

num_samples = len(x_data)
indices = torch.randperm(num_samples)
train_size = int(num_samples*0.9)
val_size = int(num_samples*0.9*0.1)
train_indices = indices[:train_size]
val_indices = indices[train_size-val_size:train_size]
test_indices = indices[train_size:]


train_data = torch.utils.data.TensorDataset(
    padded_data[train_indices], y_data[train_indices])
val_data = torch.utils.data.TensorDataset(
    padded_data[val_indices], y_data[val_indices])
test_data = torch.utils.data.TensorDataset(
    padded_data[test_indices], y_data[test_indices])

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)


input_size = padded_data.shape[2]
num_layers = 2
hidden_size = 32
output_size=y_data[0].shape[0]


class GRUNet(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUNet, self).__init__()
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.gru=nn.GRU(input_size, hidden_size,
                          num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = out[:, -1, :]
        out = self.fc(out)
        return out, h




model = GRUNet(input_size=input_size, hidden_size=hidden_size,
               num_layers=num_layers, output_size=output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 30

for epoch in range(num_epochs):
    model.train()
    for x_batch, y_batch in train_loader:
        pred, _ = model(x_batch.float(), None)

        optimizer.zero_grad()
        loss = criterion(pred, y_batch)
        torch.nn.utils.clip_gradnorm(model.parameters(), 5)
        loss.backward(retain_graph=True)
        optimizer.step()

    # eval
    val_loss = 0.0
    total = 0
    model.eval()

    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs, hidden = model(inputs.float(), None)
            val_loss += criterion(outputs, targets)
            total += targets.size(0)
            print(torch.mean(abs((outputs**2 - targets**2))**0.5,axis=0))
            plt.scatter(outputs[:,0],targets[:,0], alpha=0.1)
    plt.axis('equal')
    plt.show()
    avg_val_loss = val_loss / len(val_loader)
    print(
        f"Epoch {epoch+1}: Train Loss = {loss:.4f}, Val Loss = {avg_val_loss:.4f}")
    
    # test
    val_loss = 0.0
    total = 0
    model.eval()

    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs, hidden = model(inputs.float(), None)
            val_loss += criterion(outputs, targets)
            total += targets.size(0)
            plt.scatter(outputs[:,0],targets[:,0], alpha=0.1)
    plt.axis('equal')
    plt.title('phi length')
    plt.show()

    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs, hidden = model(inputs.float(), None)
            val_loss += criterion(outputs, targets)
            total += targets.size(0)
            # print(torch.mean(abs((outputs**2 - targets**2))**0.5,axis=0))
            plt.scatter(outputs[:,1],targets[:,1], alpha=0.1)
    plt.axis('equal')
    plt.title('theta length')
    plt.show()

    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs, hidden = model(inputs.float(), None)
            val_loss += criterion(outputs, targets)
            total += targets.size(0)
            # print(torch.mean(abs((outputs**2 - targets**2))**0.5,axis=0))
            plt.scatter(outputs[:,2],targets[:,2], alpha=0.1)
    plt.axis('equal')
    plt.title('theta cost')
    plt.show()

    notify()

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss_fn': criterion
}, 'data/{}_{}.pt'.format(agent_name, note))