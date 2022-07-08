import torch
import numpy as np

from customdataset import *
from operators import *
from metrics import *
from torchvision import transforms
from torch.nn import utils

class DNCNN(torch.nn.Module):
    def __init__(self, c, batch_size, num_layers=17, kernel_size=3, features=64):
        super().__init__()
        self.nchannels = c
        self.nlayers = num_layers
        self.ksz = kernel_size
        self.nfeatures = features
        self.padding = self.ksz // 2
        self.bsz = batch_size

        layers = []
        layers.append(utils.spectral_norm(torch.nn.Conv2d(in_channels=self.nchannels, out_channels= self.nfeatures, kernel_size=self.ksz, padding=self.padding, bias=False)))
        layers.append(torch.nn.ReLU(inplace=True))
        for _ in range(self.nlayers-2):
            layers.append(utils.spectral_norm(torch.nn.Conv2d(in_channels=self.nfeatures, out_channels=self.nfeatures, kernel_size=self.ksz, padding=self.padding, bias=False)))
            layers.append(torch.nn.BatchNorm2d(self.nfeatures))
            layers.append(torch.nn.ReLU(inplace=True))
        layers.append(utils.spectral_norm(torch.nn.Conv2d(in_channels=self.nfeatures, out_channels=self.nchannels, kernel_size=self.ksz, padding=self.padding, bias=False)))
        # Put them altogether and call it dncnn
        self.dncnn = torch.nn.Sequential(*layers)
    
    def forward(self, measurement):
        return self.dncnn(measurement)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_location = './data/'

transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

def training(model, loader, operator, loss_function, optimizer, device):
    model.train()
    L = []
    for batch_idx, X in enumerate(loader):
        if batch_idx % 10 == 0:
            print(f'current batch: {batch_idx}')
        X = X.to(device)
        d = operator.forward(X)
        pred = model(d)
        batch_loss = torch.div(loss_function(pred, X), model.bsz)
        batch_loss.backward()
        optimizer.step()
        L.append(batch_loss.item())
    return L

bsz = 64 # batch size
kernel_size = 5
kernel_sigma = 5
noise_sigma = 1e-2

# get training and validation dataset
train_dataset = CelebADataset(data_location, train=True, transform=transform)
valid_dataset = CelebADataset(data_location, train=False, transform=transform)
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                     batch_size=bsz, shuffle=True, drop_last=True)
valid_dataloader = torch.utils.data.DataLoader(dataset=valid_dataset, 
                    batch_size=bsz, shuffle=False, drop_last=True)

# A matches the same notation in the paper
A = GaussianBlur(sigma=kernel_sigma, kernel_size=kernel_size).to(device=device)
# This maps image x* to measurement d
measurement_process = OperatorPlusNoise(A, noise_sigma=noise_sigma)

num_channels = 3
lossfunction = torch.nn.MSELoss(reduction='sum')
learning_rate = 2e-4
num_epoch = 200
dncnn_model = DNCNN(c=num_channels, batch_size=bsz)
dncnn_model.to(device)
dncnn_model.load_state_dict(torch.load('./dncnn_pretrain/weights3.pth'))
dncnn_model.eval()
optimizer = torch.optim.Adam(dncnn_model.parameters(), lr=learning_rate)

avg_loss = []
for epoch in range(num_epoch):
    epoch_loss_list = training(model=dncnn_model, loader=train_dataloader, 
                                operator=A, loss_function=lossfunction, 
                                optimizer=optimizer, device=device)
    epoch_avg_loss = np.mean(epoch_loss_list)
    avg_loss.append(epoch_avg_loss)
    print("Epoch "+str(epoch)+" finished, average loss: " +str(epoch_avg_loss))
torch.save(dncnn_model.state_dict(), './dncnn_pretrain/weights5.pth')
np.save("./dncnn_pretrain/avglossvsepoch5", np.array(avg_loss))

