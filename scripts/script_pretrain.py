import torch
import numpy as np
from torchvision import transforms
import sys
sys.path.append('./support')
from customdataset import *
from functions import *
from metrics import *
from models import *
from operators import *



# Load the data
device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_location = './data/'

transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)
bsz = 32 # batch size
blur_kernel_size = 5
kernel_sigma = 5.0
noise_sigma = 1e-2

train_dataset = CelebADataset(data_location, transform=transform)
test_dataset = CelebADataset(data_location, train=False, transform=transform)
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bsz, shuffle=True, drop_last=True)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bsz, shuffle=False, drop_last=True)

# A matches the same notation in the paper
A = GaussianBlur(sigma=kernel_sigma, kernel_size=blur_kernel_size).to(device=device)
# This maps image x* to measurement d 
measurement_process = OperatorPlusNoise(A, noise_sigma=noise_sigma)

num_channels = 3
lossfunction = torch.nn.MSELoss(reduction='sum')
learning_rate = 0.001
num_epoch = 200
dncnn_kernel_size = 3

# model initialization
dncnn_model = DNCNN(c=num_channels, batch_size=bsz, kernel_size=dncnn_kernel_size).to(device)
optimizer = torch.optim.Adam(dncnn_model.parameters(), lr=learning_rate)
avg_loss_epoch = []
avg_grad_norm = []
data_batch = iter(test_dataloader).next()
temppath = "./results/dncnn_pretrain/"
for epoch in range(num_epoch):
    epoch_loss_list, grad_norm_list = train_dncnn(dncnn_model, train_dataloader, measurement_process, lossfunction, optimizer, device)
    epoch_loss = np.mean(epoch_loss_list)
    epoch_grad_norm = np.mean(grad_norm_list)
    avg_loss_epoch.append(epoch_loss)
    avg_grad_norm.append(epoch_grad_norm)
    print("Epoch " + str(epoch+1) +" finished, average loss:" +str(epoch_loss) + " average gradient norm: "+ str(epoch_grad_norm))

torch.save(dncnn_model.state_dict(), temppath+"pretrained_weights.pth")
np.save(temppath+"avg_loss_epoch", np.array(avg_loss_epoch))
np.save(temppath+"avg_grad_norm", np.array(avg_grad_norm))