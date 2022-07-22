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
kernel_size = 9
kernel_sigma = 5.0
noise_sigma = 1e-2

train_dataset = CelebADataset(data_location, transform=transform)
valid_dataset = CelebADataset(data_location, train=False, transform=transform)
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bsz, shuffle=True, drop_last=True)
# Want to calculate valid dataset in one batch
valid_dataloader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=bsz, shuffle=False, drop_last=True)

# A matches the same notation in the paper
A = GaussianBlur(sigma=kernel_sigma, kernel_size=kernel_size).to(device=device)
# This maps image x* to measurement d 
measurement_process = OperatorPlusNoise(A, noise_sigma=noise_sigma)

num_channels = 3
lossfunction = torch.nn.MSELoss(reduction='sum')
learning_rate = 0.01 # Try different
step_size = 0.01
num_epoch = 200
dncnn_kernel_size = 3
model = DEGRAD(c=num_channels, batch_size=bsz, blur_operator=A, step_size=step_size, kernel_size=dncnn_kernel_size)
model.to(device)


# dncnn_model = DNCNN(c=num_channels, batch_size=bsz, kernel_size=dncnn_kernel_size)
# dncnn_model.to(device)
# dncnn_model.load_state_dict(torch.load("./results/dncnn_pretrain/pretrained_weights.pth"))
# model.dncnn = dncnn_model
# Un-comment the following but comment the above four lines if you are loading trained DEGRAD model
model.load_state_dict(torch.load("./results/degrad_fixlr/trained_model2.pth"))
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


avg_loss_epoch = [] # average training loss across epochs
avg_n_iters = [] # average number of iterations across epochs
avg_grad_norm = [] # average parameters' gradient norm across epochs
valid_loss_list = [] # validation loss values across epochs
temppath = "./results/degrad_fixlr/"
lowest_loss = np.Inf
for epoch in range(num_epoch):
    epoch_loss_list, epoch_n_iters_list, grad_norm_list = train_jfb(model, train_dataloader, measurement_process, lossfunction, optimizer, device)
    epoch_loss = np.mean(epoch_loss_list)
    epoch_n_iters = np.mean(epoch_n_iters_list)
    epoch_grad_norm = np.mean(grad_norm_list)
    avg_loss_epoch.append(epoch_loss)
    avg_n_iters.append(epoch_n_iters)
    avg_grad_norm.append(epoch_grad_norm)
    print("Epoch " + str(epoch+1) +" finished, average loss:" +str(epoch_loss)+ " average number of iterations: " + str(epoch_n_iters) + ", average gradient norm: "+ str(epoch_grad_norm))
    valid_loss = valid_jfb(model, valid_dataloader, measurement_process, lossfunction, device)
    valid_loss_list.append(valid_loss)
    if valid_loss < lowest_loss:
        lowest_loss = valid_loss
        torch.save(model.state_dict(), temppath+"trained_model3.pth")
        print("epoch "+str(epoch+1)+" weights saved")

plotting(avg_loss_epoch, avg_n_iters, avg_grad_norm, epoch+1, temppath)
np.save(temppath+"avg_loss_epoch3", np.array(avg_loss_epoch))
np.save(temppath+"avg_n_iters3", np.array(avg_n_iters))
np.save(temppath+"avg_grad_norm3", np.array(avg_grad_norm))
np.save(temppath+"valid_loss_list3", np.array(valid_loss_list))
