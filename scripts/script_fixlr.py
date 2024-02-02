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
# data_location = "./data200/" 
# or other location that stores data.

transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)
bsz = 16 # batch size
kernel_size = 5
kernel_sigma = 1.0 
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
learning_rate = 0.001 # Try different
step_size = 0.001
num_epoch = 100
dncnn_kernel_size = 3
model = DEGRAD(c=num_channels, batch_size=bsz, blur_operator=A, step_size=step_size, kernel_size=dncnn_kernel_size)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

dncnn_model = DNCNN(c=num_channels, batch_size=bsz, kernel_size=dncnn_kernel_size)
dncnn_model.to(device)
dncnn_model.load_state_dict(torch.load("./results/dncnn_pretrain/pretrained_weights.pth"))
model.dncnn = dncnn_model

avg_loss_epoch = [] # average training loss across epochs
avg_n_iters = [] # average number of iterations across epochs
avg_grad_norm = [] # average parameters' gradient norm across epochs
avg_train_ssim = []
valid_loss_list = [] # validation loss values across epochs
valid_ssim_list = [] # validation ssim values across epochs
temppath = "./results/degrad_fixlr_200_0928/" ## replace by directory that stores results
lowest_loss = np.Inf
ssim_calculator = SSIM()

model.max_num_iter = 50
lowest_loss = np.Inf
for epoch in range(num_epoch):
    epoch_loss_list, epoch_n_iters_list, grad_norm_list, train_ssim_list = train_jfb(model, train_dataloader, measurement_process, lossfunction, optimizer, device, ssim_calculator)
    epoch_loss = np.mean(epoch_loss_list)
    epoch_n_iters = np.mean(epoch_n_iters_list)
    epoch_grad_norm = np.mean(grad_norm_list)
    epoch_ssim = np.mean(train_ssim_list)
    avg_loss_epoch.append(epoch_loss)
    avg_n_iters.append(epoch_n_iters)
    avg_grad_norm.append(epoch_grad_norm)
    avg_train_ssim.append(epoch_ssim)
    valid_loss, valid_ssim = valid_jfb(model, valid_dataloader, measurement_process, lossfunction, device, ssim_calculator)
    valid_loss_list.append(valid_loss)
    valid_ssim_list.append(valid_ssim)
    if valid_loss < lowest_loss:
        lowest_loss = valid_loss
        torch.save(model.state_dict(), temppath+"trained_model.pth")
        print("Epoch "+str(epoch+1)+" weights saved")
    print(f"Epoch {epoch+1} finished, avg loss {epoch_loss:.3f}, avg #iters {epoch_n_iters:.3f}, avg grad norm {epoch_grad_norm:.3f}, train ssim {epoch_ssim:.3f}, valid loss {valid_loss:.3f}, valid ssim {valid_ssim:.3f}")

# plotting(avg_loss_epoch, avg_n_iters, avg_grad_norm, epoch+1, temppath)
np.save(temppath+"avg_loss_epoch", np.array(avg_loss_epoch))
np.save(temppath+"avg_n_iters", np.array(avg_n_iters))
np.save(temppath+"avg_grad_norm", np.array(avg_grad_norm))
np.save(temppath+"avg_train_ssim", np.array(avg_train_ssim))
np.save(temppath+"valid_loss_list", np.array(valid_loss_list))
np.save(temppath+"valid_ssim_list", np.array(valid_ssim_list))
