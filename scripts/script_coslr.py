import torch

from torchvision import transforms
from support.operators import *
from support.metrics import *
from support.functions import *
from support.customdataset import *
from support.models import *


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
bsz = 64 # batch size
kernel_size = 5
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
learning_rate = 0.0001
step_size = 0.001
num_epoch = 100
Tmax = 100
model = DEGRAD(c=num_channels, batch_size=bsz, blur_operator=A, step_size=step_size)
model.to(device)
dncnn_model = DNCNN(c=num_channels, batch_size=bsz)
dncnn_model.to(device)
dncnn_model.load_state_dict(torch.load("./dncnn_pretrain/weights4.pth"))
model.dncnn = dncnn_model
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Tmax, eta_min=0, last_epoch=- 1, verbose=False)
avg_loss_epoch = []
avg_n_iters = []
avg_grad_norm = []
temppath = "./degrad_coslr/"
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
    if valid_loss < lowest_loss:
        lowest_loss = valid_loss
        torch.save(model.dncnn.state_dict(), temppath+"dncnn_weights.pth")
        print("epoch "+str(epoch+1)+" weights saved")
    if epoch % 10 == 0:
        plotting(avg_loss_epoch, avg_n_iters, avg_grad_norm, epoch, temppath)
    
        np.save(temppath+"avg_loss_epoch"+str(epoch), np.array(avg_loss_epoch))
        np.save(temppath+"avg_n_iters"+str(epoch), np.array(avg_n_iters))
        np.save(temppath+"avg_grad_norm"+str(epoch), np.array(avg_grad_norm))
    scheduler.step(epoch)

torch.save(model.state_dict(), temppath+'trained_model.pth')
torch.save(scheduler.state_dict(), temppath+"scheduler.pth")
np.save(temppath+"avg_loss_epoch", np.array(avg_loss_epoch))
np.save(temppath+"avg_n_iters", np.array(avg_n_iters))
np.save(temppath+"avg_grad_norm", np.array(avg_grad_norm))
