import torch

from torchvision import transforms
from operators import *
from metrics import *
from functions import *
from customdataset import *
from DEmodels import *



# Load the data
device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_location = '/users/lliu58/data/lliu58/Jacobian-free-Backprop-Implicit-Networks/data/'
# data_location = './data/'

transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)
bsz = 16 # batch size
kernel_size = 5
kernel_sigma = 5.0
noise_sigma = 1e-2

train_dataset = CelebADataset(data_location, transform=transform)
test_dataset = CelebADataset(data_location, train=False, transform=transform)
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bsz, shuffle=True, drop_last=True)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bsz, shuffle=False, drop_last=True)

# A matches the same notation in the paper
A = GaussianBlur(sigma=kernel_sigma, kernel_size=kernel_size).to(device=device)
# This maps image x* to measurement d 
measurement_process = OperatorPlusNoise(A, noise_sigma=noise_sigma)

num_channels = 3
lossfunction = torch.nn.MSELoss(reduction='sum')
learning_rate = 0.001
step_size = 0.001
num_epoch = 100
degrad_model = DEGRAD(c=num_channels, blur_operator=A, step_size=step_size)
degrad_model.to(device)
optimizer = torch.optim.Adam(degrad_model.parameters(), lr=learning_rate)
avg_loss_epoch = []
avg_n_iters = []
avg_grad_norm = []
data_batch = iter(test_dataloader).next()
for epoch in range(num_epoch):
    epoch_loss_list, epoch_n_iters_list, grad_norm_list = train_jfb(degrad_model, train_dataloader, measurement_process, lossfunction, optimizer, device)
    epoch_loss = np.mean(epoch_loss_list)
    epoch_n_iters = np.mean(epoch_n_iters_list)
    epoch_grad_norm = np.mean(grad_norm_list)
    avg_loss_epoch.append(epoch_loss)
    avg_n_iters.append(epoch_n_iters)
    avg_grad_norm.append(epoch_grad_norm)
    print("Epoch " + str(epoch+1) +" finished, average loss:" +str(epoch_loss)+ " average number of iterations: " + str(epoch_n_iters) + "out of 100, average gradient norm: "+ str(epoch_grad_norm))
    # test_batch(degrad_model, data_batch, device)
    if epoch % 10 == 0:
        plotting(avg_loss_epoch, avg_n_iters, avg_grad_norm, epoch)