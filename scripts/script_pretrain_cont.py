import torch

from torchvision import transforms
from support.operators import *
from support.metrics import *
from support.functions import *
from support.customdataset import *
from support.models import *



# Load the data
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# data_location = '/users/lliu58/data/lliu58/new/Jacobian-free-Backprop-Implicit-Networks/data/'
data_location = './data/'

transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)
bsz = 32 # batch size
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
num_epoch = 30
degrad_model = DEGRAD(c=num_channels, batch_size=bsz, blur_operator=A, step_size=step_size)
degrad_model.to(device)
degrad_model.load_state_dict(torch.load('./degrad_1/weights_only.pth'))
optimizer = torch.optim.Adam(degrad_model.parameters(), lr=learning_rate)
avg_loss_epoch = []
avg_n_iters = []
avg_grad_norm = []
data_batch = iter(test_dataloader).next()
# temppath = "./data/lliu58/new/Jacobian-free-Backprop-Implicit-Networks/degrad_output_imgs/"
temppath = "./degrad_fixlr/"
for epoch in range(num_epoch):
    epoch_loss_list, epoch_n_iters_list, grad_norm_list = train_jfb(degrad_model, train_dataloader, measurement_process, lossfunction, optimizer, device)
    epoch_loss = np.mean(epoch_loss_list)
    epoch_n_iters = np.mean(epoch_n_iters_list)
    epoch_grad_norm = np.mean(grad_norm_list)
    avg_loss_epoch.append(epoch_loss)
    avg_n_iters.append(epoch_n_iters)
    avg_grad_norm.append(epoch_grad_norm)
    print("Epoch " + str(epoch+1) +" finished, average loss:" +str(epoch_loss)+ " average number of iterations: " + str(epoch_n_iters) + " out of 150, average gradient norm: "+ str(epoch_grad_norm))
    # test_batch(degrad_model, data_batch, device)
    if epoch % 10 == 0:
        plotting(avg_loss_epoch, avg_n_iters, avg_grad_norm, epoch)
    
        np.save(temppath+"avg_loss_epoch"+str(epoch), np.array(avg_loss_epoch))
        np.save(temppath+"avg_n_iters"+str(epoch), np.array(avg_n_iters))
        np.save(temppath+"avg_grad_norm"+str(epoch), np.array(avg_grad_norm))

np.save(temppath+"avg_loss_epoch", np.array(avg_loss_epoch))
np.save(temppath+"avg_n_iters", np.array(avg_n_iters))
np.save(temppath+"avg_grad_norm", np.array(avg_grad_norm))