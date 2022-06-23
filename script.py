import torch

from torchvision import transforms
from operators import *
from metrics import *
from functions import *
from customdataset import *


device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_location = './data/'

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

