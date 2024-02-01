import torch
import numpy as np
import time
import gc

from torchvision import transforms
from matplotlib import pyplot as plt

import sys
sys.path.append('./support')
from customdataset import *
from functions import *
from metrics import *
from models import *
from operators import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# data_location = './data/'
data_location = "./data200/"

bsz = 16 # batch size
kernel_size = 3
kernel_sigma = 5.0
noise_sigma = 1e-2

A = GaussianBlur(sigma=kernel_sigma, kernel_size=kernel_size).to(device=device) # A matches the same notation in the paper
measurement_process = OperatorPlusNoise(A, noise_sigma=noise_sigma) # This maps image x* to measurement d

# initialize the model
num_channels = 3
lossfunction = torch.nn.MSELoss(reduction='sum')
learning_rate = 0.0001 # Try different
step_size = 0.001
num_epoch = 20
dncnn_kernel_size = 3

model = DEGRAD(c=num_channels, batch_size=bsz, blur_operator=A, step_size=step_size, kernel_size=dncnn_kernel_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

img_sizes = list(range(16, 128 + 1, 16)) # or other sequences of image dimensions
jacobian_based_time = np.zeros(len(img_sizes))
jacobian_free_time = np.zeros(len(img_sizes))

for i in range(len(img_sizes)):
    im_size = img_sizes[i]
    print(f'Image size: {im_size}')
    transform = transforms.Compose(
        [
            transforms.Resize((im_size, im_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    train_dataset = CelebADataset(data_location, transform=transform)
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bsz, shuffle=True, drop_last=True)

    data_batch = next(iter(train_dataloader)).to(device)

    for t in range(20):
        gc.collect()
        torch.cuda.empty_cache()
        start = time.time()
        model.jacobian_based_update(data_batch, lossfunction, optimizer)
        end = time.time()
        jacobian_based_time[i] += end - start

        gc.collect()
        torch.cuda.empty_cache()
        start = time.time()
        model.jacobian_free_update(data_batch, lossfunction, optimizer)
        end = time.time()
        jacobian_free_time[i] += end - start

    jacobian_based_time[i] /= 20
    jacobian_free_time[i] /= 20

plt.plot(img_sizes, jacobian_based_time, label='Jacobian-based')
plt.plot(img_sizes, jacobian_free_time, label='Jacobian-free')
plt.xlabel('Image size')
plt.ylabel('Avg. Time Per Batch (s)')
plt.legend()
# plt.savefig('{target directory + filename.pdf}')
plt.show()
