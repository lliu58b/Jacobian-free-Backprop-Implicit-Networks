import torch
import sys
sys.path.append('./support')
import spectral_norm_chen as chen

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
    
    def current_grad_norm(self):
        with torch.no_grad():
            S = 0
            for p in self.parameters():
                if p.grad==None:
                    continue # some gradients don't exist
                param_norm = torch.norm(p.grad.detach().data)
                S += param_norm.item() ** 2
            S = S ** 0.5
        return S

class DEGRAD(torch.nn.Module):
    def __init__(self, c, batch_size, blur_operator, step_size, kernel_size):
        super().__init__()

        self.nchannels = c
        self.A = blur_operator
        self.eta = torch.nn.Parameter(step_size*torch.ones(()))
        self.max_num_iter = 150
        self.threshold = 1e-3 # this is specified in Gilton et al., 2021
        self.bsz = batch_size

        # Trainable parameters
        # chen.spectral_norm may be replaced by torch.nn.utils.spectral_norm
        self.dncnn = DNCNN(c=self.nchannels, batch_size=self.bsz, kernel_size=kernel_size)
    
    def forward(self, measurement):
        with torch.no_grad():
            xstar, n_iters = self.find_fixed_point(measurement)
        Txstar = xstar - self.eta * (self.A.adjoint(torch.sub(self.A.forward(xstar), measurement)) + self.dncnn(xstar))
        return Txstar, n_iters

    def find_fixed_point(self, measurement):
        with torch.no_grad():
            x0 = self.A.adjoint(measurement)
            temp = x0
            for i in range(self.max_num_iter):
                x = temp - self.eta * (self.A.adjoint(torch.sub(self.A.forward(temp), measurement)) + self.dncnn(temp))
                if self._converge(x, temp):
                    return x, (i+1)
                else:
                    temp = x
        return temp, self.max_num_iter
    
    def current_grad_norm(self):
        with torch.no_grad():
            S = 0
            for p in self.parameters():
                if p.grad==None:
                    continue # some gradients don't exist
                param_norm = torch.norm(p.grad.detach().data)
                S += param_norm.item() ** 2
            S = S ** 0.5
        return S
    
    def _converge(self, x1, x2):
        x1 = torch.reshape(x1, [self.bsz, -1])
        x2 = torch.flatten(x2, start_dim=1)
        n1 = torch.max(torch.norm(torch.sub(x1, x2), dim=1))
        n2 = torch.max(torch.norm(x2, dim=1))
        if  n1/n2 < self.threshold:
            return True
        else:
            return False

class DEPROX(torch.nn.Module):
    def __init__(self, c, batch_size, blur_operator, step_size, kernel_size=3):
        super().__init__()

        self.nchannels = c
        self.A = blur_operator
        self.eta = torch.nn.Parameter(step_size*torch.ones(()))
        self.max_num_iter = 100
        self.threshold = 1e-3
        self.bsz = batch_size

        # Trainable parameters
        self.dncnn = DNCNN(c=self.nchannels, batch_size=self.bsz, kernel_size=kernel_size)
    
    def forward(self, measurement):
        with torch.no_grad():
            xstar, n_iters = self.find_fixed_point(measurement)
        Txstar = self.dncnn(xstar - self.eta * self.A.adjoint(torch.sub(self.A.forward(xstar), measurement)))
        return Txstar, n_iters

    def find_fixed_point(self, measurement):
        with torch.no_grad():
            x0 = self.A.adjoint(measurement)
            temp = x0
            for i in range(self.max_num_iter):
                x = self.dncnn(temp - self.eta * (self.A.adjoint(torch.sub(self.A.forward(temp), measurement))))
                if self._converge(x, temp):
                    return x, (i+1)
                else:
                    temp = x
        return temp, self.max_num_iter

    def current_grad_norm(self):
        with torch.no_grad():
            S = 0
            for p in self.parameters():
                if p.grad==None:
                    continue # some gradients don't exist
                param_norm = torch.norm(p.grad.detach().data)
                S += param_norm.item() ** 2
            S = S ** 0.5

        return S

    def _converge(self, x1, x2):
        x1 = torch.reshape(x1, [self.bsz, -1])
        x2 = torch.flatten(x2, start_dim=1)
        n1 = torch.max(torch.norm(torch.sub(x1, x2), dim=1))
        n2 = torch.max(torch.norm(x2, dim=1))
        if  n1/n2 < self.threshold:
            return True
        else:
            return False


class DEADMM(torch.nn.Module):
    def __init__(self, c, batch_size, blur_operator, step_size, sample, num_layers=17, kernel_size=3, features=64):
        super().__init__()

        self.nchannels = c
        self.nlayers = num_layers
        self.ksz = kernel_size
        self.nfeatures = features
        self.padding = self.ksz // 2 # padding keeps the pictures at the right dimension
        self.A = blur_operator # operator, not the matrix A, which is A_mat
        self.alpha = step_size
        self.max_num_iter = 100
        self.threshold = 1e-3
        self.sample = sample
        self.A_mat = self.calculate_A_mat()
        self.bsz = batch_size
        
        # A = torch.eye(H*W) + torch.mul(self.alpha, (self.A_mat.adjoint(self.A_mat))) # torch.eye is for creating the identity matrix with size H*W so that it matches the size of A^T A. for (I+alpha A^T A)^(-1) later
        # U, S, Vt = torch.linalg.svd(A, full_matrices=False) # svd produces the 3 matrices
        # S1 = torch.div(1, torch.diag(S)) # a row vector with 1/each singular value
        #print()

        # self.invert = torch.matmul(torch.multiply(V, S1), torch.transpose(U)) # torch.multiply(V, S1) is the same as matrix multiplication


        # Trainable parameters
        layers = []
        layers.append(chen.spectral_norm(torch.nn.Conv2d(in_channels=self.nchannels, out_channels= self.nfeatures, kernel_size=self.ksz, padding=self.padding, bias=False)))
        layers.append(torch.nn.ReLU(inplace=True))
        for _ in range(self.nlayers-2):
            layers.append(chen.spectral_norm(torch.nn.Conv2d(in_channels=self.nfeatures, out_channels=self.nfeatures, kernel_size=self.ksz, padding=self.padding, bias=False)))
            layers.append(torch.nn.BatchNorm2d(self.nfeatures))
            layers.append(torch.nn.ReLU(inplace=True))
        layers.append(chen.spectral_norm(torch.nn.Conv2d(in_channels=self.nfeatures, out_channels=self.nchannels, kernel_size=self.ksz, padding=self.padding, bias=False)))
        # Put them altogether and call it dncnn
        self.dncnn = torch.nn.Sequential(*layers)
        
    # To change: forward and find_fixed_point

    def forward(self, measurement):
        with torch.no_grad():
            xstar, ustar = self.find_fixed_point(measurement)
        Tzstar = self.dncnn(torch.sub(xstar, ustar))
        Txstar = torch.matmul(self.invert, torch.mul(self.alpha, self.A.adjoint(measurement)) + Tzstar + ustar)
        return Txstar

    def find_fixed_point(self, measurement): #y is measurement
        with torch.no_grad():
            x0 = self.A.adjoint(measurement) # x0 is initialized as Ay
            z0 = torch.zeros(x0.shape) #z0 is initialized as all 0's
            u0 = 0

            tempx = x0
            tempz = z0
            tempu = u0
            for i in range(self.max_num_iter):
                z = self.dncnn(torch.sub(tempx, tempu)) # self.dncnn is R_theta, torch.sub means tempx-tempu
                x = torch.matmul(self.invert, torch.mul(self.alpha, self.A.adjoint(measurement)) + z + tempu) # use z, tempu
                u = tempu + z - x # use tempu, x, z, because x, z have been updated
                if torch.sqrt(torch.square(torch.norm(torch.sub(x, tempx)))+torch.square(torch.norm(torch.sub(u, tempu)))) < self.threshold: # if norm is below error threshold, return
                    return x, u
                else:
                    tempx = x
                    tempz = z
                    tempu = u
        return tempx, tempu
    
    def calculate_A_mat(self):

        (num_channels, H, W) = self.sample[0, :, :, :].shape
        A_kernel = torch.squeeze(A.gaussian_kernel[0, :, :])
        A_mat = torch.zeros([H*W, H*W])

        # Apply to standard basis in R^{H * W} to construct A
        for i in range(H*W): 
            basis = torch.zeros([num_channels, H*W]).to('cuda')
            basis[:, i] = 1
            basis = torch.reshape(basis, [num_channels, H, W])
            basis = A.forward(basis) 
            A_mat[i, :] = torch.reshape(torch.squeeze(basis[0, :, :]), [1, -1])
            A_mat = torch.transpose(A_mat, 0, 1)

        return A_mat