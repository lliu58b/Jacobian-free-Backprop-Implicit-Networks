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
        self.threshold = 1e-3
        self.bsz = batch_size

        # Trainable parameters
        # chen.spectral_norm may be replaced by torch.nn.utils.spectral_norm
        self.dncnn = DNCNN(c=self.nchannels, batch_size=self.bsz, kernel_size=kernel_size)
    
    def forward(self, measurement):
        self.eta = torch.clamp(self.eta, min=1e-6, max=2)
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
    def __init__(self, c, blur_operator, step_size, num_layers=17, kernel_size=3, features=64):
        super().__init__()

        self.nchannels = c
        self.nlayers = num_layers
        self.ksz = kernel_size
        self.nfeatures = features
        self.padding = self.ksz // 2
        self.A = blur_operator
        self.eta = torch.nn.Parameter(step_size*torch.ones(()))
        self.max_num_iter = 100
        self.threshold = 1e-3

        # Trainable parameters

        # Get things out of spectral_norm_chen
        layers = []
        layers.append(chen.spectral_norm(torch.nn.Conv2d(in_channels=self.nchannels, out_channels=self.nfeatures, kernel_size=self.ksz, padding=self.padding, bias=False)))
        layers.append(torch.nn.ReLU(inplace=True))
        for _ in range(self.nlayers-2):
            layers.append(chen.spectral_norm(torch.nn.Conv2d(in_channels=self.nfeatures, out_channels=self.nfeatures, kernel_size=self.ksz, padding=self.padding, bias=False)))
            layers.append(torch.nn.BatchNorm2d(self.nfeatures))
            layers.append(torch.nn.ReLU(inplace=True))
        layers.append(chen.spectral_norm(torch.nn.Conv2d(in_channels=self.nfeatures, out_channels=self.nchannels, kernel_size=self.ksz, padding=self.padding, bias=False)))
        # Put them altogether and call it dncnn
        self.dncnn = torch.nn.Sequential(*layers)
    
    def forward(self, measurement):
        with torch.no_grad():
            xstar = self.find_fixed_point(measurement)
        Txstar = self.dncnn(xstar - self.eta * self.A.adjoint(torch.sub(self.A.forward(xstar), measurement)))
        return Txstar

    def find_fixed_point(self, measurement):
        with torch.no_grad():
            x0 = self.A.adjoint(measurement)
            temp = x0
            for i in range(self.max_num_iter):
                x = self.dncnn(temp - self.eta * (self.A.adjoint(torch.sub(self.A.forward(temp), measurement))))
                if torch.norm(torch.sub(x, temp)) < self.threshold:
                    return x
                else:
                    temp = x
        return temp

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
