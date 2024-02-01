import torch
import sys
sys.path.append('./support')
import spectral_norm_chen as chen

from torch.nn import utils
from functions import cg_batch


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
        # torch.nn.utils.spectral_norm may be replaced by chen.spectral_norm
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
        self.max_num_iter = 50
        self.threshold = 1e-3 # this is specified in Gilton et al., 2021
        self.bsz = batch_size

        # Trainable parameters
        self.dncnn = DNCNN(c=self.nchannels, batch_size=self.bsz, kernel_size=kernel_size)

    def forward(self, d):
        with torch.no_grad():
            xstar, n_iters = self.find_fixed_point(d)
        Txstar = self.g(d, xstar)
        return Txstar, n_iters

    def find_fixed_point(self, d):
        with torch.no_grad():
            x0 = self.A.adjoint(d)
            xstar, num_iter = self.anderson(lambda X: self.g(d, X), x0, tol=self.threshold)
        return xstar, num_iter

    def g(self, d, x):
        return (x - self.eta * (self.A.adjoint(torch.sub(self.A.forward(x), d)) + self.dncnn(x)))

    def anderson(self, f, x0, m=5, lam=1e-4, max_iter=50, tol=1e-2, beta = 1.0):
        #######################################################################
        # This method is obtained from NeurIPS 2020 tutorial on Deep Implicit Layers - Neural ODEs, 
        # Deep Equilibirum Models, and Beyond, created by Zico Kolter, David Duvenaud, and Matt Johnson, 
        # Chapter 4: https://implicit-layers-tutorial.org/deep_equilibrium_models/
        #######################################################################
        """ Anderson acceleration for fixed point iteration. """
        bsz, d, H, W = x0.shape
        X = torch.zeros(bsz, m, d*H*W, dtype=x0.dtype, device=x0.device)
        F = torch.zeros(bsz, m, d*H*W, dtype=x0.dtype, device=x0.device)
        X[:,0], F[:,0] = x0.view(bsz, -1), f(x0).view(bsz, -1)
        X[:,1], F[:,1] = F[:,0], f(F[:,0].view_as(x0)).view(bsz, -1)

        H = torch.zeros(bsz, m+1, m+1, dtype=x0.dtype, device=x0.device)
        H[:,0,1:] = H[:,1:,0] = 1
        y = torch.zeros(bsz, m+1, 1, dtype=x0.dtype, device=x0.device)
        y[:,0] = 1

        K = 1
        temp = None
        for k in range(2, max_iter):
            n = min(k, m)
            G = F[:,:n]-X[:,:n]
            H[:,1:n+1,1:n+1] = torch.bmm(G,G.transpose(1,2)) + lam*torch.eye(n, dtype=x0.dtype,device=x0.device)[None]
            alpha = torch.linalg.solve(H[:,:n+1,:n+1],y[:,:n+1].reshape([bsz, -1]))
            alpha = alpha[:, 1:n+1]  # (bsz x n)

            X[:,k%m] = beta * (alpha[:,None] @ F[:,:n])[:,0] + (1-beta)*(alpha[:,None] @ X[:,:n])[:,0]
            F[:,k%m] = f(X[:,k%m].view_as(x0)).view(bsz, -1)

            if k == 2:
                temp = torch.reshape(X[:,k%m], [bsz, -1])
            else:
                temp2 = torch.reshape(X[:,k%m], [bsz, -1])
                maxnorm1 = torch.max(torch.norm(torch.sub(temp2, temp), dim=1))
                maxnorm2 = torch.max(torch.norm(temp2, dim=1))
                if maxnorm1/maxnorm2 < tol:
                    break
                else:
                    K += 1
                    temp = temp2 # go to next iteration

        return X[:,k%m].view_as(x0), K

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

    def jacobian_based_update(self, X, loss_function, optimizer):
        #######################################################################
        # This part is an adaptation of Jacobian-based optimization by
        # Fung, Samy Wu, et al. "Jfb: Jacobian-free backpropagation for
        # implicit networks." Proceedings of the AAAI Conference on Artificial
        # Intelligence. Vol. 36. No. 6. 2022, obtained at 
        # https://github.com/Typal-Research/jacobian_free_backprop/
        #######################################################################
        d = self.A.forward(X) # actual input of the network
        with torch.no_grad():
            xstar, n_iters = self.find_fixed_point(d)
        xstar.requires_grad = True
        Txstar = self.g(d, xstar)

        # compute dl/dx*
        loss = loss_function(Txstar, X)
        dldx = torch.autograd.grad(
            outputs = loss, inputs = Txstar,
            retain_graph = True, create_graph = True,
            only_inputs = True)[0]

        # compute dl/dx* J^T
        dldx_dTdx = torch.autograd.grad(
            outputs = Txstar, inputs = xstar, grad_outputs = dldx,
            retain_graph = True, create_graph = True,
            only_inputs = True
        )[0]

        dldx_J = dldx = dldx_dTdx

        dldx_JT = torch.autograd.grad(
            outputs = dldx_J, inputs = dldx, grad_outputs = dldx,
            retain_graph = True, create_graph = True,
            only_inputs = True
        )[0]

        dldx_JT = dldx_JT.detach()
        # vectorize channels (when R is a CNN)
        dldx_JT = dldx_JT.view(self.bsz, -1)
        # CG requires it to have dims: n_samples x n_features x n_rh
        dldx_JT = dldx_JT.unsqueeze(2)  # unsqueeze for number of rhs.

        def v_JJT_matvec(v, u=xstar, Ru=Txstar):
            # inputs:
            # v = vector to be multiplied by JJT
            # u = fixed point vector u (requires grad)
            # Ru = R applied to u (requires grad)

            # assumes one rhs:
            # x (n_samples, n_dim, n_rhs) -> (n_samples, n_dim)

            v = v.squeeze(2)      # squeeze number of RHS
            v = v.view(Ru.shape)  # reshape to filter space
            v.requires_grad = True

            # compute v*J = v*(I - dRdu)
            v_dRdu = torch.autograd.grad(outputs=Ru, inputs=u,
                                          grad_outputs=v,
                                          retain_graph=True,
                                          create_graph=True,
                                          only_inputs=True)[0]
            v_J = v - v_dRdu

            # compute v_JJT
            v_JJT = torch.autograd.grad(outputs=v_J, inputs=v,
                                        grad_outputs=v_J,
                                        retain_graph=True,
                                        create_graph=True,
                                        only_inputs=True)[0]

            v = v.detach()
            v_J = v_J.detach()
            Amv = v_JJT.detach()
            Amv = Amv.view(Ru.shape[0], -1)
            Amv = Amv.unsqueeze(2).detach()
            return Amv

        normal_eq_sol, info = cg_batch(v_JJT_matvec, dldx_JT, M_bmm=None,
                                               X0=None, rtol=0, atol=1e-3,
                                               maxiter=50,
                                               verbose=False)

        if info['optimal']:
            xstar.requires_grad = False
            Txstar.backward(normal_eq_sol)
            loss = loss_function(Txstar, X)
            loss.backward()
            xstar.requires_grad = False
            optimizer.step()
    
    # this was written when comparing running time, 
    # which actually wraps the code
    # for recording purposes, ``train_jfb'' and ``test_jfb''
    # used the unwrapped version.
    def jacobian_free_update(self, X, loss_function, optimizer):
        d = self.A.forward(X) # actual input of the network
        with torch.no_grad():
            xstar, _ = self.find_fixed_point(d)
        xstar.requires_grad = True
        Txstar = self.g(d, xstar)
        loss = loss_function(Txstar, X)
        loss.backward()
        optimizer.step()
