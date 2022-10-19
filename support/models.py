import torch


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
        layers.append(torch.nn.Conv2d(in_channels=self.nchannels, out_channels= self.nfeatures, kernel_size=self.ksz, padding=self.padding, bias=False))
        layers.append(torch.nn.ReLU(inplace=True))
        for _ in range(self.nlayers-2):
            layers.append(torch.nn.Conv2d(in_channels=self.nfeatures, out_channels=self.nfeatures, kernel_size=self.ksz, padding=self.padding, bias=False))
            layers.append(torch.nn.BatchNorm2d(self.nfeatures))
            layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.Conv2d(in_channels=self.nfeatures, out_channels=self.nchannels, kernel_size=self.ksz, padding=self.padding, bias=False))
        # Put them altogether and call it dncnn
        self.dncnn = torch.nn.Sequential(*layers)
    
    def forward(self, d):
        return self.dncnn(d)


class DEGRAD(torch.nn.Module):
    def __init__(self, c, batch_size, blur_operator, step_size, kernel_size, anderson=True):
        super().__init__()

        self.nchannels = c
        self.A = blur_operator
        self.eta = torch.nn.Parameter(step_size*torch.ones(()))
        self.max_num_iter = 50
        self.threshold = 1e-3 # this is specified in Gilton et al., 2021
        self.bsz = batch_size
        self.anderson = anderson

        # Trainable parameters
        # chen.spectral_norm may be replaced by torch.nn.utils.spectral_norm
        self.dncnn = DNCNN(c=self.nchannels, batch_size=self.bsz, kernel_size=kernel_size)
    
    def forward(self, d):
        with torch.no_grad():
            xstar, n_iters = self.find_fixed_point(d)
        Txstar = self.g(d, xstar)
        return Txstar, n_iters

    def find_fixed_point(self, d):
        with torch.no_grad():
            x0 = self.A.adjoint(d)

            # we could use anderson acceleration or not
            if self.anderson:
                x, num_iter = self.anderson(lambda X: self.g(x0, X), x0, max_iter=self.max_num_iter, tol=self.threshold)
                return x, num_iter
            else:
                temp = x0
                for i in range(self.max_num_iter):
                    x = self.g(d, temp)
                    if self._converge(x, temp):
                        return x, (i+1)
                    else:
                        temp = x
                return temp, self.max_num_iter
        
    
    def g(self, d, x):
        return (x - self.eta * (self.A.adjoint(torch.sub(self.A.forward(x), d)) + self.dncnn(x)))
    
    def anderson(self, f, x0, m=5, lam=1e-4, max_iter=50, tol=1e-2, beta = 1.0):
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

    def _converge(self, x1, x2):
        x1 = torch.reshape(x1, [self.bsz, -1])
        x2 = torch.flatten(x2, start_dim=1)
        n1 = torch.max(torch.norm(torch.sub(x1, x2), dim=1))
        n2 = torch.max(torch.norm(x2, dim=1))
        if  n1/n2 < self.threshold:
            return True
        else:
            return False
    
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
    
    def forward(self, d):
        with torch.no_grad():
            xstar, n_iters = self.find_fixed_point(d)
        Txstar = self.dncnn(xstar - self.eta * self.A.adjoint(torch.sub(self.A.forward(xstar), d)))
        return Txstar, n_iters

    def find_fixed_point(self, d):
        with torch.no_grad():
            x0 = self.A.adjoint(d)
            temp = x0
            for i in range(self.max_num_iter):
                x = self.dncnn(temp - self.eta * (self.A.adjoint(torch.sub(self.A.forward(temp), d))))
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

    def forward(self, d):
        with torch.no_grad():
            xstar, ustar = self.find_fixed_point(d)
        Tzstar = self.dncnn(torch.sub(xstar, ustar))
        Txstar = torch.matmul(self.invert, torch.mul(self.alpha, self.A.adjoint(d)) + Tzstar + ustar)
        return Txstar

    def find_fixed_point(self, d): #y is d
        with torch.no_grad():
            x0 = self.A.adjoint(d) # x0 is initialized as Ay
            z0 = torch.zeros(x0.shape) #z0 is initialized as all 0's
            u0 = 0

            tempx = x0
            tempz = z0
            tempu = u0
            for i in range(self.max_num_iter):
                z = self.dncnn(torch.sub(tempx, tempu)) # self.dncnn is R_theta, torch.sub means tempx-tempu
                x = torch.matmul(self.invert, torch.mul(self.alpha, self.A.adjoint(d)) + z + tempu) # use z, tempu
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
            basis = self.A.forward(basis) 
            A_mat[i, :] = torch.reshape(torch.squeeze(basis[0, :, :]), [1, -1])
            A_mat = torch.transpose(A_mat, 0, 1)

        return A_mat