'''
Helper functions
'''
import torch
from matplotlib import pyplot as plt
import time

def cg_batch(A_bmm, B, M_bmm=None, X0=None, rtol=1e-3, atol=0., maxiter=None, verbose=False):
    ###############################################################################
    # This method was retrieved from https://github.com/sbarratt/torch_cg , 
    # which is a PyTorch implementation of the preconditioned conjugate gradient method by
    # Shane Barrett. 
    ###############################################################################
    """
    Solves a batch of PD matrix linear systems using the preconditioned CG algorithm.

    This function solves a batch of matrix linear systems of the form

        A_i X_i = B_i,  i=1,...,K,

    where A_i is a n x n positive definite matrix and B_i is a n x m matrix,
    and X_i is the n x m matrix representing the solution for the ith system.

    Args:
        A_bmm: A callable that performs a batch matrix multiply of A and a K x n x m matrix.
        B: A K x n x m matrix representing the right hand sides.
        M_bmm: (optional) A callable that performs a batch matrix multiply of the preconditioning
            matrices M and a K x n x m matrix. (default=identity matrix)
        X0: (optional) Initial guess for X, defaults to M_bmm(B). (default=None)
        rtol: (optional) Relative tolerance for norm of residual. (default=1e-3)
        atol: (optional) Absolute tolerance for norm of residual. (default=0)
        maxiter: (optional) Maximum number of iterations to perform. (default=5*n)
        verbose: (optional) Whether or not to print status messages. (default=False)
    """
    K, n, m = B.shape

    if M_bmm is None:
        M_bmm = lambda x: x
    if X0 is None:
        X0 = M_bmm(B)
    if maxiter is None:
        maxiter = 5 * n

    assert B.shape == (K, n, m)
    assert X0.shape == (K, n, m)
    assert rtol > 0 or atol > 0
    assert isinstance(maxiter, int)

    X_k = X0
    R_k = B - A_bmm(X_k)
    Z_k = M_bmm(R_k)

    P_k = torch.zeros_like(Z_k)

    P_k1 = P_k
    R_k1 = R_k
    R_k2 = R_k
    X_k1 = X0
    Z_k1 = Z_k
    Z_k2 = Z_k

    B_norm = torch.norm(B, dim=1)
    stopping_matrix = torch.max(rtol*B_norm, atol*torch.ones_like(B_norm))

    if verbose:
        print("%03s | %010s %06s" % ("it", "dist", "it/s"))

    optimal = False
    start = time.perf_counter()
    for k in range(1, maxiter + 1):
        start_iter = time.perf_counter()
        Z_k = M_bmm(R_k)

        if k == 1:
            P_k = Z_k
            R_k1 = R_k
            X_k1 = X_k
            Z_k1 = Z_k
        else:
            R_k2 = R_k1
            Z_k2 = Z_k1
            P_k1 = P_k
            R_k1 = R_k
            Z_k1 = Z_k
            X_k1 = X_k
            denominator = (R_k2 * Z_k2).sum(1)
            denominator[denominator == 0] = 1e-8
            beta = (R_k1 * Z_k1).sum(1) / denominator
            P_k = Z_k1 + beta.unsqueeze(1) * P_k1

        denominator = (P_k * A_bmm(P_k)).sum(1)
        denominator[denominator == 0] = 1e-8
        alpha = (R_k1 * Z_k1).sum(1) / denominator
        X_k = X_k1 + alpha.unsqueeze(1) * P_k
        R_k = R_k1 - alpha.unsqueeze(1) * A_bmm(P_k)
        end_iter = time.perf_counter()

        residual_norm = torch.norm(A_bmm(X_k) - B, dim=1)

        if verbose:
            print("%03d | %8.4e %4.2f" %
                  (k, torch.max(residual_norm-stopping_matrix),
                    1. / (end_iter - start_iter)))

        if (residual_norm <= stopping_matrix).all():
            optimal = True
            break

    end = time.perf_counter()

    if verbose:
        if optimal:
            print("Terminated in %d steps (reached maxiter). Took %.3f ms." %
                  (k, (end - start) * 1000))
        else:
            print("Terminated in %d steps (optimal). Took %.3f ms." %
                  (k, (end - start) * 1000))


    info = {
        "niter": k,
        "optimal": optimal
    }

    return X_k, info

def train_jfb(model, loader, operator, loss_function, optimizer, device, ssim_calculator):
    model.train()
    model.dncnn.train()
    L = []
    n_iters_list = []
    grad_norm_list = []
    ssim_list = []
    for batch_idx, X in enumerate(loader):
        optimizer.zero_grad()
        X = X.to(device)
        d = operator.forward(X)
        pred, n_iters = model(d)
        batch_loss = torch.div(loss_function(pred, X), model.bsz)
        batch_loss.backward()
        grad_norm_list.append(model.current_grad_norm())
        optimizer.step()
        L.append(batch_loss.item())
        n_iters_list.append(n_iters)
        with torch.no_grad():
            batch_ssim = ssim_calculator(pred, X)
        ssim_list.append(batch_ssim.item())
    return L, n_iters_list, grad_norm_list, ssim_list

def valid_jfb(model, loader, operator, loss_function, device, ssim_calculator):
    model.eval()
    model.dncnn.eval()
    acc = 0
    ssim = 0
    for batch_idx, X in enumerate(loader):
        X = X.to(device)
        d = operator.forward(X)
        pred, _ = model(d)
        batch_loss = loss_function(pred, X)
        batch_ssim = ssim_calculator(pred, X)
        acc += batch_loss.item()
        ssim += batch_ssim.item()
    return acc/(batch_idx + 1), ssim/(batch_idx + 1)


def plotting(loss_list, n_iters_list, grad_norm_list, epoch_number, path):
    fig = plt.figure()
    fig.add_subplot(1, 3, 1)
    plt.plot(loss_list)
    plt.xlabel("# epochs")
    plt.ylabel("avg loss of epoch")
    fig.add_subplot(1, 3, 2)
    plt.plot(n_iters_list)
    plt.xlabel("# epochs")
    plt.ylabel("# iterations applied")
    fig.add_subplot(1, 3, 3)
    plt.plot(grad_norm_list)
    plt.xlabel("#epochs")
    plt.ylabel("avg gradient norm")
    plt.savefig(path+"results"+str(epoch_number)+".png")
