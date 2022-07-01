'''
Functions for training and testing different models
'''
from matplotlib import pyplot as plt

def train_jfb(model, loader, operator, loss_function, optimizer, device):
    model.train()
    L = []
    n_iters_list = []
    grad_norm_list = []
    for batch_idx, X in enumerate(loader):
        if batch_idx % 10 == 0:
            print(f'current batch: {batch_idx}')
        X = X.to(device)
        d = operator.forward(X)
        pred, n_iters = model(d)
        batch_loss = loss_function(pred, X)
        batch_loss.backward()
        grad_norm_list.append(model.current_grad_norm())
        optimizer.step()
        L.append(batch_loss.item())
        n_iters_list.append(n_iters)
    return L, n_iters_list, grad_norm_list

def plotting(loss_list, n_iters_list, grad_norm_list, epoch_number):
    fig = plt.figure()
    fig.set_size_inches(18.5, 10)
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
    plt.savefig("./data/lliu58/Jacobian-free-Backprop-Implicit-Networks/degrad_output2/epoch"+str(epoch_number)+"results.png")
    # plt.savefig("./results/"+str(epoch_number)+"results.png")