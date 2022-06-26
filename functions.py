'''
Functions for training and testing different models
'''
def train_jfb(model, loader, operator, loss_function, optimizer, device):
    model.train()
    L = []
    n_iters_list = []
    grad_norm_list = []
    for batch_idx, X in enumerate(loader):
        print(batch_idx)
        if batch_idx > 2:
            break
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