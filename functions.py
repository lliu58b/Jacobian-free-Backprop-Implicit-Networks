'''
Functions for training and testing different models
'''
def train_jfb(model, loader, operator, loss_function, optimizer, device):
    model.train()
    L = []
    for batch_idx, X in enumerate(loader):
        X = X.to(device)
        # if batch_idx % 10 == 0:
        #     print(f'now batch {batch_idx} is finished')
        d = operator.forward(X)
        pred = model(d)
        batch_loss = loss_function(pred, X)
        batch_loss.backward()
        optimizer.step()
        L.append(batch_loss.item())
    return L