import torch
import numpy as np
from collections import Counter

device = "cuda" if torch.cuda.is_available() else "cpu"

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.type(torch.FloatTensor).to(device), y.type(torch.LongTensor).to(device)

        # Compute prediction error
        pred = model(X)
        # print(type(pred))
        # print(pred.shape,y.shape)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    correct = (pred.argmax(1) == y).type(torch.float).sum().item()/len(X)
    return loss.item(), 100*correct
    # return loss, 100*correct




def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.type(torch.FloatTensor).to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, 100*correct
    # return 100*correct

def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


# 2*EEG+EOG
def train_3ch(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # print(size)
    model.train()
    # print(enumerate(dataloader).shape)
    for batch, (X_0,X_1,y) in enumerate(dataloader):
        # print(list(enumerate(dataloader)))
        # print(X_0.type(torch.FloatTensor).to(device).shape,y.type(torch.LongTensor).shape)
        X0, y = X_0.type(torch.FloatTensor).to(device), y.type(torch.LongTensor).to(device)
        X1 = X_1.type(torch.FloatTensor).to(device)
        # X2 = X_2.type(torch.FloatTensor).to(device)
        # X3 = X_3.type(torch.FloatTensor).to(device)
        # Compute prediction error
        pred = model(X0,X1)
        # print('pred.shape',pred.shape)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X_0)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    correct = (pred.argmax(1) == y).type(torch.float).sum().item()/size
    return loss.item(), 100*correct
    # return loss, 100*correct

def test_3ch(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X_0,X_1,y in dataloader:
            X0, y = X_0.type(torch.FloatTensor).to(device), y.type(torch.LongTensor).to(device)
            X1 = X_1.type(torch.FloatTensor).to(device)
            pred = model(X0,X1)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return  test_loss,100*correct


# 2*EEG+EOG+EMG
def train_4ch(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # print("train size:",size)
    model.train()
    # print(enumerate(dataloader).shape)
    for batch, (X_0,X_1,X_2,y) in enumerate(dataloader):
        # print(list(enumerate(dataloader)))
        # print(X_0.type(torch.FloatTensor).to(device).shape,y.type(torch.LongTensor).shape)
        X0, y = X_0.type(torch.FloatTensor).to(device), y.type(torch.LongTensor).to(device)
        X1 = X_1.type(torch.FloatTensor).to(device)
        X2 = X_2.type(torch.FloatTensor).to(device)
        # Compute prediction error
        pred = model(X0,X1,X2)
        # print('y.shape',y.shape)
        loss = loss_fn(pred, y)
        # print('pred.shape',pred.shape)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X_0)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    correct = (pred.argmax(1) == y).type(torch.float).sum().item()/len(X_0)
    return loss.item(), 100*correct
    # return loss, 100*correct

def test_4ch(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    # print("test size:",size)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X_0,X_1,X_2,y in dataloader:
            X0, y = X_0.type(torch.FloatTensor).to(device), y.type(torch.LongTensor).to(device)
            X1 = X_1.type(torch.FloatTensor).to(device)
            X2 = X_2.type(torch.FloatTensor).to(device)
            pred = model(X0,X1,X2)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return  test_loss,100*correct

def train_salient(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # print("Train size:",size)
    model.train()
    # print(len(dataloader))
    for batch, (X_0,X_1,y) in enumerate(dataloader):
        # print(list(enumerate(dataloader)))
        # print(X_0.type(torch.FloatTensor).to(device).shape,y.type(torch.LongTensor).shape)
        X0, y = X_0.type(torch.FloatTensor).to(device), y.type(torch.LongTensor).to(device)
        # print('X0.shape',X0.shape)
        X1 = X_1.type(torch.FloatTensor).to(device)
        
        y = y.reshape(-1)
        # print('y.shape',y.shape)
        # Compute prediction error
        pred = model(X0,X1)
        
        pred = pred.reshape(-1,5)
        # print('pred.shape',pred.shape)
        # pred /= torch.sum(pred,dim=-1).unsqueeze(dim = -1)
        # pred = torch.clamp(pred,0,1)
        # pred = model(X0)
        # print(pred.shape)#,pred)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X_0)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    correct = (pred.argmax(1) == y).type(torch.float).sum().item()/(len(X_0)*20)
    return loss.item(), 100*correct
    # return loss, 100*correct

def test_salient(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    print("size:",size)
    num_batches = len(dataloader)
    print("num_batches:",num_batches)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X_0,X_1,y in dataloader:
            X0, y = X_0.type(torch.FloatTensor).to(device), y.type(torch.LongTensor).to(device)
            X1 = X_1.type(torch.FloatTensor).to(device)
            y = y.reshape(-1)
            pred = model(X0,X1)
            # pred = model(X0)
            pred = pred.reshape(-1,5)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches*20
    correct /= size*20
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    # return  100*correct
    return test_loss, 100*correct

def train_1ch(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # print(size)
    model.train()
    # print(enumerate(dataloader).shape)
    for batch, (X_0,y) in enumerate(dataloader):
        # print(list(enumerate(dataloader)))
        # print('X_0.type(torch.FloatTensor).to(device).shape,y.type(torch.LongTensor).shape:',X_0.type(torch.FloatTensor).to(device).shape,y.type(torch.LongTensor).shape)
        X0, y = X_0.type(torch.FloatTensor).to(device), y.type(torch.LongTensor).to(device)
        # X1 = X_1.type(torch.FloatTensor).to(device)
        # X2 = X_2.type(torch.FloatTensor).to(device)
        # X3 = X_3.type(torch.FloatTensor).to(device)
        # Compute prediction error

        pred = model(X0)
        # print('pred.shape',pred.shape)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X_0)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    correct = (pred.argmax(1) == y).type(torch.float).sum().item()/size
    return loss.item(), 100*correct
    # return loss, 100*correct

def test_1ch(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X_0,y in dataloader:
            X0, y = X_0.type(torch.FloatTensor).to(device), y.type(torch.LongTensor).to(device)
            # X1 = X_1.type(torch.FloatTensor).to(device)
            pred = model(X0)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return  test_loss,100*correct

def train_printnet(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # print(size)
    model.train()
    # print(enumerate(dataloader).shape)
    for batch, (X_0,X_1,X_2,X_3,y) in enumerate(dataloader):
        # print(list(enumerate(dataloader)))
        # print(X_0.type(torch.FloatTensor).to(device).shape,y.type(torch.LongTensor).shape)
        X0, y = X_0.type(torch.FloatTensor).to(device), y.type(torch.LongTensor).to(device)
        X1 = X_1.type(torch.FloatTensor).to(device)
        X2 = X_2.type(torch.FloatTensor).to(device)
        X3 = X_3.type(torch.FloatTensor).to(device)
        # Compute prediction error
        pred = model(X0,X1,X2,X3)
        # print('pred.shape',pred.shape)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X_0)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    correct = (pred.argmax(1) == y).type(torch.float).sum().item()/len(X_0)
    return loss.item(), 100*correct
    # return loss, 100*correct

def test_printnet(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X_0,X_1,X_2,X_3,y in dataloader:
            X0, y = X_0.type(torch.FloatTensor).to(device), y.type(torch.LongTensor).to(device)
            X1 = X_1.type(torch.FloatTensor).to(device)
            X2 = X_2.type(torch.FloatTensor).to(device)
            X3 = X_3.type(torch.FloatTensor).to(device)
            pred = model(X0,X1,X2,X3)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, 100*correct
