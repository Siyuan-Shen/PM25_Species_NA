import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from Training_pkg.Statistic_Func import linear_regression
from Training_pkg.ConvNet_Data_Func import Dataset,Dataset_Val
import torch.nn.functional as F



def train(model, X_train, y_train, BATCH_SIZE, learning_rate, TOTAL_EPOCHS):
    train_loader = DataLoader(Dataset(X_train, y_train), BATCH_SIZE, shuffle=True)
    print('*' * 25, type(train_loader), '*' * 25)
    criterion = nn.MSELoss()
    losses = []
    train_acc = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    ## scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=0.5,patience=3,threshold=0.005)
    model.train()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5,gamma=0.5)
    for epoch in range(TOTAL_EPOCHS):
        correct = 0
        counts = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = torch.squeeze(labels.type(torch.FloatTensor))
            labels = labels.to(device)
            optimizer.zero_grad()  # Set grads to zero
            outputs = model(images) #dimension: Nx1
            outputs = torch.squeeze(outputs)
            loss = criterion(outputs, labels)#, images[:,16,5,5],GeoPM25_mean,GeoPM25_std)#,images[:,-1,5,5],SitesNumber_mean,SitesNumber_std)
            loss.backward()  ## backward
            optimizer.step()  ## refresh training parameters
            losses.append(loss.item())

            # Calculate R2
            y_hat = model(images).cpu().detach().numpy()
            y_true = labels.cpu().detach().numpy()
            #torch.cuda.empty_cache()
            print('Epoch: ', epoch, ' i th: ', i)
            #print('y_hat:', y_hat)
            R2 = linear_regression(y_hat,y_true)
            R2 = np.round(R2, 4)
            #pred = y_hat.max(1, keepdim=True)[1] # 得到最大值及索引，a.max[0]为最大值，a.max[1]为最大值的索引
            correct += R2
            counts  += 1
            if (i + 1) % 10 == 0:
                # 每10个batches打印一次loss
                print('Epoch : %d/%d, Iter : %d/%d,  Loss: %.4f' % (epoch + 1, TOTAL_EPOCHS,
                                                                    i + 1, len(X_train) // BATCH_SIZE,
                                                                    loss.item()))
        accuracy = correct / counts
        print('Epoch: ',epoch, ', Loss: ', loss.item(),', Training set accuracy:',accuracy)
        train_acc.append(accuracy)
        print('Epoch: ',epoch,'\nLearning Rate:',optimizer.param_groups[0]['lr'])
        scheduler.step()
        # Each epoch calculate test data accuracy
    return losses,  train_acc


def predict(inputarray, model, batchsize):
    #output = np.zeros((), dtype = float)
    model.eval()
    final_output = []
    final_output = np.array(final_output)
    predictinput = DataLoader(Dataset_Val(inputarray), batch_size= batchsize)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for i, image in enumerate(predictinput):
            image = image.to(device)
            output = model(image).cpu().detach().numpy()
            final_output = np.append(final_output,output)

    return final_output