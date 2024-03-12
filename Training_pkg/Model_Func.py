import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from Training_pkg.utils import *
from Training_pkg.Statistic_Func import linear_regression
from Training_pkg.ConvNet_Data_Func import Dataset,Dataset_Val
import torch.nn.functional as F



def train(model, X_train, y_train,X_test,y_test, BATCH_SIZE, learning_rate, TOTAL_EPOCHS):
    train_loader = DataLoader(Dataset(X_train, y_train), BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(Dataset(X_test, y_test), BATCH_SIZE, shuffle=True)
    print('*' * 25, type(train_loader), '*' * 25)
    losses = []
    valid_losses = []
    train_acc = []
    test_acc  = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = SelfDesigned_LossFunction()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    scheduler = lr_strategy_lookup_table(optimizer=optimizer)
    if ResNet_setting:
        for epoch in range(TOTAL_EPOCHS):
            correct = 0
            counts = 0
            for i, (images, labels) in enumerate(train_loader):
                model.train()
                images = images.to(device)
                labels = torch.squeeze(labels.type(torch.FloatTensor))
                labels = labels.to(device)
                optimizer.zero_grad()  # Set grads to zero
                outputs = model(images) #dimension: Nx1
                outputs = torch.squeeze(outputs)
                loss = criterion(outputs, labels)
                loss.backward()  ## backward
                optimizer.step()  ## refresh training parameters
                losses.append(loss.item())

                # Calculate R2
                y_hat = outputs.cpu().detach().numpy()
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
            valid_correct = 0
            valid_counts  = 0
            scheduler.step() 
            for i, (valid_images, valid_labels) in enumerate(validation_loader):
                model.eval()
                valid_images = valid_images.to(device)
                valid_labels = valid_labels.to(device)
                valid_output = model(valid_images)
                valid_output = torch.squeeze(valid_output)
                valid_loss   = criterion(valid_output, valid_labels)
                valid_losses.append(valid_loss.item())
                test_y_hat   = valid_output.cpu().detach().numpy()
                test_y_true  = valid_labels.cpu().detach().numpy()
                Valid_R2 = linear_regression(test_y_hat,test_y_true)
                Valid_R2 = np.round( Valid_R2, 4)
                valid_correct += Valid_R2
                valid_counts  += 1    
                print('Epoch : %d/%d, Iter : %d/%d,  Validate Loss: %.4f, Validate R2: %.4f' % (epoch + 1, TOTAL_EPOCHS,
                                                                    i + 1, len(X_train) // BATCH_SIZE,
                                                                    valid_loss.item(), Valid_R2)) 
            accuracy = correct / counts
            test_accuracy = valid_correct / valid_counts
            print('Epoch: ',epoch, ', Training Loss: ', loss.item(),', Training accuracy:',accuracy, ', \nTesting Loss:', valid_loss.item(),', Testing accuracy:', test_accuracy)

            train_acc.append(accuracy)
            test_acc.append(test_accuracy)
            print('Epoch: ',epoch,'\nLearning Rate:',optimizer.param_groups[0]['lr'])
            
       
            # Each epoch calculate test data accuracy
    if LateFusion_setting:
        initial_channel_index, latefusion_channel_index = find_latfusion_index()
        for epoch in range(TOTAL_EPOCHS):
            correct = 0
            counts = 0
            for i, (images, labels) in enumerate(train_loader):
                model.train()
                images = images.to(device)
                labels = torch.squeeze(labels.type(torch.FloatTensor))
                labels = labels.to(device)
                optimizer.zero_grad()  # Set grads to zero
                outputs = model(images[:,initial_channel_index,:,:], images[:,latefusion_channel_index,:,:]) #dimension: Nx1
                outputs = torch.squeeze(outputs)
                loss = criterion(outputs, labels)
                loss.backward()  ## backward
                optimizer.step()  ## refresh training parameters
                losses.append(loss.item())

                # Calculate R2
                y_hat = outputs.cpu().detach().numpy()
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
            valid_correct = 0
            valid_counts  = 0
            scheduler.step() 
            for i, (valid_images, valid_labels) in enumerate(validation_loader):
                model.eval()
                valid_images = valid_images.to(device)
                valid_labels = valid_labels.to(device)
                valid_output = model(valid_images[:,initial_channel_index,:,:], valid_images[:,latefusion_channel_index,:,:])
                valid_output = torch.squeeze(valid_output)
                valid_loss   = criterion(valid_output, valid_labels)
                valid_losses.append(valid_loss.item())
                test_y_hat   = valid_output.cpu().detach().numpy()
                test_y_true  = valid_labels.cpu().detach().numpy()
                Valid_R2 = linear_regression(test_y_hat,test_y_true)
                Valid_R2 = np.round( Valid_R2, 4)
                valid_correct += Valid_R2
                valid_counts  += 1    
                print('Epoch : %d/%d, Iter : %d/%d,  Validate Loss: %.4f, Validate R2: %.4f' % (epoch + 1, TOTAL_EPOCHS,
                                                                    i + 1, len(X_train) // BATCH_SIZE,
                                                                    valid_loss.item(), Valid_R2)) 
            accuracy = correct / counts
            test_accuracy = valid_correct / valid_counts
            print('Epoch: ',epoch, ', Training Loss: ', loss.item(),', Training accuracy:',accuracy, ', \nTesting Loss:', valid_loss.item(),', Testing accuracy:', test_accuracy)

            train_acc.append(accuracy)
            test_acc.append(test_accuracy)
            print('Epoch: ',epoch,'\nLearning Rate:',optimizer.param_groups[0]['lr'])
            
       
            # Each epoch calculate test data accuracy

    return losses,  train_acc, valid_losses, test_acc


def predict(inputarray, model, batchsize):
    #output = np.zeros((), dtype = float)
    model.eval()
    final_output = []
    final_output = np.array(final_output)
    predictinput = DataLoader(Dataset_Val(inputarray), batch_size= batchsize)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if ResNet_setting:
        with torch.no_grad():
            for i, image in enumerate(predictinput):
                image = image.to(device)
                output = model(image).cpu().detach().numpy()
                final_output = np.append(final_output,output)
    elif LateFusion_setting:
        initial_channel_index, latefusion_channel_index = find_latfusion_index()
        with torch.no_grad():
            for i, image in enumerate(predictinput):
                image = image.to(device)
                output = model(image[:,initial_channel_index,:,:],image[:,latefusion_channel_index,:,:]).cpu().detach().numpy()
                final_output = np.append(final_output,output)

    return final_output


class SelfDesigned_LossFunction(nn.Module):
    def __init__(self,size_average=None,reduce=None,reduction:str='mean')->None:
        super(SelfDesigned_LossFunction,self).__init__()
        self.Loss_Type = Loss_type
    def forward(self,model_output,target):
        if self.Loss_Type == 'MSE':
            loss = F.mse_loss(model_output, target)
            return loss

    
    