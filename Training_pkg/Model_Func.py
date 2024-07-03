import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from Training_pkg.utils import *
from Training_pkg.Statistic_Func import linear_regression
from Training_pkg.ConvNet_Data_Func import Dataset,Dataset_Val
from Training_pkg.Loss_Func import SelfDesigned_LossFunction
import torch.nn.functional as F



def train(model, X_train, y_train,X_test,y_test,input_mean, input_std,width,height, BATCH_SIZE, learning_rate, TOTAL_EPOCHS,initial_channel_names,main_stream_channels,side_stream_channels):
    train_loader = DataLoader(Dataset(X_train, y_train), BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(Dataset(X_test, y_test), 2000, shuffle=True)
    print('*' * 25, type(train_loader), '*' * 25)
    losses = []
    valid_losses = []
    train_acc = []
    test_acc  = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = SelfDesigned_LossFunction(losstype=Loss_type)
    #optimizer = torch.optim.Adam(params=model.parameters(),betas=(), lr=learning_rate)
    optimizer = optimizer_lookup(model_parameters=model.parameters(),learning_rate=learning_rate)
    scheduler = lr_strategy_lookup_table(optimizer=optimizer)

    if species == 'NO3':
        GeoSpecies_index = initial_channel_names.index('Geo{}'.format('NIT'))
    else:
        GeoSpecies_index = initial_channel_names.index('Geo{}'.format(species))
    
    if ResNet_setting or ResNet_MLP_setting:
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
                loss = criterion(outputs, labels, images[:,GeoSpecies_index,int((width-1)/2),int((height-1)/2)],input_mean[GeoSpecies_index,int((width-1)/2),int((height-1)/2)],input_std[GeoSpecies_index,int((width-1)/2),int((height-1)/2)])
                loss.backward()  ## backward
                optimizer.step()  ## refresh training parameters
                losses.append(loss.item())

                # Calculate R2
                y_hat = outputs.cpu().detach().numpy()
                y_true = labels.cpu().detach().numpy()

               
                #torch.cuda.empty_cache()
                print('Epoch: ', epoch, ' i th: ', i, 'y_hat size: ',y_hat.shape)
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
                print('valid_images size: {}'.format(valid_images.shape),'valid_labels size: {}'.format(valid_labels.shape))
                valid_output = model(valid_images)
                valid_output = torch.squeeze(valid_output)
                valid_loss   = criterion(valid_output, valid_labels, valid_images[:,GeoSpecies_index,int((width-1)/2),int((height-1)/2)],input_mean[GeoSpecies_index,int((width-1)/2),int((height-1)/2)],input_std[GeoSpecies_index,int((width-1)/2),int((height-1)/2)])
                valid_losses.append(valid_loss.item())
                test_y_hat   = valid_output.cpu().detach().numpy()
                test_y_true  = valid_labels.cpu().detach().numpy()
                #print('test_y_hat size: {}'.format(test_y_hat.shape),'test_y_true size: {}'.format(test_y_true.shape))
                Valid_R2 = linear_regression(test_y_hat,test_y_true)
                Valid_R2 = np.round(Valid_R2, 4)
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

    elif ResNet_Classification_Settings:
        criterion = SelfDesigned_LossFunction(losstype=Classification_loss_type)
        bins = torch.tensor(np.linspace(ResNet_Classification_left_bin,ResNet_Classification_right_bin,ResNet_Classification_bins_number)).float()
        bins = bins.to(device)
        for epoch in range(TOTAL_EPOCHS):
            correct = 0
            counts = 0
            for i, (images, labels) in enumerate(train_loader):
                labels[np.where(labels > ResNet_Classification_right_bin)] = ResNet_Classification_right_bin
                labels[np.where(labels < ResNet_Classification_left_bin)]  = ResNet_Classification_left_bin
                model.train()
                images = images.to(device)
                labels = torch.squeeze(labels.type(torch.FloatTensor))
                labels = labels.to(device)
                optimizer.zero_grad()  # Set grads to zero
                classification_output = model(images) #dimension: Nx1
                classification_output = torch.squeeze(classification_output)
                classification_labels = torch.tensor((labels-ResNet_Classification_left_bin)/abs((ResNet_Classification_right_bin-ResNet_Classification_left_bin)/(ResNet_Classification_bins_number-1)),dtype=torch.long)
                classification_labels.to(device)
                loss = criterion(classification_output, classification_labels, images[:,GeoSpecies_index,int((width-1)/2),int((height-1)/2)],input_mean[GeoSpecies_index,int((width-1)/2),int((height-1)/2)],input_std[GeoSpecies_index,int((width-1)/2),int((height-1)/2)])
                loss.backward()  ## backward
                optimizer.step()  ## refresh training parameters
                losses.append(loss.item())

                # Calculate R2
                outputs = torch.matmul(classification_output,bins)
                y_hat = outputs.cpu().detach().numpy()
                y_true = labels.cpu().detach().numpy()

               
                #torch.cuda.empty_cache()
                print('Epoch: ', epoch, ' i th: ', i, 'y_hat size: ',y_hat.shape)
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
                valid_labels[np.where(valid_labels > ResNet_Classification_right_bin)] = ResNet_Classification_right_bin
                valid_labels[np.where(valid_labels < ResNet_Classification_left_bin)]  = ResNet_Classification_left_bin
                model.eval()
                valid_images = valid_images.to(device)
                valid_labels = valid_labels.to(device)
                print('valid_images size: {}'.format(valid_images.shape),'valid_labels size: {}'.format(valid_labels.shape))
                classification_valid_output = model(valid_images)
                classification_valid_output = torch.squeeze(classification_valid_output)
                classification_valid_labels = torch.tensor((valid_labels-ResNet_Classification_left_bin)/abs((ResNet_Classification_right_bin-ResNet_Classification_left_bin)/(ResNet_Classification_bins_number-1)),dtype=torch.long)
                
                valid_loss   = criterion(classification_valid_output, classification_valid_labels, valid_images[:,GeoSpecies_index,int((width-1)/2),int((height-1)/2)],input_mean[GeoSpecies_index,int((width-1)/2),int((height-1)/2)],input_std[GeoSpecies_index,int((width-1)/2),int((height-1)/2)])
                valid_losses.append(valid_loss.item())
                valid_output = torch.matmul(classification_valid_output,bins)
                test_y_hat   = valid_output.cpu().detach().numpy()
                test_y_true  = valid_labels.cpu().detach().numpy()
                #print('test_y_hat size: {}'.format(test_y_hat.shape),'test_y_true size: {}'.format(test_y_true.shape))
                Valid_R2 = linear_regression(test_y_hat,test_y_true)
                Valid_R2 = np.round(Valid_R2, 4)
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
    elif ResNet_MultiHeadNet_Settings:
        Classfication_criterion = SelfDesigned_LossFunction(losstype=Classification_loss_type)
        bins = torch.tensor(np.linspace(ResNet_MultiHeadNet_left_bin,ResNet_MultiHeadNet_right_bin,ResNet_MultiHeadNet_bins_number)).float()
        bins = bins.to(device)
        for epoch in range(TOTAL_EPOCHS):
            correct = 0
            counts = 0
            for i, (images, labels) in enumerate(train_loader):
                labels[np.where(labels > ResNet_MultiHeadNet_right_bin)] = ResNet_MultiHeadNet_right_bin
                labels[np.where(labels < ResNet_MultiHeadNet_left_bin)]  = ResNet_MultiHeadNet_left_bin
                model.train()
                images = images.to(device)
                labels = torch.squeeze(labels.type(torch.FloatTensor))
                labels = labels.to(device)
                optimizer.zero_grad()  # Set grads to zero
                regression_output, classification_output = model(images) #dimension: Nx1
                regression_output = torch.squeeze(regression_output)
                classification_output = torch.squeeze(classification_output)

                classification_labels = torch.tensor((labels-ResNet_MultiHeadNet_left_bin)/abs((ResNet_MultiHeadNet_right_bin-ResNet_MultiHeadNet_left_bin)/(ResNet_MultiHeadNet_bins_number-1)),dtype=torch.long)
                classification_labels.to(device)
                regression_loss = criterion(regression_output,labels,images[:,GeoSpecies_index,int((width-1)/2),int((height-1)/2)],input_mean[GeoSpecies_index,int((width-1)/2),int((height-1)/2)],input_std[GeoSpecies_index,int((width-1)/2),int((height-1)/2)])
                classfication_loss = Classfication_criterion(classification_output, classification_labels, images[:,GeoSpecies_index,int((width-1)/2),int((height-1)/2)],input_mean[GeoSpecies_index,int((width-1)/2),int((height-1)/2)],input_std[GeoSpecies_index,int((width-1)/2),int((height-1)/2)])
                loss =  ResNet_MultiHeadNet_regression_loss_coefficient * regression_loss + ResNet_MultiHeadNet_classfication_loss_coefficient * classfication_loss
                loss.backward()  ## backward
                optimizer.step()  ## refresh training parameters
                losses.append(loss.item())

                # Calculate R2
                outputs = ResNet_MultiHeadNet_regression_portion * regression_output + ResNet_MultiHeadNet_classifcation_portion*torch.matmul(classification_output,bins)
                y_hat = outputs.cpu().detach().numpy()
                y_true = labels.cpu().detach().numpy()

               
                #torch.cuda.empty_cache()
                print('Epoch: ', epoch, ' i th: ', i, 'y_hat size: ',y_hat.shape)
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
                valid_labels[np.where(valid_labels > ResNet_MultiHeadNet_right_bin)] = ResNet_MultiHeadNet_right_bin
                valid_labels[np.where(valid_labels < ResNet_MultiHeadNet_left_bin)]  = ResNet_MultiHeadNet_left_bin
                model.eval()
                valid_images = valid_images.to(device)
                valid_labels = valid_labels.to(device)
                print('valid_images size: {}'.format(valid_images.shape),'valid_labels size: {}'.format(valid_labels.shape))
                regresssion_valid_output,classification_valid_output = model(valid_images)
                classification_valid_output = torch.squeeze(classification_valid_output)
                classification_valid_labels = torch.tensor((valid_labels-ResNet_MultiHeadNet_left_bin)/abs((ResNet_MultiHeadNet_left_bin-ResNet_MultiHeadNet_right_bin)/(ResNet_MultiHeadNet_bins_number-1)),dtype=torch.long)
                
                valid_classifcation_loss   = Classfication_criterion(classification_valid_output, classification_valid_labels, valid_images[:,GeoSpecies_index,int((width-1)/2),int((height-1)/2)],input_mean[GeoSpecies_index,int((width-1)/2),int((height-1)/2)],input_std[GeoSpecies_index,int((width-1)/2),int((height-1)/2)])
                valid_regression_loss      = criterion(regresssion_valid_output, valid_labels, valid_images[:,GeoSpecies_index,int((width-1)/2),int((height-1)/2)],input_mean[GeoSpecies_index,int((width-1)/2),int((height-1)/2)],input_std[GeoSpecies_index,int((width-1)/2),int((height-1)/2)])
                valid_loss = ResNet_MultiHeadNet_regression_loss_coefficient * valid_regression_loss + ResNet_MultiHeadNet_classfication_loss_coefficient * valid_classifcation_loss
                valid_losses.append(valid_loss.item())
                valid_output = ResNet_MultiHeadNet_regression_portion * regresssion_valid_output + ResNet_MultiHeadNet_classifcation_portion * torch.matmul(classification_valid_output,bins)

                test_y_hat   = valid_output.cpu().detach().numpy()
                test_y_true  = valid_labels.cpu().detach().numpy()
                #print('test_y_hat size: {}'.format(test_y_hat.shape),'test_y_true size: {}'.format(test_y_true.shape))
                Valid_R2 = linear_regression(test_y_hat,test_y_true)
                Valid_R2 = np.round(Valid_R2, 4)
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

    elif LateFusion_setting:
        initial_channel_index, latefusion_channel_index = find_latfusion_index(total_channel_names=initial_channel_names,initial_channels=main_stream_channels,late_fusion_channels=side_stream_channels)
        
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
                loss = criterion(outputs, labels, images[:,GeoSpecies_index,int((width-1)/2),int((height-1)/2)],input_mean[GeoSpecies_index,int((width-1)/2),int((height-1)/2)],input_std[GeoSpecies_index,int((width-1)/2),int((height-1)/2)])
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
                valid_loss   = criterion(valid_output, valid_labels, valid_images[:,GeoSpecies_index,int((width-1)/2),int((height-1)/2)],input_mean[GeoSpecies_index,int((width-1)/2),int((height-1)/2)],input_std[GeoSpecies_index,int((width-1)/2),int((height-1)/2)])
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
    elif MultiHeadLateFusion_settings:
        initial_channel_index, latefusion_channel_index = find_latfusion_index(total_channel_names=initial_channel_names,initial_channels=main_stream_channels,late_fusion_channels=side_stream_channels)
        criterion_MH = SelfDesigned_LossFunction(losstype=Classification_loss_type)
        for epoch in range(TOTAL_EPOCHS):
            correct = 0
            counts = 0
            for i, (images, labels) in enumerate(train_loader):
                labels[np.where(labels > MultiHeadLateFusion_right_bin)] = MultiHeadLateFusion_right_bin
                labels[np.where(labels < MultiHeadLateFusion_left_bin)]  = MultiHeadLateFusion_left_bin
                model.train()
                images = images.to(device)
                labels = torch.squeeze(labels.type(torch.FloatTensor))
                labels = labels.to(device)
                optimizer.zero_grad()  # Set grads to zero
                
                regression_output, classification_output = model(images[:,initial_channel_index,:,:], images[:,latefusion_channel_index,:,:]) #dimension: Nx1
                regression_output = torch.squeeze(regression_output)
                classification_output = torch.squeeze(classification_output)

                loss = criterion(regression_output, labels, images[:,GeoSpecies_index,int((width-1)/2),int((height-1)/2)],input_mean[:,GeoSpecies_index,int((width-1)/2),int((height-1)/2)],input_std[:,GeoSpecies_index,int((width-1)/2),int((height-1)/2)])
                loss.backward()  ## backward
                
                classification_labels = torch.tensor((labels-MultiHeadLateFusion_left_bin)/abs((MultiHeadLateFusion_right_bin-MultiHeadLateFusion_left_bin)/(MultiHeadLateFusion_bins_number-1)),dtype=torch.long)
                classification_labels.to(device)
                loss_MH = criterion_MH(classification_output, classification_labels, images[:,GeoSpecies_index,int((width-1)/2),int((height-1)/2)],input_mean[GeoSpecies_index,int((width-1)/2),int((height-1)/2)],input_std[GeoSpecies_index,int((width-1)/2),int((height-1)/2)])
                loss_MH.backward() #retain_graph=True
                
                optimizer.step()  ## refresh training parameters
                losses.append(loss.item())

                # Calculate R2
                bins = torch.tensor(np.linspace(MultiHeadLateFusion_left_bin,MultiHeadLateFusion_right_bin,MultiHeadLateFusion_bins_number)).float()
                bins = bins.to(device)

                outputs = MultiHeadLateFusion_regression_portion*regression_output + MultiHeadLateFusion_classifcation_portion*torch.matmul(classification_output,bins)
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
                valid_labels[np.where(valid_labels > MultiHeadLateFusion_right_bin)] = MultiHeadLateFusion_right_bin
                valid_labels[np.where(valid_labels < MultiHeadLateFusion_left_bin)]  = MultiHeadLateFusion_left_bin
                model.eval()
                valid_images = valid_images.to(device)
                valid_labels = valid_labels.to(device)
                valid_regression_output, valid_classification_output = model(valid_images[:,initial_channel_index,:,:], valid_images[:,latefusion_channel_index,:,:])
                
                valid_regression_output = torch.squeeze(valid_regression_output)
                valid_classification_output = torch.squeeze(valid_classification_output)
                valid_loss   = criterion(valid_regression_output, valid_labels, valid_images[:,GeoSpecies_index,int((width-1)/2),int((height-1)/2)],input_mean[GeoSpecies_index,int((width-1)/2),int((height-1)/2)],input_std[GeoSpecies_index,int((width-1)/2),int((height-1)/2)])
                valid_losses.append(valid_loss.item())
                bins = torch.tensor(np.linspace(MultiHeadLateFusion_left_bin,MultiHeadLateFusion_right_bin,MultiHeadLateFusion_bins_number)).float()
                bins = bins.to(device)
                valid_output = MultiHeadLateFusion_regression_portion*valid_regression_output + MultiHeadLateFusion_classifcation_portion*torch.matmul(valid_classification_output,bins)

                test_y_hat   = valid_output.cpu().detach().numpy()
                test_y_true  = valid_labels.cpu().detach().numpy()
                Valid_R2 = linear_regression(test_y_hat,test_y_true)
                Valid_R2 = np.round(Valid_R2, 4)
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


def predict(inputarray, model, batchsize,initial_channel_names,mainstream_channel_names,sidestream_channel_names):
    #output = np.zeros((), dtype = float)
    model.eval()
    final_output = []
    final_output = np.array(final_output)
    predictinput = DataLoader(Dataset_Val(inputarray), batch_size= batchsize)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if ResNet_setting or ResNet_MLP_setting:
        with torch.no_grad():
            for i, image in enumerate(predictinput):
                image = image.to(device)
                output = model(image).cpu().detach().numpy()
                final_output = np.append(final_output,output)
    elif ResNet_Classification_Settings:
        with torch.no_grad():
            bins = torch.tensor(np.linspace(ResNet_Classification_left_bin,ResNet_Classification_right_bin,ResNet_Classification_bins_number)).float()
            bins = bins.to(device)
            for i, image in enumerate(predictinput):
                image = image.to(device)
                classification_output = model(image)
                classification_output = torch.squeeze(classification_output)
                outputs = torch.matmul(classification_output,bins)
                outputs = outputs.cpu().detach().numpy()
                final_output = np.append(final_output,outputs)
    elif ResNet_MultiHeadNet_Settings:
        with torch.no_grad():
            bins = torch.tensor(np.linspace(ResNet_MultiHeadNet_left_bin,ResNet_MultiHeadNet_right_bin,ResNet_MultiHeadNet_bins_number)).float()
            bins = bins.to(device)
            for i, image in enumerate(predictinput):
                image = image.to(device)
                regression_output, classification_output = model(image).cpu().detach().numpy()
                regression_output = torch.squeeze(regression_output)
                classification_output = torch.squeeze(classification_output)
                outputs = ResNet_MultiHeadNet_regression_portion*regression_output + ResNet_MultiHeadNet_classifcation_portion*torch.matmul(classification_output,bins)
                outputs = outputs.cpu().detach().numpy()
                final_output = np.append(final_output,outputs)
    elif LateFusion_setting:
        initial_channel_index, latefusion_channel_index = find_latfusion_index(total_channel_names=initial_channel_names,initial_channels=mainstream_channel_names,late_fusion_channels=sidestream_channel_names)
        with torch.no_grad():
            for i, image in enumerate(predictinput):
                image = image.to(device)
                output = model(image[:,initial_channel_index,:,:],image[:,latefusion_channel_index,:,:]).cpu().detach().numpy()
                final_output = np.append(final_output,output)
    elif MultiHeadLateFusion_settings:
        initial_channel_index, latefusion_channel_index = find_latfusion_index(total_channel_names=initial_channel_names,initial_channels=mainstream_channel_names,late_fusion_channels=sidestream_channel_names)
        for i, image in enumerate(predictinput):
            image = image.to(device)
            regression_output, classification_output = model(image[:,initial_channel_index,:,:],image[:,latefusion_channel_index,:,:])
            regression_output = torch.squeeze(regression_output)
            classification_output = torch.squeeze(classification_output)
            bins = torch.tensor(np.linspace(MultiHeadLateFusion_left_bin,MultiHeadLateFusion_right_bin,MultiHeadLateFusion_bins_number)).float()
            bins = bins.to(device)
            outputs = MultiHeadLateFusion_regression_portion*regression_output + MultiHeadLateFusion_classifcation_portion*torch.matmul(classification_output,bins)
            print('regression_output shape: ',regression_output.shape, '\nclassification_output shape:', classification_output.shape,
                  '\nbins shape: ', bins.shape, '\noutputs shape: ', outputs.shape)
            outputs = outputs.cpu().detach().numpy()
            final_output = np.append(final_output,outputs)
    return final_output




    
    