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
    
    

    if TwoCombineModels_Settings:
        TwoCombineModel_Variable_index = initial_channel_names.index(TwoCombineModels_Variable)
        criterion_LowEnd = SelfDesigned_LossFunction(losstype=Loss_type)
        criterion_FarEnd = SelfDesigned_LossFunction(losstype=Loss_type)
        
        optimizer_LowEnd = optimizer_lookup(model_parameters=model.model_A.parameters(),learning_rate=learning_rate)
        optimizer_FarEnd = optimizer_lookup(model_parameters=model.model_B.parameters(),learning_rate=learning_rate)
        scheduler_LowEnd = lr_strategy_lookup_table(optimizer=optimizer_LowEnd)
        scheduler_FarEnd = lr_strategy_lookup_table(optimizer=optimizer_FarEnd)
        GeoSpecies_train = X_train[:,TwoCombineModel_Variable_index,int((width-1)/2),int((height-1)/2)]*input_std[TwoCombineModel_Variable_index,int((width-1)/2),int((height-1)/2)] + input_mean[GeoSpecies_index,int((width-1)/2),int((height-1)/2)]
        GeoSpecies_train_LowEnd_index = np.where(GeoSpecies_train <= TwoCombineModels_threshold)
        GeoSpecies_train_FarEnd_index = np.where(GeoSpecies_train > TwoCombineModels_threshold)
        X_train_LowEnd = X_train[GeoSpecies_train_LowEnd_index, :, :, :]
        X_train_FarEnd = X_train[:,:,:,:]#X_train[GeoSpecies_train_FarEnd_index, :, :, :] #
        y_train_LowEnd = y_train[GeoSpecies_train_LowEnd_index]
        y_train_FarEnd = y_train[:]#y_train[GeoSpecies_train_FarEnd_index]#
        train_loader_LowEnd = DataLoader(Dataset(X_train_LowEnd, y_train_LowEnd), BATCH_SIZE, shuffle=True)
        train_loader_FarEnd = DataLoader(Dataset(X_train_FarEnd, y_train_FarEnd), BATCH_SIZE, shuffle=True)

        GeoSpecies_valid = X_test[:,TwoCombineModel_Variable_index,int((width-1)/2),int((height-1)/2)]*input_std[TwoCombineModel_Variable_index,int((width-1)/2),int((height-1)/2)] + input_mean[GeoSpecies_index,int((width-1)/2),int((height-1)/2)]
        GeoSpecies_valid_LowEnd_index = np.where(GeoSpecies_valid <= TwoCombineModels_threshold)
        GeoSpecies_valid_FarEnd_index = np.where(GeoSpecies_valid > TwoCombineModels_threshold)
        X_valid_LowEnd = X_test[GeoSpecies_valid_LowEnd_index, :, :, :]
        X_valid_FarEnd = X_test[:,:,:,:]#X_test[GeoSpecies_valid_FarEnd_index, :, :, :]#
        y_valid_LowEnd = y_test[GeoSpecies_valid_LowEnd_index]
        y_valid_FarEnd = y_test[:]#y_test[GeoSpecies_valid_FarEnd_index]#
        validation_loader_LowEnd = DataLoader(Dataset(X_valid_LowEnd, y_valid_LowEnd), 2000, shuffle=True)
        validation_loader_FarEnd = DataLoader(Dataset(X_valid_FarEnd, y_valid_FarEnd), 2000, shuffle=True)

        train_acc_LowEnd = []
        train_acc_FarEnd = []
        test_acc_LowEnd  = []
        test_acc_FarEnd  = []
        for epoch in range(TOTAL_EPOCHS):
            correct_LowEnd = 0
            counts_LowEnd = 0
            correct_FarEnd = 0
            counts_FarEnd = 0
            for i, (images_LowEnd, labels_LowEnd) in enumerate(train_loader_LowEnd):
                model.train()
                print('images_LowEnd shape: {}, labels_LowEnd shape: {}'.format(images_LowEnd.shape,labels_LowEnd.shape))
                images_LowEnd = images_LowEnd.to(device)
                labels_LowEnd = labels_LowEnd.to(device)
                #images_LowEnd = torch.squeeze(images_LowEnd)
                #labels_LowEnd = torch.squeeze(labels_LowEnd)
                
                optimizer_LowEnd.zero_grad()  # Set grads to zero
                outputs_LowEnd = model.model_A(images_LowEnd) #dimension: Nx1
                outputs_LowEnd = torch.squeeze(outputs_LowEnd)

                loss_LowEnd = criterion_LowEnd(outputs_LowEnd, labels_LowEnd, images_LowEnd[:,GeoSpecies_index,int((width-1)/2),int((height-1)/2)],input_mean[GeoSpecies_index,int((width-1)/2),int((height-1)/2)],input_std[GeoSpecies_index,int((width-1)/2),int((height-1)/2)])
                loss_LowEnd.backward()   ## backward
                optimizer_LowEnd.step()  ## refresh training parameters
                losses.append(loss_LowEnd.item())

                # Calculate R2
                y_hat_LowEnd = outputs_LowEnd.cpu().detach().numpy()
                y_true_LowEnd = labels_LowEnd.cpu().detach().numpy()
                y_hat_LowEnd = np.array(y_hat_LowEnd,ndmin=1)
                #torch.cuda.empty_cache()
                print('Epoch: ', epoch, ' i th: ', i, 'y_hat size: ',y_hat_LowEnd.shape)
                #print('y_hat:', y_hat)
                R2 = linear_regression(y_hat_LowEnd,y_true_LowEnd)
                R2 = np.round(R2, 4)
                #pred = y_hat.max(1, keepdim=True)[1] # 得到最大值及索引，a.max[0]为最大值，a.max[1]为最大值的索引
                correct_LowEnd += R2
                counts_LowEnd  += 1
                if (i + 1) % 10 == 0:
                # 每10个batches打印一次loss
                    print('Low End Model Epoch : %d/%d, Iter : %d/%d,  Loss: %.4f' % (epoch + 1, TOTAL_EPOCHS,
                                                                    i + 1, len(X_train) // BATCH_SIZE,
                                                                    loss_LowEnd.item()))
            scheduler_LowEnd.step()
            for i, (images_FarEnd, labels_FarEnd) in enumerate(train_loader_FarEnd):
                model.train()
                
                images_FarEnd = images_FarEnd.to(device)
                labels_FarEnd = labels_FarEnd.to(device)
                optimizer_FarEnd.zero_grad()  # Set grads to zero
                outputs_FarEnd = model.model_B(images_FarEnd) #dimension: Nx1
                outputs_FarEnd = torch.squeeze(outputs_FarEnd)
                loss_FarEnd = criterion_FarEnd(outputs_FarEnd, labels_FarEnd, images_FarEnd[:,GeoSpecies_index,int((width-1)/2),int((height-1)/2)],input_mean[GeoSpecies_index,int((width-1)/2),int((height-1)/2)],input_std[GeoSpecies_index,int((width-1)/2),int((height-1)/2)])
                loss_FarEnd.backward()   ## backward
                optimizer_FarEnd.step()  ## refresh training parameters
                losses.append(loss_FarEnd.item())

                # Calculate R2
                y_hat_FarEnd = outputs_FarEnd.cpu().detach().numpy()
                y_true_FarEnd = labels_FarEnd.cpu().detach().numpy()
                y_hat_FarEnd = np.array(y_hat_FarEnd,ndmin=1)
                #torch.cuda.empty_cache()
                print('Epoch: ', epoch, ' i th: ', i, 'y_hat size: ',y_true_FarEnd.shape)
                #print('y_hat:', y_hat)
                R2 = linear_regression(y_hat_FarEnd,y_true_FarEnd)
                R2 = np.round(R2, 4)
                #pred = y_hat.max(1, keepdim=True)[1] # 得到最大值及索引，a.max[0]为最大值，a.max[1]为最大值的索引
                correct_FarEnd += R2
                counts_FarEnd  += 1
                if (i + 1) % 10 == 0:
                # 每10个batches打印一次loss
                    print('Low End Model Epoch : %d/%d, Iter : %d/%d,  Loss: %.4f' % (epoch + 1, TOTAL_EPOCHS,
                                                                    i + 1, len(X_train) // BATCH_SIZE,
                                                                    loss_FarEnd.item()))
            scheduler_FarEnd.step() 

            valid_correct_LowEnd = 0
            valid_counts_LowEnd  = 0
            valid_correct_FarEnd = 0
            valid_counts_FarEnd  = 0
            for i, (valid_images_LowEnd, valid_labels_LowEnd) in enumerate(validation_loader_LowEnd):  
                model.eval()
                #valid_images_LowEnd = torch.squeeze(valid_images_LowEnd)
                valid_images_LowEnd = valid_images_LowEnd.to(device)

                #valid_labels_LowEnd = torch.squeeze(valid_labels_LowEnd.type(torch.FloatTensor))
                valid_labels_LowEnd = valid_labels_LowEnd.to(device)
                print('valid_images size: {}'.format(valid_labels_LowEnd.shape),'valid_labels size: {}'.format(valid_labels_LowEnd.shape))
                valid_outputs_LowEnd = model.model_A(valid_images_LowEnd)
                valid_outputs_LowEnd = torch.squeeze(valid_outputs_LowEnd)
                valid_loss_LowEnd = criterion(valid_outputs_LowEnd, valid_labels_LowEnd, valid_images_LowEnd[:,GeoSpecies_index,int((width-1)/2),int((height-1)/2)],input_mean[GeoSpecies_index,int((width-1)/2),int((height-1)/2)],input_std[GeoSpecies_index,int((width-1)/2),int((height-1)/2)])
                valid_losses.append(valid_loss_LowEnd.item())
                valid_y_hat_LowEnd = valid_outputs_LowEnd.cpu().detach().numpy()
                valid_y_true_LowEnd = valid_labels_LowEnd.cpu().detach().numpy()
                #print('test_y_hat size: {}'.format(test_y_hat.shape),'test_y_true size: {}'.format(test_y_true.shape))
                Valid_R2 = linear_regression(valid_y_hat_LowEnd,valid_y_true_LowEnd)
                Valid_R2 = np.round(Valid_R2, 4)
                valid_correct_LowEnd += Valid_R2
                valid_counts_LowEnd  += 1    
                print('Epoch : %d/%d, Iter : %d/%d,  Validate Loss: %.4f, Validate R2: %.4f' % (epoch + 1, TOTAL_EPOCHS,
                                                                    i + 1, len(X_train) // BATCH_SIZE,
                                                                    valid_loss_LowEnd.item(), Valid_R2))
            for i, (valid_images_FarEnd, valid_labels_FarEnd) in enumerate(validation_loader_FarEnd):  
                model.eval()
                #valid_images_FarEnd = torch.squeeze(valid_images_FarEnd)
                valid_images_FarEnd = valid_images_FarEnd.to(device)

                #valid_labels_FarEnd = torch.squeeze(valid_labels_FarEnd.type(torch.FloatTensor))
                valid_labels_FarEnd = valid_labels_FarEnd.to(device)
                print('valid_images size: {}'.format(valid_labels_FarEnd.shape),'valid_labels size: {}'.format(valid_labels_FarEnd.shape))
                valid_outputs_FarEnd = model.model_B(valid_images_FarEnd)
                valid_outputs_FarEnd = torch.squeeze(valid_outputs_FarEnd)
                valid_loss_FarEnd = criterion(valid_outputs_FarEnd, valid_labels_FarEnd, valid_images_FarEnd[:,GeoSpecies_index,int((width-1)/2),int((height-1)/2)],input_mean[GeoSpecies_index,int((width-1)/2),int((height-1)/2)],input_std[GeoSpecies_index,int((width-1)/2),int((height-1)/2)])
                valid_losses.append(valid_loss_FarEnd.item())
                valid_y_hat_FarEnd = valid_outputs_FarEnd.cpu().detach().numpy()
                valid_y_true_FarEnd = valid_labels_FarEnd.cpu().detach().numpy()
                #print('test_y_hat size: {}'.format(test_y_hat.shape),'test_y_true size: {}'.format(test_y_true.shape))
                Valid_R2 = linear_regression(valid_y_hat_FarEnd,valid_y_true_FarEnd)
                Valid_R2 = np.round(Valid_R2, 4)
                valid_correct_FarEnd += Valid_R2
                valid_counts_FarEnd  += 1    
                print('Epoch : %d/%d, Iter : %d/%d,  Validate Loss: %.4f, Validate R2: %.4f' % (epoch + 1, TOTAL_EPOCHS,
                                                                    i + 1, len(X_train) // BATCH_SIZE,
                                                                    valid_loss_FarEnd.item(), Valid_R2)) 
            accuracy_LowEnd = correct_LowEnd / counts_LowEnd
            accuracy_FarEnd = correct_FarEnd / counts_FarEnd
            test_accuracy_LowEnd = valid_correct_LowEnd / valid_counts_LowEnd
            test_accuracy_FarEnd = valid_correct_FarEnd / valid_counts_FarEnd
            print('Epoch: ',epoch, ', LowEnd Training Loss: ', loss_LowEnd.item(),', FarEnd Training Loss: ', loss_FarEnd.item(),',LowEnd Training accuracy:',accuracy_LowEnd, ',FarEnd Training accuracy:',accuracy_FarEnd, 
                  ', \n LowEnd Testing Loss:', valid_loss_LowEnd.item(),'FarEnd Testing Loss:', valid_loss_FarEnd.item(),',LowEnd Testing accuracy:', test_accuracy_LowEnd,', FarEnd Testing accuracy:', test_accuracy_FarEnd,)
            train_acc_LowEnd.append(accuracy_LowEnd)
            train_acc_FarEnd.append(accuracy_FarEnd)
            test_acc_LowEnd.append(test_accuracy_LowEnd)
            test_acc_FarEnd.append(test_accuracy_FarEnd)
            print('Epoch: ',epoch,'\nLearning Rate:',optimizer.param_groups[0]['lr'])

        train_acc.extend(train_acc_LowEnd)
        train_acc.extend(train_acc_FarEnd)
        test_acc.extend(test_acc_LowEnd)
        test_acc.extend(test_acc_FarEnd)
    
    elif ResNet_setting or ResNet_MLP_setting:
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
    if species == 'NO3':
        GeoSpecies_index = initial_channel_names.index('Geo{}'.format('NIT'))
    else:
        GeoSpecies_index = initial_channel_names.index('Geo{}'.format(species))
    
    model.eval()
    final_output = []
    final_output = np.array(final_output)
    predictinput = DataLoader(Dataset_Val(inputarray), batch_size= batchsize)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if TwoCombineModels_Settings:
        TwoCombineModel_Variable_index = initial_channel_names.index(TwoCombineModels_Variable)
        with torch.no_grad():
            for i, image in enumerate(predictinput):
                image = torch.squeeze(image)
                width = image.shape[2]
                height = image.shape[3]
                temp_output = np.zeros((image.shape[0]),dtype=np.float32)
                GeoSpecies = image[:,TwoCombineModel_Variable_index,int((width-1)/2),int((height-1)/2)]
                GeoSpecies_LowEnd_index = np.where(GeoSpecies <= TwoCombineModels_threshold)[0]
                GeoSpecies_FarEnd_index = np.where(GeoSpecies >= TwoCombineModels_threshold)[0]
                image_LowEnd = image[GeoSpecies_LowEnd_index,:,:,:].to(device)
                image_FarEnd = image[:,:,:,:].to(device)
                image_LowEnd = torch.squeeze(image_LowEnd)
                image_FarEnd = torch.squeeze(image_FarEnd)
                output_LowEnd = model.model_A(image_LowEnd).cpu().detach().numpy()
                output_FarEnd = model.model_B(image_FarEnd).cpu().detach().numpy()
                output_LowEnd = np.squeeze(output_LowEnd)
                output_FarEnd = np.squeeze(output_FarEnd)
                temp_output[GeoSpecies_LowEnd_index] = output_LowEnd
                if len(GeoSpecies_FarEnd_index) != 0:
                    temp_output[GeoSpecies_FarEnd_index] = output_FarEnd[GeoSpecies_FarEnd_index]
                final_output = np.append(final_output,temp_output)
    elif ResNet_setting or ResNet_MLP_setting:
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




    
    