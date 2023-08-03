from __future__ import print_function
import os
import torch 
import torch.nn as nn
import numpy as np
from F5_JACCARD import Jaccard
from F23_DULANORM_UNET_V1 import DualNorm_Unet



class Config(object):
    NAME= "dfaNet"

    #set the output every STEP_PER_EPOCH iteration
    STEP_PER_EPOCH = 100
    ENCODER_CHANNEL_CFG=ch_cfg=[[8,48,96],
                                [240,144,288],
                                [240,144,288]]


dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#dev = torch.device("cpu")
device = torch.device(dev) 


def train_model(n_epochs, trainloss, validationloss, accuracy, model, scheduler, lrFile, training_generator, optim, lim, trainFile, trainaccFile, trainepochFile, validation_generator, valFile, valaccFile, pathm, i, modeltype):
    training_losses = []
    for epoch in range(n_epochs):
        model.train()
        batch_losses = []
        jI = 0
        totalBatches = 0
        scheduler.step()
        print('Epoch:', epoch,'LR:', scheduler.get_last_lr()) # Used .get_last_lr() instead of .get_lr()
        lrFile.write('Epoch:'+' '+str(epoch)+' '+'LR:'+' '+str(scheduler.get_last_lr())+"\n")
        lrFile.write(str(scheduler.state_dict())+"\n")

        mb=0
        for trainim, trainmas in training_generator:
            mb+=1
            optim.zero_grad()
            images=trainim.to(device)
            masks=trainmas.to(device)
            outputs=model(images)
            
            if trainloss =='BCEWithLogitsLoss':
                loss=nn.BCEWithLogitsLoss() #loss = nn.BCELoss() 
                # Convert the target tensor to a single-channel mask
                #masks = masks[:, 0:1, :, :]
                output = loss(outputs, masks)            
            output.backward()
            optim.step()
            
            batch_losses.append(output.item())
            batchLoad = len(masks)*lim*lim
            totalBatches = totalBatches + batchLoad
            if accuracy == 'Jaccard':
                # Here I add batchLoad-1 instead of 1 
                thisJac = Jaccard(torch.reshape(masks,(batchLoad,1)),torch.reshape(outputs,(batchLoad,1)))*batchLoad
                #thisJac = Jaccard(torch.reshape(masks,(len(masks),-1)),torch.reshape(outputs,(len(masks),-1)))* len(masks) #Jaccard(torch.reshape(masks,(batchLoad,1)),torch.reshape(outputs,(batchLoad,1)))*batchLoad
                jI = jI+thisJac.data[0]
                       
        training_loss = np.mean(batch_losses)
        training_losses.append(training_loss)
        trainFile.write(str(training_losses[epoch])+"\n")
        trainaccFile.write(str((jI/totalBatches).item())+"\n")
        trainepochFile.write(str(epoch)+"\n")
        print("Training Jaccard:",(jI/totalBatches).item()," (epoch:",epoch,")")
        lrFile.write("Training loss:"+str(training_losses[epoch])+"\n")
        lrFile.write("Training accuracy:"+str((jI/totalBatches).item())+"\n")
        
        
        torch.save(model.state_dict(), os.path.join(pathm, "iremmodel{}.pt".format(i)))
        #list(model.parameters())
        #print("\n____________________")
        #print(model.state_dict())
        validate(validationloss, accuracy, validation_generator, valFile, valaccFile, lim, lrFile, pathm, i, modeltype)
    torch.save(model.state_dict(), os.path.join(pathm, "Finaliremmodel{}.pt".format(i)))        
        
                       
def validate(validationloss, accuracy, validation_generator, valFile, valaccFile, lim, lrFile, pathm, i, modeltype):
    jI = 0
    totalBatches = 0
    validation_losses = []
    
        
    if modeltype=='DualNorm_Unet':
        model = DualNorm_Unet(n_channels=3, n_classes=1).to(device)          
        
        # # Create an instance of the model
        # model = DualNorm_Unet(n_channels=3, n_classes=1, bilinear=False, batchsize=4, nonlinear='relu', norm_type='BN',
        #                     spade_seg_mode='soft', spade_inferred_mode='mask', spade_aux_blocks='', spade_reduction=2).to(device)


    model.load_state_dict(torch.load(os.path.join(pathm, "iremmodel{}.pt".format(i))))
    model.eval()
    with torch.no_grad():
        val_losses = []
        for valim, valmas in validation_generator:
            #model.eval()
            images=valim.to(device)
            masks=valmas.to(device)
            outputs=model(images)
            if validationloss == 'BCEWithLogitsLoss':
                loss=nn.BCEWithLogitsLoss() #loss = nn.BCELoss() 
                # Convert the target tensor to a single-channel mask
                #masks = masks[:, 0:1, :, :]
                
                output = loss(outputs, masks)
            val_losses.append(output.item())
            batchLoad = len(masks)*lim*lim
            totalBatches = totalBatches + batchLoad
            if accuracy == 'Jaccard':
                thisJac = Jaccard(torch.reshape(masks,(batchLoad,1)),torch.reshape(outputs,(batchLoad,1)))*batchLoad
                #thisJac = Jaccard(torch.reshape(masks,(len(masks),-1)),torch.reshape(outputs,(len(masks),-1)))* len(masks) # #Jaccard(torch.reshape(masks,(batchLoad,1)),torch.reshape(outputs,(batchLoad,1)))*batchLoad
                jI = jI+thisJac.data[0]
                #jI = jI+thisJac.item() 
    dn=jI/totalBatches
    dni=dn.item()
    validation_loss = np.mean(val_losses)
    validation_losses.append(validation_loss)
    valFile.write(str(validation_losses[0])+"\n")
    valaccFile.write(str(dni)+"\n")
    print("Validation Jaccard:",dni)
    lrFile.write("Validation loss:"+str(validation_losses[0])+"\n")
    lrFile.write("Validation accuracy:"+str(dni)+"\n")

