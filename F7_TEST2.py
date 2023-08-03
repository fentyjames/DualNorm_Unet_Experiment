from __future__ import print_function
import torch 
import os
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from F5_JACCARD import Jaccard
from F23_DULANORM_UNET_V3 import DualNorm_Unet
from F11_SEGPLOTT import segplot

# UNetV2, CamDUNet

dev = "cuda:0"  
dev = "cpu"
device = torch.device(dev) 

def test_model(test_generator, lim, testFile, testaccFile, i, modeltype, pathm, trMeanR, trMeanG, trMeanB):
    
    
    if modeltype=='DualNorm_Unet':
      net = DualNorm_Unet(n_channels=3, n_classes=1).to(device) 
             
   
         
    net.load_state_dict(torch.load(os.path.join(pathm, "Finaliremmodel{}.pt".format(i))))

    jI = 0
    totalBatches = 0
    test_losses = []
    net.eval()
    with torch.no_grad():
        t_losses = []
        t=0
        for testim, testmas in test_generator:
            images=testim.to(device)
            masks=testmas.to(device)
            outputs = net(images)
            if t==0:
                fig=plt.figure()
                axes=[]
                fimage=images[0].permute(1, 2, 0)
                fimage[:,:,0]=(images[0][0,:,:])
                fimage[:,:,1]=(images[0][1,:,:])
                fimage[:,:,2]=(images[0][2,:,:])
                fimage=fimage.cpu().numpy()
                axes.append(fig.add_subplot(1, 2, 1))
                foutput=outputs[0].permute(1, 2, 0)
                foutput=foutput.cpu().numpy()
                plt.imshow(np.squeeze(foutput, axis=2),  cmap='gray')
                subplot_title=("Test Predicted Mask")
                axes[-1].set_title(subplot_title)
                axes.append(fig.add_subplot(1, 2, 2))
                fmask=masks[0].permute(1, 2, 0)
                fmask=fmask.cpu().numpy()
                plt.imshow(np.squeeze(fmask, axis=2),  cmap='gray')
                #plt.imshow(np.squeeze(fmask, axis=2) if fmask.shape[2] > 1 else fmask[:, :, 0], cmap='gray')
                # if len(fmask.shape) > 2:
                #     fmask = np.squeeze(fmask, axis=2)
                # else:
                #     fmask = fmask[:, :, 0]

                # plt.imshow(fmask, cmap='gray')
                # if len(fmask.shape) > 2:
                #     fmask = fmask[:, :, 0]

                plt.imshow(fmask, cmap='gray')
                                
                subplot_title=("Ground Truth Mask")
                axes[-1].set_title(subplot_title)
                n_curve = 'mask_comparison.png'
                plt.savefig(os.path.join(pathm, n_curve))
                plt.show()
                segplot(pathm, lim, fimage, foutput, fmask,  trMeanR, trMeanG, trMeanB)
            losst=nn.BCEWithLogitsLoss()  #losst = nn.BCELoss() 
            # Convert the target tensor to a single-channel mask
            #masks = masks[:, 0:1, :, :]
            
            output = losst(outputs, masks)
            t_losses.append(output.item())
            batchLoad = len(masks)*lim*lim
            totalBatches = totalBatches + batchLoad
            thisJac = Jaccard(torch.reshape(masks,(batchLoad,1)),torch.reshape(outputs,(batchLoad,1)))*batchLoad
            #thisJac = Jaccard(torch.reshape(masks,(len(masks),-1)),torch.reshape(outputs,(len(masks),-1)))* len(masks) #Jaccard(torch.reshape(masks,(batchLoad,1)),torch.reshape(outputs,(batchLoad,1)))*batchLoad
            jI = jI+thisJac.data[0]
            #jI = jI+thisJac.item()
            t +=1
                 
    dn=jI/totalBatches
    dni=dn.item()
    test_loss = np.mean(t_losses)
    test_losses.append(test_loss)
    testFile.write(str(test_losses[0])+"\n")
    testaccFile.write(str(dni)+"\n")
    print("Test Jaccard:",dni)
