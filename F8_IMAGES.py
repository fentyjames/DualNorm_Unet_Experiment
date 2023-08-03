from __future__ import print_function
import os
import torch 
import numpy as np
import scipy.io as sio
import datetime
from torchvision import transforms


def get_images(trainSetSize, fno, fsiz, tsind, trind, vlind, chindex): 
    input_images = []
    target_masks = []    
    gettingfiles = []
    #C:/Users/user/Documents/codes/RGBs
        
    if chindex == 'RGBs':
        names = os.listdir('RGBs')
        for b in names[0:trainSetSize]:
            gettingfiles.append(b)
            a = sio.loadmat('RGBs/{}'.format(b))
            a = a['inputPatch']
            input_images.append(a)
            c = sio.loadmat('labels/{}'.format(b))
            c = c['inputPatch']
            target_masks.append(c)
    
    
    indFile = open("tsind.txt","w")
    d=datetime.datetime.now()
    testnames = open("testnames_{}_{}_{}_{}_{}.txt".format(d.year, d.month, d.day, d.hour, d.minute),"w")
    for say in range(0,len(tsind)):
        indFile.write(str(tsind[say])+'\n')
        testnames.write(str(gettingfiles[tsind[say]])+'\n')
    indFile.close(); 
    testnames.close(); 

    indFile = open("trind.txt","w"); 
    for say in range(0,len(trind)):
        indFile.write(str(trind[say])+'\n')
    indFile.close(); 
    
    indFile = open("vlind.txt","w"); 
    for say in range(0,len(vlind)):
        indFile.write(str(vlind[say])+'\n')
    indFile.close(); 
    
    
    input_images = np.asarray(input_images, dtype=np.float32)
    target_masks = np.asarray(target_masks, dtype=np.float32)
    
    
    print(f"Original image size: {input_images.shape} Original target mask size: {target_masks.shape}")
    print("===============================================================================\n")
       
    lim=224
    input_images = np.reshape(input_images[0:trainSetSize*lim*lim], (trainSetSize, lim, lim, 3)) 
    input_images = np.moveaxis(input_images,3,1)
    
        
    target_masks = np.reshape(target_masks[0:trainSetSize*lim*lim], (trainSetSize, 1, lim, lim))
    #target_masks = np.resize(target_masks[0:trainSetSize*lim*lim], (trainSetSize, 1, lim, lim))
    #print("Reshaped mask size:", target_masks.shape)
    print(f"Reshaped image size: {input_images.shape} Reshaped mask size: {target_masks.shape}")
    print("===============================================================================\n\n")
    
           
    trMeanR = input_images[trind,0,:,:].mean()
    trMeanG = input_images[trind,1,:,:].mean()
    trMeanB = input_images[trind,2,:,:].mean()
    
    input_images[:,0,:,:] = input_images[:,0,:,:] - trMeanR
    input_images[:,1,:,:] = input_images[:,1,:,:] - trMeanG
    input_images[:,2,:,:] = input_images[:,2,:,:] - trMeanB
    
    input_images=torch.from_numpy(input_images)
    target_masks=torch.from_numpy(target_masks)
    
    print("image size",input_images.shape,"mask size",target_masks.shape)
    
    print("type image",type(input_images),"type mask",type(target_masks)) 
    
    return input_images, target_masks, trMeanR, trMeanG, trMeanB
    print("===============================================================================\n\n\n")


