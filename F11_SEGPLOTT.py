from __future__ import print_function
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


def segplot(pathm, lim, fimage, foutput, fmask, trMeanR, trMeanG, trMeanB):
    print("===============================================================================\n\n")
    print('image shape', fimage.shape, 'output shape', foutput.shape, 'groundtruth shape', fmask.shape)

    fimage[:, :, 0] = fimage[:, :, 0] + trMeanR
    fimage[:, :, 1] = fimage[:, :, 1] + trMeanG
    fimage[:, :, 2] = fimage[:, :, 2] + trMeanB
    fimage = (fimage - np.min(fimage)) / (np.max(fimage) - np.min(fimage))

    v = fimage[:, :, 0] / 4 + np.squeeze(foutput) / 2 + np.squeeze(fmask) / 4
    s = np.minimum(np.squeeze(fmask + foutput), np.ones((lim, lim), np.float32))


    h = 0.75 - np.squeeze(fmask) / 2
    print(s[55, 76])
    
    h = h * 179
    h = h.astype(np.uint8)

    v = v * 255
    v = v.astype(np.uint8)

    s = s * 255
    s = s.astype(np.uint8)

    #hsv_image = np.stack([h, s, v], axis=-1)
    #hsv_image = np.stack([h, s, v])
    hsv_image = cv2.merge([h, s, v])
    out = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    outname = 'segmentation_image.png'
    cv2.imwrite(os.path.join(pathm, outname), out)
    
    # Add to make an image a third dimension
    #fimage = np.expand_dims(fimage, axis=2)

    plt.imsave(os.path.join(pathm, 'test_image.png'), fimage)
    #plt.imsave(os.path.join(pathm, 'test_image_R.png'), cv2.cvtColor(fimage, cv2.COLOR_BGR2GRAY))
    plt.imsave(os.path.join(pathm, 'test_image_R.png'), fimage[:, :, 0], cmap="gray")
    plt.imsave(os.path.join(pathm, 'test_image_G.png'), fimage[:, :, 1], cmap="gray")
    plt.imsave(os.path.join(pathm, 'test_image_B.png'), fimage[:, :, 2], cmap="gray")
    plt.imsave(os.path.join(pathm, 'test_pred_mask.png'), np.squeeze(foutput))
    plt.imsave(os.path.join(pathm, 'ground_truth_mask.png'), np.squeeze(fmask))
    
    #plt.imsave(os.path.join(pathm, 'test_image_R.png'), fimage[:, :, 0], cmap="gray")
