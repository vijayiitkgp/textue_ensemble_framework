import numpy as np
import torch
from barbar import Bar
import matplotlib.pyplot as plt
#import cv2 as cv
# from features import get_features

def test_model(model, dataloader, device):
    #Initialize and accumalate ground truth, predictions, and image indices

    GT = np.array(0)
    Predictions = np.array(0)
    Index = np.array(0)
    
    running_corrects = 0
    running_corrects2 = 0
    model.eval()
    
    # Iterate over data
    print('Testing Model...')
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(Bar(dataloader)):
            inputs = inputs.to(device)
            labels = labels.to(device)
            # index = index.to(device)
            # features = get_features(inputs, model, labels)
            # cam_image = cv.cvtColor(features, cv.COLOR_RGB2BGR)
            # plt.imshow(cam_image)
            # cv.imwrite("activation_fap.jpg", cam_image)
            # cam_image.save()

            # forward
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            # output = tf_rep.run(np.asarray(inputs, dtype=np.float32))
            # print('The texture is classified as ', np.argmax(output))
            # preds2 =  np.argmax(output)
            #If test, accumulate labels for confusion matrix
            GT = np.concatenate((GT,labels.detach().cpu().numpy()),axis=None)
            Predictions = np.concatenate((Predictions,preds.detach().cpu().numpy()),axis=None)
            # Index = np.concatenate((Index,index.detach().cpu().numpy()),axis=None)
            
            running_corrects += torch.sum(preds == labels.data)
            # running_corrects2 += torch.sum(preds2 == labels.data)

            

    test_acc = running_corrects.double() / (len(dataloader.sampler))
    print('Test Accuracy pytorch: {:4f}'.format(test_acc))
    # test_acc2 = running_corrects2.double() / (len(dataloader.sampler))
    # print('Test Accuracy tensorflow: {:4f}'.format(test_acc2))
    
    test_dict = {'GT': GT[1:], 'Predictions': Predictions[1:], #'Index':Index[1:],
                 'test_acc': np.round(test_acc.cpu().numpy()*100,2)}
    
    return test_dict
