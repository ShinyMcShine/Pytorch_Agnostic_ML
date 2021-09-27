# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 16:38:05 2020
get_number_correct function from https://deeplizard.com/learn/video/p1xZ2yWU1eo
plot_confusion_matrix function from https://deeplizard.com/learn/video/0LhiS6yu2qQ
get_all_preds fucntion from https://deeplizard.com/learn/video/0LhiS6yu2qQ

This block of code will ingest your tensors and create confusion matrix.
Put this script into your directory sharing your other .py scripts

Most functions will only require the model and loader you want to use
You must also supply your labels but this can be done easily from your dataset
use dataset.classes to extract their labels.


@author: Daniel Campoy
"""
import itertools
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix



def imshow (img):
    img = img / 2 + 0.5 # unnomarlize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()
    
def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


def get_all_preds(model, loader):

    all_preds = torch.tensor([])
    for batch in loader:
        images, labels = batch

        preds = model(images)
        all_preds = torch.cat(
            (all_preds, preds)
            ,dim=0
        )
    return all_preds

def class_dist_Acc (model, loader, classes):
    
    #Class Distribution of Acc
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))     
    with torch.no_grad():
        all_preds = torch.tensor([])
        for data in loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(10):
                label = labels[i]
                
                #print('\nPrediction Label Match:', c[i].item())
                #print('Label names:', classes[label.item()])
                
                #stores Accuracy for each label
                class_correct[label] += c[i].item()
                class_total[label] += 1
                
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

if __name__ == '__main__':
    
    '''
    Here is an example of the code calling these functions
    
    #Cal results and plot Conf Matrix for each label
    with torch.no_grad():
        test_preds = get_all_preds(model,loader)
        #Need to convert List to tensor as dataset.targets output a list
        grdtruth = torch.LongTensor(testset.targets)
        preds_correct = get_num_correct(test_preds, grdtruth)
    
    print('\nTotal Correct:', preds_correct)
    print('Overall Acc:', preds_correct / len(testset))
    
    
    stacked = torch.stack((
            grdtruth, test_preds.argmax(dim=1))
            ,dim=1
            )
    
    print('Stack Shape:', stacked.shape)
    #Initalize matrix with 0s based on number of labels
    confmat = torch.zeros(10,10, dtype=torch.int64)
    #Loop through stacked and populate Confusion Matrix
    for p in stacked:
        tl, pl = p.tolist()
        confmat[tl, pl] = confmat[tl, pl]+1
    print('Built Confusion Matrix\n')
    print(confmat)
    
    cm = confusion_matrix(grdtruth, test_preds.argmax(dim=1))
    print(type(cm))
    print('SKlearn Confusion Matrix:\n', cm)
    
    plt.figure(figsize=(10,10))
    plot_confusion_matrix(cm, testset.classes)
    
    print('Reporting Class Accuracy Distribution.\n')
    
    class_dist_Acc(net, testloader, classes)
    '''