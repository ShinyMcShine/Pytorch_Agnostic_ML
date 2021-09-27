import torch
import torch.nn as nn
import torch.functional as f
import gzip
import numpy as np
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import argparse as args
import sys
import pandas as pd
import os
import json
#from torchvision.io.image import read_image, ImageReadMode
from PIL import Image
from skimage import io, transform
from torch.utils.data import Dataset
from torch.utils.data import random_split
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import time
from imgaug import augmenters as iaa
import numpy as np
from PIL import Image
from datetime import datetime
import Tensor_ConfMatrixPlot as measure

#Set seed for pytorch randomization bug
seed = np.random.randint(0,42000)

np.random.seed(seed)
np.random.default_rng(seed=seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
#random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#Set device to cuda if possible
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class imgaug_transform:
  def __init__(self,augs):
    self.aug = iaa.Sequential(augs)

  def __call__(self, img):
    img = np.array(img)

    return Image.fromarray(self.aug.augment_image(img))

#Custom dataset class for images stored as root/img_1...img_n with no subfolders as labels
class CustomImageDataset(Dataset):
	def __init__(self, csv_file, imgdir, transform=None):
		self.imgs_and_labels = pd.read_csv(csv_file)
		self.imgdir = imgdir
		self.transform = transform
	
	def __len__(self):
		return len(self.imgs_and_labels)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		#combines root and image name from dataframe column 1	
		img_path = os.path.join(self.imgdir, self.imgs_and_labels.iloc[idx,0])

		#Load image as tensor object C X H X W
		#img = read_image(img_path, mode=ImageReadMode.RGB) #original line of code but using customer tensor seems to cause error

		img = io.imread(img_path) #original line of code but using customer tensor seems to cause error
		
		#Extract label from data frame column 2
		label = self.imgs_and_labels.iloc[idx,1]
		
		#Apply transform if it was provided
		if self.transform:
			img = self.transform(img)
		# set image pixels from 0 to 1
		#img = img/255
		#return dictionary of image and its label both a tensor
		#return {"image": img, 
		#		"label": torch.from_numpy(np.array(label))}

		return img, torch.from_numpy(np.array(label))

#Define random state worker id for true randomization when using parallel processing (more than 1 worker)
def worker_init_fn(worker_id):                                                          
	np.random.seed(np.random.get_state()[1][0] + worker_id)

#Plot first 4 images from dataset
def loaddata_plotsmpl(Imagedataset):
	#establish figure
	fig = plt.figure()

	#plot first 4 images and its label for reference
	for i in range(len(Imagedataset)):
		sample = Imgdataset[i]
		smplimg = sample['image'].permute(1,2,0)
		print(i, sample['image'].shape, sample['label'])

		ax = plt.subplot(1, 4, i + 1)
		plt.tight_layout()
		ax.set_title('Sample #{}'.format(i) + ' Label: ' + str(sample['label']))
		ax.axis('off')
		plt.imshow(smplimg)

		if i == 3:
			plt.show()
			break

def get_transforms(params):
	transform_list = []
	
	#call class imaug transforms
	#imaugs = imgaug_transform()

	for tr in params["Transforms"]:
		#Define CenterCrop
		if tr["type"] == "CenterCrop":
			size = int(tr["size"])
			transform_list.append(imgaug_transform(iaa.CenterCropToSquare(size)))
		#Define Affine
		if tr["type"] == "Affine":
			scale = (float(tr['scale'][0]), float(tr["scale"][1]))
			shear = (float(tr['shear'][0]), float(tr["shear"][1]))
			rotate = (int(tr['rotate'][0]), int(tr["rotate"][1]))
			transform_list.append(imgaug_transform(iaa.Affine(scale=scale, shear=shear, rotate=rotate)))
		#Define Add
		if tr["type"] == "Add":
			p = (int(tr['p'][0]), int(tr["p"][1]))
			#False is the default for per_channel
			transform_list.append(imgaug_transform(iaa.Add(value=p, per_channel=True)))
		if tr["type"] == "GammContrast":
			contrast = (float(tr["contrast"][0]), float(tr["contrast"][1]))
			transform_list.append((imgaug_transform(iaa.GammaContrast(gamma=contrast))))
		#if tr["type"] == "RandomPad":
		#	size = int(tr[size])
		#	transform_list.append(transforms.Pad()))
		#percentage to data to flip vertical
		if tr["type"] == "RandomVerticalFlip":
			percentage = float(tr["percentage"])
			transform_list.append(imgaug_transform(iaa.Flipud(percentage)))
		if tr["type"] == "Resize_Bicubic":
			range_ofvalues = len(tr["dims"])
			dims = []
			for i in range(range_ofvalues):
				dims.insert(i, tr["dims"][i])
			dims = tuple(dims)
			transform_list.append(transforms.Resize(dims,interpolation=InterpolationMode.BICUBIC))
		if tr["type"] == "Resize_Bilinear":
			range_ofvalues = len(tr["dims"])
			dims = []
			for i in range(range_ofvalues):
				dims.insert(i, tr["dims"][i])
			dims = tuple(dims)
			print(f"Dims for Bilinear {dims}")
			transform_list.append(transforms.Resize(dims,interpolation=InterpolationMode.BILINEAR))
		#percentage of data to flip horizontal
		if tr["type"] == "RandomHorizonalFlip":
			percentage = float(tr["percentage"])
			transform_list.append(imgaug_transform(iaa.Fliplr(percentage)))
		#Set image to GrayScale
		if tr["type"] == "ToGrayScale":
			transform_list.append(transforms.Grayscale(num_output_channels=3))	
		if tr["type"] == "Normalize":
			range_ofvalues = len(tr["mean"])
			mean = []
			std = []
			for i in range(range_ofvalues):
				mean.insert(i, tr["mean"][i])
			mean = tuple(mean)
			for i in range(range_ofvalues):
				std.insert(i,tr["std"][i])
			std = tuple(std)
			
			transform_list.append(transforms.Normalize(mean, std))
		if tr["type"] == "ToTensor":
			#Convert to Tensor
			transform_list.append(transforms.ToTensor())
	
	#Compose transforms
	built_transforms = transforms.Compose(transform_list)

	return built_transforms

def getloader(path, batch_sz, built_transforms, workers, shuffle):

	dataset = datasets.ImageFolder(path, transform=built_transforms) #for tansformations of test data

	dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_sz, shuffle=shuffle, num_workers=workers, worker_init_fn=worker_init_fn)
	
	#classes = testset.classes

	return dataloader

def getcustloader (df, imgdir, built_transforms, batch_sz, workers, shuffle):
	#Create dataset of adv images with their labels
	custdataset = CustomImageDataset(df=imgdf, imgdir=imgdir)

	#create dataloader
	dataloader = DataLoader(custdataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers, worker_init_fn=worker_init_fn) 	

	return dataloader

def train (model, trainloader, criterion, optimizer, logger):
	model.to(device)
	model.train()
	running_loss = 0.0
	
	total_predictions = 0
	correct_predictions = 0	


	logger.info(f'Device in use {device}')

	for X, y_true in trainloader:
		X = X.to(device)
		#logger.info(f"X shape {X.shape}")
		y_true = y_true.to(device)

		optimizer.zero_grad()

		# Forward pass
		y_hat = model(X)

		loss = criterion(y_hat, y_true)

		loss.backward()
		optimizer.step()
		
		running_loss += loss.item() #* X.size(0)
		
		#logger.info(f"y_hat output {y_hat}")
		# Backward pass

		#Grab the softmax output define
		#softmax = nn.Softmax(dim=1).to(device)
	
		#sftmx = softmax(y_hat).to(device)

		#logger.info(f'Softmax outut == {sftmx}') 

		#pred_labels = np.argmax(sftmx, axis=1).to(device)
		_, pred_labels = torch.max(y_hat, 1)
		#logger.info(f"y_true size / {y_true.size(0)} and type {type(y_true.size(0))}")
		#print(f'Argmax output from softmax ## {pred_labels}')
		#print(f'Size of label y_true {y_true.size()}')
		#print(f'Size of Softmax {sftmx.size()}')
		
		total_predictions += y_true.size(0)
		correct_predictions += (pred_labels == y_true).sum().item()

	train_acc = correct_predictions / total_predictions
	
	trainloss = running_loss / len(trainloader.dataset)
	
	#logger.info(f'Total number of predictions : {total_predictions}')
	#logger.info(f'Train accuracy {train_acc}')
	#logger.info(f"{type(avg_train_acc[0].cpu())}")
		
	return model, trainloss, train_acc

def test (model, testloader, criterion, logger):
	running_loss = 0.0

	model.eval()
	
	total_predictions = 0
	correct_predictions = 0

	with torch.no_grad():
		for X, y_true in testloader:
			#put objects onto device
			X = X.to(device)
			#logger.info(f'Shape of each batch {X.shape}')
			y_true = y_true.to(device) #truth target

			#output = model(X) #used for softmax extraction			
			# Forward pass
			y_hat = model(X)

			loss = criterion(y_hat, y_true)
			running_loss += loss.item()# * X.size(0)					
			#Grab the softmax output define then extract
			#softmax = nn.Softmax(dim=1)
		
			#sftmx = softmax(output).cpu()
			#print(f'Softmax outut == {sftmx}') 
			#pred_labels = np.argmax(sftmx, axis=1).to(device)
			_, pred_labels = torch.max(y_hat.data, 1)
			#print(f'Argmax output from softmax ## {pred_labels}')
			#print(f'Size of label y_true {y_true.size()}')
			#print(f'Size of Softmax {sftmx.size()}')
			
			total_predictions += y_true.size(0)
			correct_predictions += (pred_labels == y_true).sum().item()

		test_acc = correct_predictions / total_predictions
		
		testloss = running_loss / len(testloader.dataset)
		
		#logger.info(f'Testing loss // {testloss}')
		#logger.info(f'Total number of predictions : {total_predictions}')
		#logger.info(f'Test accuracy {test_acc}')
		
		
		return model, testloss, test_acc

def get_accuracy(model, data_loader):
    #print(f'Device in use {device}')

	correct_predictions = 0
	total_predictions = 0
	
	with torch.no_grad():
		for img, y_true in data_loader:

			img = img.to(device)
			y_true = y_true.to(device)

			y_hat = model(img)#.to(device)
			_, predicted_labels = torch.max(y_hat, 1)

			total_predictions += y_true.size(0)
			correct_predictions += (predicted_labels == y_true).sum().item()

	return correct_predictions / total_predictions

def train_loop(model, optimizer, learning_rate, criterion, trainloader, testloader,  epochs, logger, save_model_dir, file_name, model_name, early_stop=5):
	all_trainloss = []
	all_testloss = []
	best_val_loss = float('inf')
	epochs_without_improvement = 0
	avg_test_acc = []
	avg_train_acc = []

	#Epoch loop
	for e in range(epochs):
		
		#training
		model, train_loss, train_acc = train(model, trainloader, criterion, optimizer, logger)
		all_trainloss.append(train_loss)
		#train_acc = get_accuracy(model, trainloader)
		#add train accuarcy to avg
		avg_train_acc.append(train_acc)

		# testing
		model, test_loss, test_acc = test(model, testloader, criterion, logger)
		all_testloss.append(test_loss)
		#test_acc = get_accuracy(model, testloader)
		avg_test_acc.append(test_acc)

		# early stopping if loss isn't improving
		if test_loss < best_val_loss:
			logger.info(f"Test loss is better then currently best loss")
			best_val_loss = test_loss
			logger.info(f"Current best loss is {best_val_loss}")
			epochs_without_improvement = 0
			
			torch.save(model.state_dict, save_model_dir + model_name + "_"+ file_name +".pth")
		else:
			logger.info(f"Current loss is {test_loss} and best loss is {best_val_loss}")
			epochs_without_improvement += 1			
			logger.info(f"Epochs without improvement {epochs_without_improvement}")
			if epochs_without_improvement > early_stop:
				logger.info("Stopping early due to lack of loss improvement.")
				break
		
		logger.info(f'{datetime.now().time().replace(microsecond=0)} --- '
					f'Epoch: {e}\n'
					f'Train loss: {train_loss:.4f}\n'
					f'Test loss: {test_loss:.4f}\n'
					f'Train accuracy: {100 * train_acc:.2f}\n'
					f'Test accuracy: {100 * test_acc:.2f}\n'
					f'Average training accuracy {100 * np.mean(avg_train_acc):.2f} accross all epochs\n'
					f'Average testing accuracy {100 * np.mean(avg_test_acc):.2f} accross all epochs\n')
	
					
def parse_params (paramsfile):
	
	# Opening JSON file
	with open(paramsfile) as json_file:
		data = json.load(json_file)

	#params = data.values()

	#extract model params dict
	model_params = data['model']
	#extract train data params dict
	train_params = data['traindata']
	#extract test data params dict
	test_params = data['testdata']
	#extract options params dict
	options_params = data['options']

	return model_params, train_params, test_params, options_params