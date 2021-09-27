
import argparse as args
import ML_model_base as ML
import logging
import os
import torchvision.models as models
import torch.nn as nn
import torch
import sys


def train_model(json):
	
	#grab json file name for logger 
	file_name = json.split('/')[-1]
	#Define Logger info for output file
	logger = logging.getLogger(__name__)
	logger.addHandler(logging.FileHandler(f"{file_name}_training_log.txt"))
	logger.addHandler(logging.StreamHandler(sys.stdout))
	logger.setLevel(logging.INFO)

	#parse json params
	model_params, train_params, test_params, options_params = ML.parse_params(json)

	#Hyper paramater could be a list with the dictionary 

	#Create model, criterion, and optimizer
	if model_params["Name"] =='resnet18':
		model =  models.resnet18(pretrained=False, num_classes=model_params["NumClasses"])

	if model_params["Name"] == 'resnet101':
		model = models.resnet101(pretrained=False, num_classes=model_params["NumClasses"])

	if model_params["Name"] == 'densenet121':
		model = models.densenet121(pretrained=False, num_classes=model_params["NumClasses"])

	if model_params["Criterion"] == 'cross_entropy':
		criterion = nn.CrossEntropyLoss()

	if model_params["Optimizer"] == 'SGD':
		optimizer = torch.optim.SGD(model.parameters(), momentum=model_params["Momentum"], weight_decay=model_params["WeightDecay"], lr=model_params["Learn_Rate"])

	if "Learn_Rate" in model_params:
		learn_rate = model_params["Learn_Rate"]
	
	#Extract training params
	train_imgdir = train_params["Trainpath"]
	trn_batch_size = train_params["Batch_Size"]
	train_transforms = ML.get_transforms(train_params)
	trn_shuffle = train_params["Shuffle"]
	#extract options
	workers = options_params["Workers"]
	early_stop = options_params["Early_Stop"]
	save_model = options_params["Save_Model_Dir"]
	epochs = options_params["Epochs"]

	
	#Extract testing params
	test_imgdir = test_params["Testpath"]
	tst_transforms = ML.get_transforms(test_params)
	tst_batch_size = test_params["Batch_Size"]
	tst_shuffle = test_params["Shuffle"]

	if options_params["CustImgDS"][0]["enable"] == True:
		trainloader = ML.getcustloader(options_params["CustImgDS"][0]["trncsvfile"], 
										imgdir, train_transforms, 
										trn_batch_size, workers, trn_shuffle)
		testloader = ML.getcustloader(options_params["CustImgDS"][0]["tstcsvfile"], 
										imgdir, tst_transforms, 
										tst_batch_size, workers, tst_shuffle)
	else:
		trainloader = ML.getloader(train_imgdir, trn_batch_size, train_transforms, workers, trn_shuffle)
		testloader = ML.getloader(test_imgdir, tst_batch_size, tst_transforms, workers, tst_shuffle)

	# Train model
	ML.train_loop(model, optimizer, learn_rate, criterion, 
					trainloader, testloader, epochs, logger, save_model, file_name,
					model_params["Name"], early_stop=early_stop)
	

if __name__ == '__main__':
	parser = args.ArgumentParser()
	parser.add_argument('-json', type=str, help='path of JSON params file')
	args = parser.parse_args()

	train_model(args.json)