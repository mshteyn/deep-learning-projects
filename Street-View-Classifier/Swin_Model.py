## Training protocol for Pittsburgh-area building classification ##
import numpy as np
import pandas as pd
import os
import pdb

# filter empty (no image data) and low-res jpgs from the final database
def remove_empty_images(image_dir, limit):
	import pathlib
	cut_list = []
	keep_list = []
	with os.scandir(image_dir) as it:
		for entry in it: #iterate through alls canned files in dir
			if entry.is_file(): #only consider files, not nested dirs
				fsize = entry.stat().st_size
				#decode filename, remove jpg extension
				path_str = os.fsdecode(entry)	
				# generate lists of files to keep and files to cut
				if path_str[-4:] == '.jpg':
					#if file is under 8kb, there's no image
					if fsize < limit:
						fname = path_str.rsplit('/',1)[1][:-4] 				
						cut_list.append(fname)
					else:
						fname = path_str.rsplit('/',1)[1][:-4] 				
						keep_list.append(fname)					
	out_lists = {'keep' : keep_list, 'cut' : cut_list}
	return out_lists
# remove outliers and discretize the database
def clean_data(raw_csv, out_lists, features):
	raw_data = pd.read_csv(raw_csv, sep=',') 
	# import matplotlib.pyplot as plt	
	# import seaborn as sns
	
	#keep only entries with DL'd images	
	keep_data = raw_data[raw_data['PARID'].isin(out_lists['keep'])] 
	
	#remove extreme outliers >3std
	for y_param in features.keys():
		q_low = keep_data[y_param].quantile(0.01)
		q_hi  = keep_data[y_param].quantile(0.99)
		keep_data = keep_data[(keep_data[y_param] < q_hi) & (keep_data[y_param] > q_low)]#discretize into quintiles 
	y_ct = 0 # count total ys	
	indices = [y_ct]
	xy_df = keep_data['PARID']	
	for y_param in features.keys():
		if features[y_param][1] == 1: # if discretize
			num_quantiles = features[y_param][0].item()
			# seperate into quintiles
			f_abv = y_param[::4] #abbreviation of feature name
			qlabs = np.round(np.linspace(0,1, num_quantiles+1),2)[1:]
			keep_data[y_param] = pd.qcut(keep_data[y_param], num_quantiles, labels=qlabs)
		else:
			num_quantiles = keep_data[y_param].nunique()
		# one-hot encoding according to the quintile label
		one_hot_encoding = pd.get_dummies(keep_data[y_param], prefix=f_abv, dtype=int)
		xy_df = pd.concat([xy_df, one_hot_encoding], axis=1)
		y_ct+=num_quantiles
		indices.append(y_ct)

	# keep these labels
	out = {'processed_df' : xy_df, 'y_idx' : indices}
	return out

## CLEAN THE DATA ##

#csv_path = 'housing_data_raw.csv'
csv_path ='pgh_housing_data.csv'
img_path = 'images/'
size_cutoff = 10000 # 8000 = 8kb
#remove images with no pixel data
out_lists  = remove_empty_images(img_path, size_cutoff) #remove tiny images(no image data)

# { FEATURE : [number of quintiles, remove outliers]}
features_df = pd.DataFrame.from_dict({ "FAIRMARKETTOTAL" : [5, 1], 
									#"YEARBLT" : [5, 1],
									#"FINISHEDLIVINGAREA" : [5, 1],
									#"TOTALROOMS" : [7, 0],	
									#"CONDITION" : [5, 0]	#range 1-8							
									})
out = clean_data(csv_path, out_lists, features_df)
keep_df = out['processed_df']
y_idx = out['y_idx']
num_y = y_idx[-1] # how many ys to predict
num_feat = features_df.shape[1] # number of features

## IMPORT TORCH FUNCTIONS ##
import pickle
import torch
import torch.nn as nn  
import torch.optim as optim  
import torchvision.transforms as transforms  
import torchvision
from tqdm import tqdm
from skimage import io
from skimage.transform import rescale, resize, downscale_local_mean
from torch.utils.data import Dataset, DataLoader



# Parameters
learning_rate = 0.0005
batch_size = 64
num_epochs = 15
train_dir ='images/'
test_dir = 'images/'


keep_df = keep_df.iloc[:int(keep_df.shape[0]*1)] # subsample dataset

#randomly split test/trainx	x
from sklearn.model_selection import train_test_split
train_df, eval_df = train_test_split(keep_df, test_size=0.05)

print('\n\nTraining set size: ' + str(train_df.shape[0]))
print('Testing set size: ' + str(eval_df.shape[0]))
print('Total batches per train epoch: ' + str(int(train_df.shape[0]/batch_size)) + '\n\n')

# Generate dataset for loader
class DS(Dataset):
	def __init__(self,df,img_dir,tranform=None):
		#self.label=pd.read_csv(csv_file)
		self.label = df
		self.img_dir=img_dir
		self.tranform=tranform
		
	def __len__(self):
		return len(self.label)
	
	def __getitem__(self,index):
		img_path=os.path.join(self.img_dir,self.label.iloc[index,0]+'.jpg')
		image=io.imread(img_path)
		image=resize(image,(232,320),anti_aliasing=True) #ensure universal size
		image=image[:, :-20, :]  # Crop image border
		#image=resize(image,(384,384),anti_aliasing=True) #for vit_B
		image=resize(image,(256,256),anti_aliasing=True) #for vit_L
		y_label=torch.tensor(self.label.iloc[index,1:]).int()
		self.fnames = img_path
		self.index = index
		if self.tranform:
			image=self.tranform(image)
		return (image,y_label)

# perform validation on test set
def validate(loader, model, loss_fn, feature_idxs):
	pred_labels = np.array([]).astype(int)
	total_correct = 0
	total_samples = 0	
	#switch to eval mode -> don't update weights while checking acc.
	model.eval()
	loss, accu = [],[]
	num_labels = len(feature_idxs)
	uq_err_labs, uq_err_preds = torch.zeros(num_labels, device='cuda:0'), torch.zeros(num_labels, device='cuda:0')

	with torch.no_grad():
		for x, y in tqdm(loader):
			x = x.to(device=device, dtype=torch.float)
			y = y.to(device=device, dtype=torch.float)
			scores = model(x)
			
			loss.append(loss_fn(scores[:, feature_idxs], y[:, feature_idxs]).item())
			#loss.append(loss_fn(scores, y).item()) # single loss
			y_hat = scores[:, feature_idxs].max(1)[1]
			y_col = torch.argwhere(y[:, feature_idxs])[:, 1] # y_column
			total_correct += (y_hat == y_col).sum()
			total_samples += y_hat.size(0)
			accu.append(float(total_correct)/float(total_samples)*100)
			#classify prediction errors
			err_labs  = y_col[~(y_hat == y_col)] # get misclassified label
			err_preds = y_hat[~(y_hat == y_col)] # get misclassified preds

			# get categories that were misclassified
			uq_idxs = torch.unique(err_labs, return_counts=True)[0]
			uq_errs = torch.unique(err_labs, return_counts=True)[1]
			uq_err_labs[uq_idxs]  += uq_errs
			# get wrong y_hats (predictions made)
			uq_idxs = torch.unique(err_preds, return_counts=True)[0]
			uq_preds = torch.unique(err_preds, return_counts=True)[1]
			uq_err_preds[uq_idxs]  += uq_preds
	print('\n\n'+ str(total_correct.item()) +'/' +str((total_samples)) + " classified correctly @ " + str(np.round(float(total_correct)/float(total_samples)*100, 2)) + "% accuracy.\n")
	# return both misclassified labels and mispredictions
	out = {'val_loss' : np.mean(loss), 'val_accu' : np.mean(accu), 'err_labs' : uq_err_labs, 'err_preds' : uq_err_preds}
	#switch back to train mode -> continue updating weights during transfer learning
	model.train()
	return out

#generate the loaders
train_set = DS(df = train_df, img_dir=train_dir, tranform=transforms.ToTensor())
test_set = DS(df =eval_df, img_dir=test_dir, tranform=transforms.ToTensor())
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)


# Model
#model = torchvision.models.resnet50(weights='IMAGENET1K_V2')
#model = torchvision.models.vit_b_16(weights='ViT_B_16_Weights.IMAGENET1K_V1')
#model = torchvision.models.vit_l_16(weights='ViT_L_16_Weights.IMAGENET1K_V1')
model = torchvision.models.swin_v2_t(weights='Swin_V2_T_Weights.IMAGENET1K_V1')
#most recent weights trained

model.fc=nn.Sequential(
	nn.Linear(in_features=4096,out_features=2048),
	nn.ReLU(),
	nn.Linear(in_features=2048,out_features=1024),
	nn.ReLU(),
	nn.Linear(in_features=1024,out_features=512),
	nn.ReLU(),
	nn.Linear(in_features=512,out_features=128),
	nn.ReLU(),
	nn.Linear(in_features=128,out_features=32),
	nn.ReLU(),
	nn.Linear(in_features=32,out_features=num_y,bias=True)
) 
#load previosly trained model?
load_model = 0
prev_trained_mod_dir = 'saved_mod_64_e2'
if load_model == 1:
	state_dict = torch.load(prev_trained_mod_dir)
	model.load_state_dict(state_dict)
	print('Previously saved model loaded successfully')

#to GPU if available
if torch.cuda.is_available():
	device = torch.device("cuda")
else:
	device = torch.device("cpu")

print(device)
model.cuda();
model

# model params
loss_fn = nn.CrossEntropyLoss()
from lion_pytorch import Lion
learning_rate = 0.0005
#stock LION rate is 0.0001
optimizer = Lion(model.parameters(), lr=learning_rate)
#optimizer = Lion(model.parameters(), lr=learning_rate, weight_decay=1e-2)
#learning_rate = 0.00005
#optimizer = optim.Adam(model.parameters(), lr=learning_rate)
total_cycles = 0

do_training = 1 # transfer learn + generate predictions
#do_training = 0 # generate predictions on hidden set only

#for keeping track
pct_correct, losses = np.zeros((num_feat,1)), np.zeros((num_feat,1))

#checkpoint params for saving
# define checkpoints
checkpts = np.linspace(0, len(train_set), int(batch_size/0.5)).astype(int)
cp_train_loss, cp_train_accu = np.zeros((num_feat,1)), np.zeros((num_feat,1))
cp_test_loss, cp_test_accu = np.zeros((num_feat,1)), np.zeros((num_feat,1))
save_fname = 'swin_noBrdr_3std_'+str(num_feat)+'ft_'+str(num_y)+'QNT_datafile_'+str(int(train_df.shape[0]/1000))+'k_set_optim_'+str(optimizer)[:4]+'_lr_'+str(learning_rate)
save_dict = {}

#Error matrices only work if all equal parameter sizes**
err_on = 0
if (num_y/features_df[features_df.keys()[0]][0].item() == features_df.shape[1]):
	err_labs_mat  =  np.zeros((features_df.shape[1], features_df[features_df.keys()[0]][0].item(), 1))
	pred_labs_mat =  np.zeros((features_df.shape[1], features_df[features_df.keys()[0]][0].item(), 1))
	err_on = 1


last_check = 0 # previous checkpoint -- add to this if pretraining model

#if we want tot rian
if do_training == 1:
	# Train Network
	for epoch in range(num_epochs):
		#losses = []
		#epoch = epoch+2
		for batch_idx, (x, y) in tqdm(enumerate(train_loader)):

			x = x.to(device=device, dtype=torch.float)
			y = y.to(device=device, dtype=torch.float)

			# forward pass
			scores = model(x)
			all_losses = torch.zeros(1).cuda()
			#compute loss for each class independently
			for y_lab in np.arange(len(y_idx)-1):
				#compute accuracy
				feature_idxs = np.arange(y_idx[y_lab],y_idx[y_lab+1])
				y_hat = scores[:, feature_idxs].max(1)[1]	
				y_col = torch.argwhere(y[:, feature_idxs])[:, 1] # y_column

				num_correct = (y_hat == y_col).sum()
				pct_correct[y_lab, total_cycles] = np.round(num_correct.item()/batch_size*100, 2)

				print('\n Train acc. at batch ' + str(total_cycles) + ' ' + features_df.keys()[y_lab]  + ' : ' + str(pct_correct[y_lab, -1]) + '%')
				#compute loss
				loss = loss_fn(scores[:, feature_idxs], y[:, feature_idxs]) # loss on task
				losses[y_lab, total_cycles] = loss.item() #save the loss
				all_losses = torch.cat((all_losses, loss.expand(1))) # list of losses
			# sum the loss for multi-task learning
			cum_loss = torch.sum(all_losses[1:])
			#losses.append(cum_loss.item())
			
			print('\n Avg. train acc. at batch ' + str(total_cycles) + ' : ' + str(np.mean(pct_correct[:, -1])) + '%')

			pct_correct = np.hstack((pct_correct, np.zeros((num_feat, 1))))
			losses =  np.hstack((losses, np.zeros((num_feat, 1))))

			# backward pass
			optimizer.zero_grad()
			cum_loss.backward()

			# gradient descent
			optimizer.step()

			# save loss and accuracy at checkpoints
			if total_cycles == checkpts[0]:
				print("Loss at epoch " + str(epoch) + " is " + str(np.mean(losses[-100:])))

				for task in np.arange(len(y_idx)-1): 
					which_feat = features_df.keys()[task] #feature name
					save_dict[which_feat] = {} #generate nested dict

					feature_idxs = np.arange(y_idx[task],y_idx[task+1])
					out = validate(test_loader, model, loss_fn, feature_idxs)				
					print("Test set accuracy on feature: "+ features_df.keys()[task]   + " - " + str(out['val_accu']))

					#iterate one cycle for single feature
					cp_train_accu[task, -1] = np.mean(pct_correct[task, last_check:total_cycles+1])
					cp_test_accu[ task, -1] = out['val_accu']
					cp_train_loss[task, -1] = np.mean(losses[task, last_check:total_cycles+1])
					cp_test_loss[ task, -1] = out['val_loss']
					
					if err_on == 1:
						err_labs_mat[task, :, -1] = out['err_labs'].cpu().numpy().T
						pred_labs_mat[task, :,-1] = out['err_preds'].cpu().numpy().T

					#generate dictionary
					save_dict[which_feat]['train_accuracy'] = cp_train_accu[task, :]
					save_dict[which_feat]['test_accuracy']  = cp_test_accu[task, :]
					save_dict[which_feat]['train_loss']     = cp_train_loss[task, :]
					save_dict[which_feat]['test_loss']     = cp_test_loss[task, :]
					if err_on == 1:
						save_dict[which_feat]['err_labs']  = err_labs_mat[task, :, :]
						save_dict[which_feat]['pred_labs'] = pred_labs_mat[task, :, :]

				#save basic params				
				save_dict['train_size'] = train_df.shape
				save_dict['test_size'] = eval_df.shape
				save_dict['model_type'] = str(model)[:100]
				save_dict['features_df'] = features_df			
				#save the checkpoints
				with open(save_fname, 'wb') as f:
					save_loss_data = pickle.dump(save_dict, f)

				#advance checkpoint
				last_check = checkpts[0]
				checkpts = checkpts[1:]
				#update params
				cp_train_accu = np.hstack((cp_train_accu, np.zeros((num_feat, 1))))
				cp_test_accu  = np.hstack((cp_test_accu, np.zeros((num_feat, 1))))
				cp_train_loss = np.hstack((cp_train_loss, np.zeros((num_feat, 1))))
				cp_test_loss  = np.hstack((cp_test_loss, np.zeros((num_feat, 1))))
				if err_on ==1:
					err_labs_mat  = np.dstack((err_labs_mat, np.zeros((features_df.shape[1], features_df[features_df.keys()[0]][0].item(), 1))))
					pred_labs_mat = np.dstack((pred_labs_mat, np.zeros((features_df.shape[1], features_df[features_df.keys()[0]][0].item(), 1))))

			#advance cycles	
			total_cycles+=1

		# save model and check loss and accuracy
		#torch.save(model.state_dict(), '../final_project/saved_mod_'+str(batch_size)+'_e'+'_'+save_fname[:5])
		print('saved pytorch model w/ '+str(batch_size)+' batch;' + str(epoch) +'  epochs \n')
		print("Loss at epoch " + str(epoch) + " is " + str(sum(losses)/len(losses)))
