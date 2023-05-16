# The env. must have the hugging face environment activated
# Sample instructions on how to install and acitivate are below
## conda create -n huggingface python=3.9
## conda activate huggingface
## conda install pytorch torchvision torchaudio -c pytorch  #check pytorch installation for your system
## pip install transformers datasets evaluate
## pip install ipykernel
## pip install scikit-learn


import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import pipeline
import torch
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import evaluate
import os
import pdb
import pickle
import datetime

# generate the dataset
class SMILESDataset(Dataset):
	def __init__(self, encodings, labels):
		self.encodings = encodings
		self.labels = labels
	def __getitem__(self, index):

		y_labels = torch.tensor(self.labels.iloc[index])
		input_ids = self.encodings['input_ids'][index]
		attention_mask = self.encodings['attention_mask'][index]
		out = {'input_ids' : input_ids, 'attention_mask' : attention_mask, 'labels' : y_labels}
		return out

	def __len__(self):
		return len(self.labels)


def masked_loss(logits, labels, loss_fn):
	nclass = labels.shape[1]
	#loss = []
	losses = torch.empty(nclass)
	for col in np.arange(nclass):
		mask = labels[:, col] != 999
		col_loss = loss_fn(logits[mask, col], labels[mask, col])
		losses[col] = col_loss
	
	return losses.sum()


def validate(loader, model):
	pred_labels = np.array([]).astype(int)
	total_correct = 0
	total_samples = 0   
	# switch to eval  -> don't update weights while checking acc.
	model.eval()
	classifier = pipeline("text-classification", model=model, tokenizer = 'bert-base-cased', top_k=None)

	with torch.no_grad():

		for (input_ids, attn, labels) in tqdm(loader):
			out = classifier(input_ids)[0]
			y_hats = torch.tensor([out[which]['score'] for which in out])
			x = x.to(device=device,dtype=torch.float)
			y = y.to(device=device)
			scores = model(x)
			y_hat = scores.max(1)[1]
			total_correct += (y_hat == y).sum()
			total_samples += y_hat.size(0)
			pred_labels = np.append(pred_labels, y_hat.numpy())

		print(str(total_correct.item()) +'/' +str((total_samples)) + " classified correctly @ " + str(float(total_correct)/float(total_samples)*100) + "% accuracy.")
	#switch back to train mode -> continue updating weights to fine tune
	model.train()
	return pred_labels

def check_accuracy(logits, labels):
	threshold = 0.5
	logits_mask = logits>0 # if logit is positive -> greater than 50% of class 1
	pred_labels = logits
	pred_labels[logits_mask]  = 1
	pred_labels[~logits_mask] = 0

	pred_labels.int()
	total_correct = torch.sum(pred_labels == labels)
	total_possible = torch.sum(labels != 999)
	try:
		accuracy = np.round((total_correct/total_possible).item(),2)
	except:
		accuracy = float('nan')
	return accuracy


#Dataset

train_name = 'data/train.csv'
test_name = 'data/test.csv'
train_data = pd.read_csv(train_name)
test_data = pd.read_csv(test_name)

# pre-processing

longest_strs = train_data['StdSMILES'].str.len().nlargest(n=1000) # find the outliers
max_str_len = 184 #  = max. string length in the test set -> train model can't be shorter

short_train = train_data.drop(longest_strs.index, axis=0) # drop the outliers

#sub_idx = int(len(short_train)*.3) # to test a smaller model
sub_idx = int(len(short_train)) # test full model

short_train = short_train.iloc[:sub_idx] # use only some indices for training

all_train_labels = short_train.drop(['StdSMILES'], axis=1)

train_labs = all_train_labels #if training with all labels 
#train_labs = train_labels1 if training with only 1 label

eval_idx = int(short_train.shape[0]*.9) #establish EVAL set
short_eval = short_train.iloc[eval_idx:] #generate eval set
short_train = short_train.iloc[:eval_idx] #reduce train set

long_test = test_data['StdSMILES'].str.len()[test_data['StdSMILES'].str.len()>max_str_len]
short_test = test_data.drop(long_test.index, axis=0)
#tokenize the input
make_token = 'bert' # choose which tokenizer to use
if make_token == 'bert':
	tokenizer = AutoTokenizer.from_pretrained('bert-base-cased') # use BERT-NLP tokenizer
if make_token =='chemberta':
	tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1") # SMILES specific tokenizer

smiles_train_list = short_train['StdSMILES'].tolist() # for tokenizer
train_encodings = tokenizer(smiles_train_list, padding='max_length', truncation=True, max_length=max_str_len, return_tensors='pt')

smiles_eval_list = short_eval['StdSMILES'].tolist() # for tokenizer
eval_encodings = tokenizer(smiles_eval_list, padding='max_length', truncation=True, max_length=max_str_len, return_tensors='pt')

# Initialize train and eval dataset
train_DS = SMILESDataset(encodings = train_encodings, labels = train_labs.iloc[:eval_idx])
eval_DS =  SMILESDataset(encodings = eval_encodings, labels = train_labs[eval_idx:])


model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=4)
#model = AutoModelForSequenceClassification.from_pretrained("1e-05_token_bert/cycle_29990_loss_24.5", num_labels=4) #pre-trained by me
#model = AutoModelForSequenceClassification.from_pretrained("gpt2", num_labels=4) # gpt2
#from transformers import AutoModelWithLMHead
#model = AutoModelWithLMHead.from_pretrained("seyonec/ChemBERTa-zinc-base-v1", num_labels=4) # chemberta

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
print(device)

train = 1 # fine-tune the model if train == 1

if train == 1:

    # set hyperparamters
	batch_size = 48
	lr = 0.00001

	folder = str(lr) + '_token_' + make_token
	loss_fname = 'ldata_lr_' + str(lr) + 'token_' + make_token
	accu_fname = 'adata_lr_' + str(lr) + 'token_' + make_token
	train_loader = DataLoader(dataset=train_DS, batch_size =batch_size, shuffle = True)
	eval_loader = DataLoader(dataset=eval_DS, batch_size =batch_size, shuffle = True)

	optimizer = optim.Adam(model.parameters(), lr=lr)
	loss_fn = nn.CrossEntropyLoss()
	num_train_epochs = 20
	epc_ct = 0 
	total_cycles = 0
    epoch_loss, batch_accu, avg_loss_per_epc  = [], [], []    
    # set checkpoints to save the model
	checkpts = np.linspace(500,sub_idx,int(batch_size/1.5)).astype(int)


	for epoch in range(num_train_epochs):
		batch_num = 0
		epoch_loss = []
		model.train()
		for batch in tqdm(train_loader):
			optimizer.zero_grad()
			input_ids = batch['input_ids'].to(device)
			attention_mask = batch['attention_mask'].to(device)
			labels = batch['labels'].to(device)
			outputs = model(input_ids, attention_mask=attention_mask, labels = labels)

			loss = masked_loss(outputs[1], labels, loss_fn) #remove the NaN columns from loss calculation
			batch_accu.append(check_accuracy(outputs[1], labels).item())
			loss.backward()
			optimizer.step()
			print('Loss at batch '+ str(batch_num) +': '+ str(loss.item()))
			print('Pct. correct at batch '+ str(batch_num) +': '+ str(batch_accu[-1]*100))
			batch_num+=1
			total_cycles+=1
			epoch_loss.append(loss.item())
			validate(eval_loader, model)


			if total_cycles == checkpts[0]:
				fdir = folder+'/cycle_'+str(total_cycles)+'_loss_'+str(np.round(np.nanmean(epoch_loss[-500:]),1))
				model.save_pretrained(fdir)			
				checkpts = checkpts[1:]
				print('Saved MODEL @ --' + fdir)



		avg_loss_per_epc.append(np.nanmean(epoch_loss))
		avg_accu_per_epc.append(np.nanmean(batch_accu))

		print('**NEW EPOCH.** Avg. loss at previous epoch: ' + str(epc_ct) + ' is : ' + str(np.round(np.nanmean(epoch_loss[-1000:]),1)))
		print('**NEW EPOCH.** Avg. Accuracy at previous epoch: ' + str(epc_ct) + ' is : ' + str(np.nanmean(batch_accu[-1000:])*100))		
		epc_ct+=1
		print('Saved loss data in pwd as ' + loss_fname +' . At --- '+ str(datetime.datetime.now()))
		with open(loss_fname, 'wb') as f:
			save_loss_data = pickle.dump(avg_loss_per_epc, f)

		with open(accu_fname, 'wb') as f:
			save_accu_data = pickle.dump(avg_loss_per_epc, f)			

		validate(eval_loader, model)


#generate predictions
predict = 0 #if 1, generates predictions in CSV format for the hidden set, and generates Loss plot
if predict == 1:
	
	mnames = ['1e-05_token_bert/cycle_5012_loss_24.7', '1e-05_token_bert/cycle_1628_loss_23.3', '1e-05_token_chemberta/cycle_10657_loss_25.0',
			  '1e-05_token_bert/cycle_800_loss_22.8', '1e-05_token_bert/cycle_10924_loss_25.2', '1e-05_token_bert/cycle_21049_loss_23.1']

	#mnames = ['1e-05_token_bert/cycle_21049_loss_23.1']

	test_data = pd.read_csv('data/random_submission.csv')

	for model_name in mnames:
		classifier = pipeline("text-classification", model=os.getcwd()+'/'+model_name, tokenizer = 'bert-base-cased', top_k=None)

		threshold = 0.5
		labels = np.zeros((len(test_data), 4))
		for smile in np.arange(len(test_data)):
			out = classifier(test_data.iloc[smile]['StdSMILES'])[0]
			for protein in np.arange(len(out)):
				score = out[protein]['score']
				if score>threshold:
					score =1
				else:
					score = 0
				labels[smile, protein] = score

		df_submission = pd.DataFrame(labels, columns = test_data.keys().tolist()[1:])
		df_submission.insert(0, 'StdSMILES', test_data['StdSMILES'])
		df_submission.to_csv('submission_' +model_name[-9:]+'.csv', index=False)


# generate loss plots
loss_plot = 0
if loss_plot == 1: # loads and generates loss plot
	import pickle
	import numpy as np
	import matplotlib.pyplot as plt


	with open('ldata_lr_1e-05token_bert', 'rb') as f:
		 bert_loss = np.array(pickle.load(f))


	with open('ldata_lr_1e-05token_chemberta', 'rb') as f:
		 chem_loss = np.array(pickle.load(f))

	fig, ax1 = plt.subplots(nrows = 1, ncols =1, sharex = False, sharey = False, constrained_layout=True)
	ax1.plot(bert_loss, label='ChemBERTa_loss')
	ax1.plot(chem_loss, label='BERT_loss')
	ax1.set_xlabel('# training epochs')
	ax1.set_ylabel('Sum Cross-Entropy Loss')
	ax1.legend()
