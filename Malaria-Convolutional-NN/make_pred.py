import torch
import torch.nn as nn  
import torch.optim as optim  
import torchvision.transforms as transforms  
import torchvision
import os
import pdb
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage import io
from skimage.transform import rescale, resize, downscale_local_mean
from torch.utils.data import Dataset, DataLoader


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Parameters
learning_rate = 0.001
batch_size = 64
num_epochs = 5
train_dir='train/train/'
train_csv='train_labels.csv'
test_dir='test/test/'
test_csv='test_labels.csv'


class DS(Dataset):
    def __init__(self,csv_file,img_dir,tranform=None):
        self.label=pd.read_csv(csv_file)
        self.img_dir=img_dir
        self.tranform=tranform
        
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self,index):
        img_path=os.path.join(self.img_dir,self.label.iloc[index,0])
        image=io.imread(img_path)
        image=resize(image,(250,250),anti_aliasing=True)
        y_label=torch.tensor(int(self.label.iloc[index,1]))
        self.fnames = img_path

        if self.tranform:
            image=self.tranform(image)
        return (image,y_label)


def validate(loader, model):
    pred_labels = np.array([]).astype(int)
    total_correct = 0
    total_samples = 0    
    #swithc to eval mode -> don't update weights while checking acc.
    model.eval()

    with torch.no_grad():
        for x, y in tqdm(loader):
            x = x.to(device=device,dtype=torch.float)
            y = y.to(device=device)
            scores = model(x)
            y_hat = scores.max(1)[1]
            total_correct += (y_hat == y).sum()
            total_samples += y_hat.size(0)
            pred_labels = np.append(pred_labels, y_hat.numpy())

        print(str(total_correct.item()) +'/' +str((total_samples)) + " classified correctly @ " + str(float(total_correct)/float(total_samples)*100) + "% accuracy.")
    #switch back to train mode -> continue updating weights during transfer learning
    model.train()
    return pred_labels

#generate the loaders
train_set = DS(csv_file=train_csv, img_dir=train_dir, tranform=transforms.ToTensor())
test_set = DS(csv_file=test_csv, img_dir=test_dir, tranform=transforms.ToTensor())
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

# Model
model = torchvision.models.googlenet(pretrained=True)
model.fc=nn.Sequential(
    nn.Linear(in_features=1024,out_features=512),
    nn.ReLU(),
    nn.Linear(in_features=512,out_features=128),
    nn.ReLU(),
    nn.Linear(in_features=128,out_features=32),
    nn.ReLU(),
    nn.Linear(in_features=32,out_features=2,bias=True)
) 
print(device)
model.to(device);
model

# model params
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#do_training = 1 # transfer learn + generate predictions
do_training = 0 # generate predictions on hidden set only

if do_training ==1:
	# Train Network
	for epoch in range(num_epochs):
	    losses = []

	    for batch_idx, (x, y) in tqdm(enumerate(train_loader)):

	        x = x.to(device=device,dtype=torch.float)
	        y = y.to(device=device)

	        # forward pass
	        scores = model(x)
        pdb.set_trace()
	        loss = loss_fn(scores, y)

	        losses.append(loss.item())

	        # backward pass
	        optimizer.zero_grad()
	        loss.backward()

	        # gradient descent
	        optimizer.step()

	    # save model and check loss and accuracy
	    torch.save(model.state_dict(), '../HW2/saved_mod_'+str(batch_size)+'_e'+str(epoch))
	    print('saved pytorch model w/ '+str(batch_size)+' batch;' + str(epoch) +'  epochs \n')
	    print("Loss at epoch " + str(epoch) + " is " + str(sum(losses)/len(losses)))
	    print("Train set accuracy...")
	    validate(train_loader, model)
	    print("Test set accuracy...")    
	    validate(test_loader, model)    


pred_dir='hidden_test/hidden_test'
all_mods = ['saved_mod_0128_e1', 
		'saved_mod_0128_e3', 
		'saved_mod_0128_e5', 
		'saved_mod_0064_e0', 
		'saved_mod_1024_e2',
		'saved_mod_1024_e3',
		'saved_mod_1024_e4',
		'saved_mod_1024_e5',
		'saved_mod_1024_e6',
		]

for saved_mod in all_mods:

    # Load model
    model.load_state_dict(torch.load(saved_mod))
    print('Successfully loaded model: ' + saved_mod)

    pred_dir='hidden_test/hidden_test/'
    pred_csv='sample_submission.csv'

    pred_set=DS(csv_file=pred_csv, img_dir=pred_dir, tranform=transforms.ToTensor())
    pred_loader = DataLoader(dataset=pred_set, batch_size=batch_size, shuffle=False)

    y_hats = validate(pred_loader, model)
    pd_sub = pd.read_csv(pred_csv)
    pd_sub['label'] = y_hats
    mod_id = saved_mod[-7:]
    pd_sub.to_csv('submission_' +mod_id+'.csv', index=False)
