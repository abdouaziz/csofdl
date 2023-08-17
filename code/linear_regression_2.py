#import packages
import torch
import torch.nn as nn 
from torch.utils.data import Dataset , DataLoader
import torch.optim as optim
from sklearn.datasets import load_wine

import wandb    #Monitoring 

from tqdm import tqdm



wandb.init()
#configuration 

config = {
    "num_epochs":32 ,
    "input_dim":13,
    "num_classes":3,
    "learning_rate":0.001,
    "batch_size":12
}

#Create dataset object

class WineDataset(Dataset):
    def __init__(self):
        self.df = load_wine()
        self.datas = torch.from_numpy(self.df.data)
        self.targets = torch.from_numpy(self.df.target)

    def __len__(self):
        return self.datas.shape[0]
    
    def __getitem__(self, index):
        data = self.datas[index]
        target = self.targets[index]

        return {
            "input" : data.to(torch.float) ,
            "target" : target
        }

# dataloader 

def dataloader(dataset , batch_size , shuffle):
    return DataLoader(dataset=dataset ,batch_size= batch_size , shuffle=shuffle)

#Create Model
class WineRegressionLinear(nn.Module):
    def __init__(self , input_dim ,num_classes):
        super().__init__()
        self.linear_1 = nn.Linear(input_dim, num_classes)
        self.relu =nn.ReLU()
        self.drop = nn.Dropout()
        self.linear_2 =nn.Linear(num_classes , num_classes)


    def forward(self , input):
        x = self.linear_1(input)
        x=self.relu(x)
        x= self.drop(x)
        x = self.linear_2(x)
        return x 


#Training step 
def train_step(model , dataloader , optimizer , loss_fn):
    model.train()

    for step, data in tqdm(enumerate(dataloader) , total=len(dataloader)):
        inputs = data['input']
        targets = data['target']

        output = model(inputs)

        loss =loss_fn(output , targets)

        #backward 
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step%10==0:
            print(f"Loss Training : {loss}")
            wandb.log({
                "loss":loss
            })


# Test Function
def validation_step():
    pass

#main 

def main():
    
    #dataset
    mydataset = WineDataset()
    # train_dataset , test_dataset split your dataset
    # model
    model = WineRegressionLinear(input_dim=config["input_dim"] , num_classes=config["num_classes"])
    #trainloader 
    train_loader = dataloader(dataset=mydataset , batch_size=config["batch_size"] , shuffle=True)

    optimizer = optim.SGD(model.parameters() , lr=config["learning_rate"])

    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(config["num_epochs"]):
        train_step(model , train_loader, optimizer , loss_fn)
        # validation_step


if __name__=="__main__":
    main()

    
