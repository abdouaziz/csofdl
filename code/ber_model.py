import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, BertModel
from transformers.optimization import AdamW
from transformers import BertPreTrainedModel
from transformers import AutoConfig


import pandas as pd
from sklearn.model_selection import train_test_split

from tqdm import tqdm

from sklearn.model_selection import train_test_split

import wandb  # monitoring

config = {
    "model_name": "bert-base-uncased",
    "max_length": 80,
    "hidden_state": 768,
    "csv_file": "data.csv",
    "batch_size": 2,
    "learing_rate": 2e-5,
    "n_epochs": 4,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}


class MyDataset(Dataset):
    def __init__(self, csv_file, tokenizer_name, max_length):
        self.df = pd.read_csv(csv_file)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        text = self.df["text"][index]
        label = self.df["label"][index]

        inputs = self.tokenizer(
            text=text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "label": torch.tensor(label),
        }


def dataloader(dataset, batch_size, shuffle):
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)


class CustomModel(nn.Module):
    def __init__(self, model_name, n_classes):
        super(CustomModel, self).__init__()
        self.pretrained_model = BertModel.from_pretrained(
            model_name
        )  # hidden_state 786 Bert_base
        self.classifier = nn.Linear(768, n_classes)  # MLP

    def forward(self, input_ids, attention_mask):

        output = self.pretrained_model(
            input_ids=input_ids, attention_mask=attention_mask
        )  # (batch, 768)

        output = self.classifier(output.last_hidden_state)

        return output


def train_step(model, train_loader, optimizer, loss_fn, device):
    model.train()

    total_loss = 0

    for data in tqdm(train_loader, total=len(train_loader)):

        input_ids = data["input_ids"].squeeze(1).to(device)
        attention_mask = data["attention_mask"].to(device)
        label = data["label"].to(device)

        optimizer.zero_grad()

        output = model(input_ids=input_ids, attention_mask=attention_mask)

        loss = loss_fn(
            output, label.unsqueeze(1)
        )  # tensor[1.3] ==1 tensor[1.3].item()=1.3

        loss.backward()
        optimizer.step()

        total_loss += loss.item()  # total_loss = total_loss + loss

    return total_loss / len(train_loader)


def validation_step(model, validation_loader, loss_fn, device):

    total_loss = 0
    correct_prediction = 0

    with torch.no_grad():
        for data in tqdm(train_loader, total=len(train_loader)):
            input_ids = data["input_ids"].squeeze(1).to(device)
            attention_mask = data["attention_mask"].to(device)
            label = data["label"].to(device)

            output = model(input_ids=input_ids, attention_mask=attention_mask)

            loss = loss_fn(
                output, label.unsqueeze(1)
            )  # tensor[1.3] ==1 tensor[1.3].item()=1.3
            total_loss += loss.item()  # total_loss = total_loss + loss

            pred = torch.max(torch.softmax(output, dim=1), dim=1)
            correct_prediction += torch.sum(pred.indices == label)

    return total_loss / len(validation_loader), 100 * correction_prediction / len(
        validation_loader
    )




def main():

    wandb.init(project="bert-classification")

    dataset =MyDataset(csv_file=config['csv_file'] ,tokenizer_name=config['model_name'] , max_length=config['max_length'] )

    train_dataset , validation_dataset = train_test_split(dataset , test_size =0.2)

    train_loader = dataloader(train_dataset , batch_size=config['batch_size'], shuffle=True)
    validation_loader = dataloader(validation_dataset, batch_size=config['batch_size'] , shuffle=False)


    model= CustomModel(model_name=config['model_name'] , n_classes=1)
    model.to(config['device'])

    loss_fn = nn.CrossEntropyLoss()

    optimizer = AdamW(model.parameters() , lr =config['learing_rate'])

    for epoch in range(config['n_epochs']):

        loss_train =train_step(model , train_loader , optimizer , loss_fn ,config['device'])
        loss_validation , accuracy = validation_step(model , validation_loader , loss_fn , config['device'])

        wandb.log(
            "loss_train":loss_train,
            "loss_validation":loss_validation,
            "accuracy":accuracy
        )


    # sauvegarder 
    torch.save(model , 'bert-model.pth')


if __name__ == "__main__":
    main()
