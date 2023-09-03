import torch 
import torch.nn as nn 

from huggingface_hub import PyTorchModelHubMixin
from transformers import BertModel
from transformers import AutoTokenizer


class CustomModel(nn.Module ,PyTorchModelHubMixin ):   
    def __init__(self):
        super(CustomModel , self).__init__()
        self.pretrained_model = BertModel.from_pretrained("bert-base-uncased")   # hidden_state 786 Bert_base
        self.classifier = nn.Linear(768 , 1)   # MLP

    def forward(self , input_ids , attention_mask ):

        output =self.pretrained_model(input_ids=input_ids , attention_mask=attention_mask)  #(batch, 768)

        output = self.classifier(output.last_hidden_state)

        return output 




 