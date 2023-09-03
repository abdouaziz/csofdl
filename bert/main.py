import torch
from transformers import AutoTokenizer
from utils import CustomModel
from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()


classes = ["Positive", "Negative"]

model_loaded = CustomModel.from_pretrained("abdouaziiz/bert_model")
tokenizer_loaded = AutoTokenizer.from_pretrained("abdouaziiz/bert_model")


class Item(BaseModel):
    text: str


def predict(text: Item):
    with torch.no_grad():
        inputs = tokenizer_loaded(text.text, return_tensors="pt")
        output = model_loaded(inputs["input_ids"], inputs["attention_mask"])
        pred = torch.max(output, dim=1)

    return {"indice": pred.indices.item(), "classe": classes[pred.indices.item()]}


@app.post("/predict")
def get_prediction(text: Item):
    return predict(text)
