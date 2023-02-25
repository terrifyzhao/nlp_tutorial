from utils import get_device
import torch
from transformers import BertTokenizer
from model import BertForNDCG
import numpy as np

device = get_device()
model_path = 'E:\\ptm\\roberta'

tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForNDCG.from_pretrained(model_path)
model.load_state_dict(torch.load('best_model.bin', map_location=device))
model.to(device)
model = model.eval()


def inference(text1, text2):
    encoding = tokenizer([text1] * len(text2),
                         text2,
                         max_length=128,
                         truncation=True,
                         padding=True,
                         return_tensors='pt')

    with torch.no_grad():
        res = model(**encoding.to(device))
        logits = res.cpu().numpy().flatten()
    return logits


def rank(query, docunment):
    res = inference(query, docunment)
    index = np.argsort(-res)[:5]
    print(np.array(docunment)[index])
    return index

