import numpy as np
import torch
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class SentenceEmbedding:
    def __init__(self):
        model_path = 'E:\\ptm\\simbert'
        self.model = BertModel.from_pretrained(model_path)
        self.model.to(device)
        self.model.eval()
        self.tokenizer = BertTokenizer.from_pretrained(model_path)

    def encode(self, content, batch_size=256, max_length=None, padding='max_length'):
        outputs = None
        if isinstance(content, list) and len(content) > batch_size:
            for epoch in tqdm(range(len(content) // batch_size + 1)):
                batch_content = content[epoch * batch_size:(epoch + 1) * batch_size]
                if batch_content:
                    output = self._embedding(batch_content, max_length, padding)
                    if outputs is None:
                        outputs = output
                    else:
                        outputs = np.concatenate([outputs, output], axis=0)
            return outputs
        else:
            return self._embedding(content, max_length, padding)

    def _embedding(self, content, max_length, padding):

        if max_length is None:
            if isinstance(content, str):
                max_length = len(content) + 2
            else:
                max_length = max([len(c) for c in content]) + 2
        max_length = min(max_length, 512)
        inputs = self.tokenizer(content,
                                return_tensors="pt",
                                truncation=True,
                                padding=padding,
                                max_length=max_length)
        with torch.no_grad():
            outputs = self.model(**inputs.to(device))
            output = outputs[1].cpu().numpy()

        return output
