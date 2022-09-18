import random
import torch
import numpy as np
from utils import get_device, one_hot


class MixUp:

    def __init__(self, model, tokenizer, num_labels, layer='embedding'):
        self.tokenizer = tokenizer
        self.model = model
        self.device = get_device()
        self.num_labels = num_labels
        self.layer = layer

    def cross_entropy(self, logits, labels):
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(torch.sum(exp_logits, dim=1, keepdim=True))
        return -torch.mean(torch.sum(log_prob * labels, dim=1))

    def augmentation(self, data):
        input_ids = data['input_ids'].to(self.device)
        attention_mask = data['attention_mask'].to(self.device)
        label = data['labels'].to(self.device).long()

        batch_size = len(input_ids)
        # 打乱顺序用于计算mix embedding
        index = torch.randperm(batch_size).to(self.device)
        lam = np.random.beta(0.5, 0.5)

        label_mix = one_hot(label, self.num_labels) * lam + one_hot(label[index], self.num_labels) * (1 - lam)
        hook = None

        def single_forward_hook(module, inputs, outputs):
            mix_input = outputs * lam + outputs[index] * (1 - lam)
            return mix_input

        def multi_forward_hook(module, inputs, outputs):
            mix_input = outputs[0] * lam + outputs[0][index] * (1 - lam)
            return tuple([mix_input])

        if self.layer == 'embedding':
            hook = self.model.bert.embeddings.register_forward_hook(single_forward_hook)
        elif self.layer == 'pooler':
            hook = self.model.bert.pooler.register_forward_hook(single_forward_hook)
        elif self.layer == 'inner':
            # 随机选一层
            layer_num = random.randint(1, self.model.config.num_hidden_layers) - 1
            hook = self.model.bert.encoder.layer[layer_num].register_forward_hook(multi_forward_hook)

        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask,
                             labels=label.to(self.device))
        logits = outputs.logits
        hook.remove()

        # 计算loss
        loss = self.cross_entropy(logits, label_mix)
        return loss
