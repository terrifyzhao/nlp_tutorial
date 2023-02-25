import random
import torch
import numpy as np
from utils import get_device, one_hot


class MixUp:
    """
    1、正常训练，bp
    2、mixup训练，bp
    3、梯度累加，更新权重
    """

    def __init__(self, model, tokenizer, num_labels, layer='embedding'):
        self.tokenizer = tokenizer
        self.model = model
        self.device = get_device()
        self.num_labels = num_labels
        self.layer = layer

    def cross_entropy(self, logits, label):
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(torch.sum(exp_logits, dim=1, keepdim=True))
        return -torch.mean(torch.sum(log_prob * label, dim=1))

    def augmentation(self, data):
        # 这里的data是一个batch的数据
        input_ids = data['input_ids'].to(self.device)
        attention_mask = data['attention_mask'].to(self.device)
        label = data['labels'].to(self.device).long()

        # 把batch内的数据打乱 4
        batch_size = len(data)
        # [1, 3, 2, 0]
        index = torch.randperm(batch_size).to(self.device)
        lam = np.random.beta(0.5, 0.5)

        label_mix = one_hot(label, self.num_labels) * lam + one_hot(label[index], self.num_labels) * (1 - lam)

        def my_hook(module, inputs, outputs):
            x_mix = outputs * lam + outputs[index] * (1 - lam)
            return x_mix

        # pytorch的钩子
        hook = self.model.bert.embeddings.register_forward_hook(my_hook)

        outputs = self.model(input_ids, attention_mask)
        logits = outputs.logits
        hook.remove()

        loss = self.cross_entropy(logits, label_mix)
        return loss
