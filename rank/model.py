import torch
from torch import nn
import torch.utils.checkpoint
from typing import List, Optional, Tuple, Union
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel
from pytorchltr.loss import LambdaNDCGLoss1, PairwiseLogisticLoss


class BertForNDCG(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.dense = nn.Linear(config.hidden_size, 1)
        self.post_init()

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            num=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.dense(pooled_output)

        if num is None:
            return logits

        loss_fct = LambdaNDCGLoss1()

        score = logits.view(-1, num)
        batch = score.shape[0]

        label = torch.arange(5, 0, -1).squeeze(0).repeat(batch, 1).to(logits.device)
        n = torch.Tensor([num] * batch).to(logits.device)
        loss = loss_fct(score, label, n)

        loss = torch.mean(loss)
        return loss, score
