"""
通用模型，包括各种embedding，各种head，
定义了输入IEModelConfig和输出GPOutput
定义了loss：loss_for_ie
定义了模型架构IEModel

"""

import torch
from transformers import PreTrainedModel
from typing import Optional

from algorithms_ai.deep_learning.information_extraction.modeling_ie_configuration import IEModelConfig
from algorithms_ai.deep_learning.information_extraction.modeling_ie_head import GlobalPointerHead, EfficientGlobalPointerHead
from algorithms_ai.deep_learning.information_extraction.modeling_ie_loss import loss_for_ie
from algorithms_ai.deep_learning.information_extraction.modeling_ie_output import GPOutput


class IEModel(PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    config_class = IEModelConfig

    def __init__(self, config: IEModelConfig):
        super(IEModel, self).__init__(config)
        self.entity2id = config.entity2id
        self.relation2id = config.relation2id

        self.head_size = config.head_size
        self.model_head = config.model_head

        self.encoder_type = config.encoder_type
        self.encoder_path = config.encoder_path
        self.encoder, self.encoder_hidden_size = self.get_encoder(config.encoder_type, config.encoder_path,
                                                                  config.encoder_config)
        config.encoder_hidden_size = self.encoder_hidden_size
        config.encoder_config = self.encoder.config

        self.build_model()

    def get_encoder(self, encoder_type, encoder_path, encoder_config):
        if encoder_type == 'bert':
            from transformers import BertModel
            encoder = BertModel.from_pretrained(encoder_path)
            encoder_hidden_size = encoder.config.hidden_size
        else:
            from transformers import BertModel
            encoder = BertModel.from_pretrained(encoder_path)
            encoder_hidden_size = encoder.config.hidden_size
        return encoder, encoder_hidden_size

    def build_model(self):
        if self.entity2id:
            if self.model_head == 'GlobalPointerHead':
                self.entity_detect = GlobalPointerHead(heads=len(self.entity2id),
                                                       head_size=self.head_size,
                                                       hidden_size=self.encoder_hidden_size,
                                                       RoPE=True, tril_mask=True)
            else:
                self.entity_detect = EfficientGlobalPointerHead(heads=len(self.entity2id),
                                                                head_size=self.head_size,
                                                                hidden_size=self.encoder_hidden_size,
                                                                RoPE=True, tril_mask=True)

        if self.relation2id:
            if self.model_head == 'GlobalPointerHead':
                self.relation_detect = GlobalPointerHead(heads=2,
                                                         head_size=self.head_size,
                                                         hidden_size=self.encoder_hidden_size,
                                                         RoPE=True, tril_mask=True)
                self.s_o_head = GlobalPointerHead(heads=len(self.relation2id),
                                                  head_size=self.head_size,
                                                  hidden_size=self.encoder_hidden_size,
                                                  RoPE=False, tril_mask=False)

                self.s_o_tail = GlobalPointerHead(heads=len(self.relation2id),
                                                  head_size=self.head_size,
                                                  hidden_size=self.encoder_hidden_size,
                                                  RoPE=False, tril_mask=False)
            else:
                self.relation_detect = EfficientGlobalPointerHead(heads=2,
                                                                  head_size=self.head_size,
                                                                  hidden_size=self.encoder_hidden_size,
                                                                  RoPE=True, tril_mask=True)
                self.s_o_head = EfficientGlobalPointerHead(heads=len(self.relation2id),
                                                           head_size=self.head_size,
                                                           hidden_size=self.encoder_hidden_size,
                                                           RoPE=False, tril_mask=False)
                self.s_o_tail = EfficientGlobalPointerHead(heads=len(self.relation2id),
                                                           head_size=self.head_size,
                                                           hidden_size=self.encoder_hidden_size,
                                                           RoPE=False, tril_mask=False)

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs
    ):
        outputs = self.encoder(
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

        last_hidden_state = outputs[0]
        if self.entity2id:
            entities_outputs = self.entity_detect(last_hidden_state, attention_mask=attention_mask)
        else:
            entities_outputs = torch.tensor(0, device=last_hidden_state.device)

        if self.relation2id:
            relations_outputs = self.relation_detect(last_hidden_state, attention_mask=attention_mask)
            so_head_outputs = self.s_o_head(last_hidden_state, attention_mask=attention_mask)
            so_tail_outputs = self.s_o_tail(last_hidden_state, attention_mask=attention_mask)
        else:
            relations_outputs, so_head_outputs, so_tail_outputs = torch.tensor(0,
                                                                               device=last_hidden_state.device), torch.tensor(
                0, device=last_hidden_state.device), torch.tensor(0, device=last_hidden_state.device)

        loss = None
        losses = dict()
        if labels is not None:
            if self.entity2id:
                entity_loss = loss_for_ie(y_true=labels[0], y_pred=entities_outputs, mask_zero=True)
                losses['entity_loss'] = entity_loss
            if self.relation2id:
                relation_loss = loss_for_ie(y_true=labels[1], y_pred=relations_outputs, mask_zero=True)
                relation_head_loss = loss_for_ie(y_true=labels[2], y_pred=so_head_outputs, mask_zero=True)
                relation_tail_loss = loss_for_ie(y_true=labels[3], y_pred=so_tail_outputs, mask_zero=True)
                losses['relation_loss'] = relation_loss
                losses['relation_head_loss'] = relation_head_loss
                losses['relation_tail_loss'] = relation_tail_loss

            loss = sum(losses.values())

        return GPOutput(
            loss=loss,
            entity_outputs=entities_outputs,
            relations_outputs=relations_outputs,
            so_head_outputs=so_head_outputs,
            so_tail_outputs=so_tail_outputs,
            input_ids_mask=kwargs['input_ids_mask'],
            # logits=[entities_outputs, relations_outputs, so_head_outputs, so_tail_outputs],
            # hidden_states=last_hidden_state,
            # attentions=outputs.attentions,
        )


if __name__ == '__main__':
    pass