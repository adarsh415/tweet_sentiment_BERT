import config
import transformers
import torch.nn as nn

class BERTBaseUncased(nn.Module):
    def __init__(self):
        super(BERTBaseUncased, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH)
        self.l0 = nn.Linear(768, 2)

    def forward(self, ids, mask, token_type_ids):

        sequence_output, pooled_output = self.bert(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )
        # sequence_output looks like (batch_size, num_tokens, 768)
        logits = self.l0(sequence_output)
        # we want (batch_size, num_tokens, 2)
        # we will split init  # we want (batch_size, num_tokens, 1),  # we want (batch_size, num_tokens, 1)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits =  start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        # now its become (batch_size, num_tokens), (batch_size, num_tokens)

        return start_logits, end_logits