from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM
import torch
import torch.nn.functional as F

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertForMaskedLM.from_pretrained('bert-base-chinese')

s = "[CLS]你好啊[SEP]"
tokens = tokenizer.tokenize(s)
print(" ".join(tokens))
segments_ids = torch.tensor([[1, 1, 1, 1, 1]])
labels = torch.tensor([[22, 23, 23, 23, 0]])
ids = torch.tensor([tokenizer.convert_tokens_to_ids(tokens)])
print(ids.shape)

loss = model(ids, segments_ids, masked_lm_labels=labels)
vocab_size = 21128
result = model(ids, segments_ids)
print(result.size())
myloss = F.cross_entropy(result.view(-1, 21128), labels.view(-1), ignore_index=0)

xloss = 0
for i in range(4):
    xloss += F.cross_entropy(result[0][i].unsqueeze(0), labels[0][i].unsqueeze(0))

print(loss, myloss)
print(xloss / 4)

x = torch.randn((64, 12, 200))
y = x[:, 0, 5:7]
print(y.size())
