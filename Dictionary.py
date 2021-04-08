from pytorch_pretrained_bert import BertTokenizer
import json
import numpy as np

#######
# BERT SEPICIAL TOKEN:
#   UNK     100
#   CLS     101
#   SEP     102
#   MASK    103


class Dictionary():
    def __init__(self, trainset, valset, testset):
        min_count = 5

        # 统计词频
        tokens = {}
        for dset in [trainset, valset, testset]:
            for i in range(len(dset)):
                q1, q2 = dset[i][0], dset[i][1]
                for idx in q1 + q2:
                    tokens[idx] = tokens.get(idx, 0) + 1

        tokens = {i: j for i, j in tokens.items() if j >= min_count}  # tok: counts
        tokens = sorted(tokens.items(), key=lambda s: -s[1])  # tok: counts(sorted)
        tokens = {
            t[0]: i         # tok: rank_id
            for i, t in enumerate(tokens)
        }
        self.tokens = tokens       # tok, rank_id

    def aliment_bert_id(self, tokenizer: BertTokenizer):
        # BERT词频
        counts = json.load(open('counts.json'))
        del counts['[CLS]']
        del counts['[SEP]']

        token_dict = tokenizer.ids_to_tokens
        freqs = [
            counts.get(tok, 0) for id, tok in sorted(token_dict.items(), key=lambda s: s[0])
        ]
        keep_tokens = list(np.argsort(freqs)[::-1])  # sorted id

        self.tok_to_bert_id = {}
        for tok, id in self.tokens.items():
            if id >= len(keep_tokens):
                bert_id = 100
            else:
                bert_id = keep_tokens[id]
            self.tok_to_bert_id[tok] = bert_id

        # 0: pad, 100: unk, 101: cls, 102: sep, 103: mask
        # 5: no
        # 6: yes

    @property
    def bert_pad_idx(self):
        return 0

    @property
    def bert_unk_idx(self):
        return 100

    @property
    def bert_cls_idx(self):
        return 101

    @property
    def bert_sep_idx(self):
        return 102

    @property
    def bert_mask_idx(self):
        return 103
