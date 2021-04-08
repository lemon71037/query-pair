from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm


def str_to_int_list(s):
    return [int(x) for x in s.split()]


def str_list_to_int_list(slist):
    return [str_to_int_list(s) for s in slist]


def batch_padding(q, q_len):
    max_len = np.max(q_len)
    for i in range(len(q)):
        q[i] = q[i] + [0] * (max_len - q_len[i])
    return np.array(q)


def my_collate(batch):
    tok_ids = [item[0] for item in batch]
    seg_ids = [item[1] for item in batch]
    out_ids = [item[2] for item in batch]
    labels = np.array([item[3] for item in batch])

    tok_len = np.array([len(item[0]) for item in batch])

    tok_ids = torch.from_numpy(batch_padding(tok_ids, tok_len)).long()
    seg_ids = torch.from_numpy(batch_padding(seg_ids, tok_len)).long()
    out_ids = torch.from_numpy(batch_padding(out_ids, tok_len)).long()
    labels = torch.from_numpy(labels).float()

    return [tok_ids, seg_ids, out_ids, labels]


class OppoQuerySet(Dataset):
    def __init__(self, df, dataset='train'):

        q1 = str_list_to_int_list(df['q1'].values)
        q2 = str_list_to_int_list(df['q2'].values)
        self.dataset = dataset

        if dataset is not 'test':
            a = df['label'].values
            self.data = zip(q1, q2, a)
        else:
            self.data = zip(q1, q2, [-5] * len(q1))

        self.data = list(self.data)
        self.processed_data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if len(self.processed_data) == 0:
            q1, q2 = self.data[index][:2]
            q1_len = len(q1)
            q2_len = len(q2)
            return q1, q2, q1_len, q2_len, self.data[index][2]
        else:
            return self.processed_data[index]

    def random_mask(self, text_ids, tokens):
        """随机mask
        """
        input_ids, output_ids = [], []
        rands = np.random.random(len(text_ids))
        for r, i in zip(rands, text_ids):
            if r < 0.15 * 0.8:
                input_ids.append(103)
                output_ids.append(i)
            elif r < 0.15 * 0.9:
                input_ids.append(i)
                output_ids.append(i)
            elif r < 0.15:
                input_ids.append(np.random.choice(len(tokens)))
                output_ids.append(i)
            else:
                input_ids.append(i)
                output_ids.append(0)
        return input_ids, output_ids

    def sample_convert(self, text1, text2, label, random=False, tokens=None):
        """
        转换为MLM格式
        output_id = 0 即为非预测部分
        """
        text1_ids = [tokens.get(t, 100) for t in text1]
        text2_ids = [tokens.get(t, 100) for t in text2]
        if random:
            if np.random.random() < 0.5:
                text1_ids, text2_ids = text2_ids, text1_ids
            text1_ids, out1_ids = self.random_mask(text1_ids, tokens)
            text2_ids, out2_ids = self.random_mask(text2_ids, tokens)
        else:
            out1_ids = [0] * len(text1_ids)
            out2_ids = [0] * len(text2_ids)
        token_ids = [101] + text1_ids + [102] + text2_ids + [102]
        segment_ids = [0] * len(token_ids)
        output_ids = [label + 5] + out1_ids + [0] + out2_ids + [0]
        return token_ids, segment_ids, output_ids

    def process_data(self, dictionary, random=False):
        self.processed_data = []
        for i in tqdm(range(len(self.data)), desc="Process Data: ", ncols=100, total=len(self.data)):
            q1, q2, label = self.data[i]
            token_ids, segment_ids, output_ids = \
                self.sample_convert(q1, q2, label, random, tokens=dictionary.tok_to_bert_id)
            self.processed_data.append((token_ids, segment_ids, output_ids, label))


if __name__ == '__main__':
    df_train = pd.read_table("./baseline_tfidf_lr/oppo_breeno_round1_data/gaiic_track3_round1_train_20210228.tsv",
                             names=['q1', 'q2', 'label']).fillna("0")
    df_test = pd.read_table('./baseline_tfidf_lr/oppo_breeno_round1_data/gaiic_track3_round1_testA_20210228.tsv',
                            names=['q1', 'q2']).fillna("0")

    max_value = -1
    min_value = 999999
    max_len = -1

    trainset = OppoQuerySet(df_train, dataset='train')
    trainloader = DataLoader(trainset, batch_size=1, shuffle=False, collate_fn=my_collate)

    count = 0
    for q1, q2, a in trainloader:
        print(q1)
        print(q2)
        print(a)
        count = count + 1
        if count >= 10:
            break
