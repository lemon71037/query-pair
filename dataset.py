from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch
import Dictionary
from tqdm import tqdm
from sklearn.utils import shuffle
from pytorch_pretrained_bert import BertTokenizer

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
    """ batch: (q1_idx, q2_idx, tok1_idx, tok2_idx, out1_ids, out2_ids, label)
    """
    new_batch = []
    for i in range(6):
        ids = [item[i] for item in batch]
        ids_len = [len(item[i]) for item in batch]
        ids = batch_padding(ids, ids_len)
        ids = torch.from_numpy(ids).long()
        new_batch.append(ids)

    labels = torch.FloatTensor([item[6] for item in batch])
    new_batch.append(labels)

    return new_batch


class OppoQuerySet(Dataset):
    def __init__(self, df, dataset='train'):

        q1 = str_list_to_int_list(df['q1'].values)
        q2 = str_list_to_int_list(df['q2'].values)
        self.dataset = dataset

        if dataset is not 'test':
            a = df['label'].values
            self.data = zip(q1, q2, a)
        else:
            self.data = zip(q1, q2, [-1] * len(q1))

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
        共输出 原始q1 q2, 用于MLM的inq1 inq2 out1 out2, q1 q2 inq1 inq2同时用于crf训练
        """
        text1_ids = [tokens.get(t, 100) for t in text1]
        text2_ids = [tokens.get(t, 100) for t in text2]

        mask1_ids, out1_ids = self.random_mask(text1_ids, tokens)
        mask2_ids, out2_ids = self.random_mask(text2_ids, tokens)

        tok1_ids = [101] + mask1_ids + [102]
        tok2_ids = [101] + mask2_ids + [102]

        q1_ids = [101] + text1_ids + [102]
        q2_ids = [101] + text2_ids + [102]



        out1_ids = [0] + out1_ids + [0]
        out2_ids = [0] + out2_ids + [0]

        return q1_ids, q2_ids, tok1_ids, tok2_ids, out1_ids, out2_ids

    def process_data(self, dictionary, random=False):
        self.processed_data = []
        for i in tqdm(range(len(self.data)), desc="Process Data: ", ncols=100, total=len(self.data)):
            q1, q2, label = self.data[i]
            
            q1_ids, q2_ids, tok1_ids, tok2_ids, out1_ids, out2_ids = \
                self.sample_convert(q1, q2, label, random, tokens=dictionary.tok_to_bert_id)
            self.processed_data.append((q1_ids, q2_ids, tok1_ids, tok2_ids, out1_ids, out2_ids, label))

if __name__ == '__main__':
    df_train = pd.read_table("./baseline_tfidf_lr/oppo_breeno_round1_data/gaiic_track3_round1_train_20210228.tsv",
                             names=['q1', 'q2', 'label']).fillna("0")
    df_test = pd.read_table('./baseline_tfidf_lr/oppo_breeno_round1_data/gaiic_track3_round1_testA_20210228.tsv',
                            names=['q1', 'q2']).fillna("0")
    df_train = shuffle(df_train)

    max_value = -1
    min_value = 999999
    max_len = -1

    # split train and val
    train_len = int(0.8 * len(df_train))
    df_val = df_train[train_len:]
    df_train = df_train[0:train_len]

    trainset = OppoQuerySet(df_train, dataset='train')
    valset = OppoQuerySet(df_val, dataset='val')
    testset = OppoQuerySet(df_test, dataset='test')

    # load dictionary
    dictionary = Dictionary.Dictionary(trainset, valset, testset)
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    dictionary.aliment_bert_id(tokenizer)

    trainset.process_data(dictionary, random=True)
    trainloader = DataLoader(trainset, batch_size=3, shuffle=False, collate_fn=my_collate)

    for q1_ids, q2_ids, tok1_ids, tok2_ids, out1_ids, out2_ids, label in trainloader:
        print(q1_ids)
        print(tok1_ids)
        print(out1_ids)
        print(q2_ids)
        print(tok2_ids)
        print(out2_ids)
        break
