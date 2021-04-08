import pandas as pd
from dataset import OppoQuerySet, DataLoader, my_collate
from sklearn.utils import shuffle
from pytorch_pretrained_bert import BertModel, BertTokenizer, BertForMaskedLM
from Dictionary import Dictionary
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold
import numpy as np

# Load Dataset
df_train = pd.read_table("./baseline_tfidf_lr/oppo_breeno_round1_data/gaiic_track3_round1_train_20210228.tsv",
                         names=['q1', 'q2', 'label']).fillna("0")
df_test = pd.read_table('./baseline_tfidf_lr/oppo_breeno_round1_data/gaiic_track3_round1_testA_20210228.tsv',
                        names=['q1', 'q2']).fillna("0")

df_train_ = shuffle(df_train)

# split train and val
kfold = KFold(n_splits=4, shuffle=False, random_state=None)
test_results = np.zeros((4, len(df_test)))
for index, (train_index, val_index) in enumerate(kfold.split(df_train_)):
    df_val = df_train_.iloc[val_index]
    df_train = df_train_.iloc[train_index]

    trainset = OppoQuerySet(df_train, dataset='train')
    valset = OppoQuerySet(df_val, dataset='val')
    testset = OppoQuerySet(df_test, dataset='test')

    # load  dictionary
    dictionary = Dictionary(trainset, valset, testset)

    # aliment dictionary id with bert token id
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    model = BertForMaskedLM.from_pretrained('bert-base-chinese')
    dictionary.aliment_bert_id(tokenizer)

    # process data
    trainset.process_data(dictionary, random=True)
    testset.process_data(dictionary, random=False)
    valset.process_data(dictionary, random=False)

    # ## TEST
    # tok_id, seg_id, output, label = testset[2]
    # print(tokenizer.convert_ids_to_tokens(tok_id))

    #########
    bs = 32
    epoch_num = 50
    lr = 1e-5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataloader
    trainloader = DataLoader(trainset, batch_size=bs, shuffle=True, collate_fn=my_collate, num_workers=4)
    valloader = DataLoader(valset, batch_size=bs, shuffle=False, collate_fn=my_collate, num_workers=4)
    testloader = DataLoader(testset, batch_size=bs, shuffle=False, collate_fn=my_collate, num_workers=4)

    # model
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    max_val_score = 0.0

    for epoch in range(epoch_num):
        train_score = 0.
        train_num = 0
        train_right = 0

        val_score = 0.
        val_num = 0
        val_right = 0
        model.train()

        train_y_true = []
        train_y_pred = []

        for i, (q1_ids, q2_ids, tok1_ids, tok2_ids, out1_ids, out2_ids, label) in tqdm(enumerate(trainloader),
                                                                                       ncols=100,
                                                                                       desc="Epoch %d" % (epoch + 1),
                                                                                       total=len(trainloader)):
            # prepare data
            optim.zero_grad()
            q1, q2, q11, q21, q1_out, q2_out, label = q1_ids.to(device), q2_ids.to(device), \
                                                      tok1_ids.to(device), tok2_ids.to(device), out1_ids.to(
                device), out2_ids.to(device), label.to(device)

            # predict
            pred, loss = model(q1, q2, q11, q21, q1_out, q2_out, label)  # bs 128

            # backward
            loss.backward()
            optim.step()

            # train score and acc
            train_right += ((pred >= 0.5).float() * label + (pred < 0.5).float() * (1 - label)).sum().item()
            train_num += label.size(0)

            train_y_true.extend(list(label.cpu().numpy()))
            train_y_pred.extend(list(pred.cpu().detach().numpy()))
        train_score = roc_auc_score(train_y_true, train_y_pred)

        model.eval()
        val_y_true = []
        val_y_pred = []
        for i, (q1_ids, q2_ids, tok1_ids, tok2_ids, out1_ids, out2_ids, label) in enumerate(valloader):
            q1, q2, q11, q21, q1_out, q2_out, label = q1_ids.to(device), q2_ids.to(device), \
                                                      tok1_ids.to(device), tok2_ids.to(device), out1_ids.to(
                device), out2_ids.to(device), label.to(device)
            # predict
            pred, _ = model(q1, q2)  # bs 128

            # val score
            val_right += ((pred >= 0.5).float() * label + (pred < 0.5).float() * (1 - label)).sum().item()
            val_num += label.size(0)

            val_y_true.extend(list(label.cpu().numpy()))
            val_y_pred.extend(list(pred.cpu().detach().numpy()))

        val_score = roc_auc_score(val_y_true, val_y_pred)

        print("Epoch: {}/{} \t train score: {} \t train_acc: {}\t val_score: {}\t val_acc: {}".
              format(epoch, epoch_num, train_score, train_right / train_num, val_score, val_right / val_num))

        if val_score / val_num > max_val_score:
            max_val_score = val_score / val_num
            torch.save(model.state_dict(), 'best_model.pth')

            model.eval()
            pred_size = 0
            for i, (q1_ids, q2_ids, tok1_ids, tok2_ids, out1_ids, out2_ids, label) in enumerate(testloader):
                q1, q2 = q1_ids.to(device), q2_ids.to(device)

                # predict
                pred = model(q1, q2)  # bs 128
                if i == 0:
                    pred_size = pred.size(0)
                for k in range(pred.size(0)):
                    p = pred[k]
                    test_results[index][i * pred_size + k] = p.item()

            with open('result.tsv', 'w') as f:
                for j in range(len(test_results[index])):
                    f.write('%f\n' % test_results[index][j])

    model.to(torch.device('cpu'))
avg_results = np.mean(test_results, axis=0)
with open('avg_result.tsv', 'w') as f:
    for i in range(len(avg_results)):
        f.write('%f\n' % avg_results[i])
