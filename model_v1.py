import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel


def cosine_similarity_of(x, y):
    bs = x.size(0)

    # normlize
    x = x.view(bs, -1)
    y = y.view(bs, -1)
    x_feat = nn.functional.normalize(x, dim=1)
    y_feat = nn.functional.normalize(y, dim=1)

    # cosine_similarity
    return x_feat.matmul(y_feat.transpose(1, 0))


class QueryModel(nn.Module):

    def __init__(self):
        super(QueryModel, self).__init__()

        self.encoder = BertModel.from_pretrained('bert-base-chinese')
        # 0-1
        self.classifier = nn.Sequential(
            nn.Linear(768, 512, bias=True),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

        # for contrastive loss, 768->2048
        self.projection_head = nn.Sequential(
            nn.Linear(768, 2048, bias=True),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 2048)
        )

        # for mlm, 768->21128
        self.mlm_head = nn.Sequential(
            nn.Linear(768, 2048, bias=True),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 21128)
        )

    def forward(self, q1, q2, q11=None, q21=None, q1_out=None, q2_out=None, label=None, temp=0.5):
        bs = q1.size(0)

        loss = 0

        _, q1_feat = self.encoder(q1, output_all_encoded_layers=False)  # bs, 768
        _, q2_feat = self.encoder(q2, output_all_encoded_layers=False)  # bs, 768
        # joint feature
        joint_feature = q1_feat * q2_feat
        predict = self.classifier(joint_feature)  # bs, 1

        if label is not None:
            loss += nn.functional.binary_cross_entropy(predict, label)

        if q1_out is not None:
            q11_out, q11_feat = self.encoder(q1, output_all_encoded_layers=False)  # bs,seq_len, 768; bs, 768
            q21_out, q21_feat = self.encoder(q2, output_all_encoded_layers=False)  # bs,seq_len, 768; bs, 768

            # mlm predict
            q11_pred = self.mlm_head(q11_out)
            q21_pred = self.mlm_head(q21_out)

            # mlm loss
            loss += nn.functional.cross_entropy(q11_pred.view(-1, 21128), q11_out.view(-1), ignore_index=0)
            loss += nn.functional.cross_entropy(q21_pred.view(-1, 21128), q21_out.view(-1), ignore_index=0)

            # crl loss
            q1_q11_sim = cosine_similarity_of(q1_feat, q11_feat)
            q2_q21_sim = cosine_similarity_of(q2_feat, q21_feat)
            crl_label = torch.LongTensor([i for i in range(bs)]).cuda()  # batchsize * 1
            loss += nn.functional.cross_entropy(q1_q11_sim / temp, crl_label)
            loss += nn.functional.cross_entropy(q2_q21_sim / temp, crl_label)

            if label is not None:
                nonzero = torch.sum(label).int().item()

                if nonzero > 0:
                    top_arg = torch.topk(label, k=nonzero)
                    sub_q1_feat = q1_feat[top_arg]
                    sub_q2_feat = q2_feat[top_arg]
                    q1_q2_sim = cosine_similarity_of(sub_q1_feat, sub_q2_feat)
                    crl_label = torch.LongTensor([i for i in range(nonzero)]).cuda()
                    loss += nn.functional.cross_entropy(q1_q2_sim / temp, crl_label)

        return predict, loss


if __name__ == '__main__':
    # test cosine similarity
    x = torch.randn(3, 212)
    print(cosine_similarity_of(x, x))

    from pytorch_pretrained_bert import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    s = "我爱中国"
    tokens = tokenizer.tokenize(s)
    ids = torch.tensor([tokenizer.convert_tokens_to_ids(tokens)])

    model = QueryModel()
    p, l = model(ids, ids, ids, ids)
    print(p, l)
