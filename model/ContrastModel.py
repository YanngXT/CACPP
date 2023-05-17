import torch
import torch.nn as nn
import torch.nn.functional as F
pep2label = list()

import sys
from configuration import config as cf
print(sys.path)

class newModel(nn.Module):
    def __init__(self, vocab_size=25):  # 4):
        super().__init__()
        config = cf.get_train_config()
        self.devicenum = config.devicenum

        # self.filter_sizes = [1, 2]  # ACPred
        self.filter_sizes = [1, 2, 3, 4, 6, 8, 16, 32]  # Contrast
        self.embedding_dim = 100  # the MGF process dim
        dim_cnn_out = 128
        filter_num = 64  

        self.embedding = nn.Embedding(vocab_size, self.embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, filter_num, (fsz, self.embedding_dim)) for fsz in self.filter_sizes])
        self.dropout = nn.Dropout(0.5)
        self.block1 = nn.Sequential(nn.Linear(len(self.filter_sizes) * filter_num, 256),
                                    nn.BatchNorm1d(256),
                                    nn.LeakyReLU(),
                                    nn.Linear(256, 64),
                                    )
        self.classification = nn.Sequential(
            nn.Linear(64, 2),
        )
    def forward(self, x):
        x = x.cuda(device=self.devicenum)
        # # 输入x的维度为(batch_size, max_len), max_len可以通过torchtext设置或自动获取为训练样本的最大=长度
        x = self.embedding(x)  # 经过embedding,x的维度为(batch_size, max_len, embedding_dim)
        # print('embedding x', x.size())#([64, 200, 100])

        # 经过view函数x的维度变为(batch_size, input_chanel=1, w=max_len, h=embedding_dim)
        x = x.view(x.size(0), 1, x.size(1), self.embedding_dim)
        # print('view x', x.size())#([64, 1, 200, 100])

        # 经过卷积运算,x中每个运算结果维度为(batch_size, out_chanel, w, h=1)
        x = [F.relu(conv(x)) for conv in self.convs]
        # print(x)
        # print('conv x', len(x), [x_item.size() for x_item in x])

        # 经过最大池化层,维度变为(batch_size, out_chanel, w=1, h=1)
        x = [F.max_pool2d(input=x_item, kernel_size=(x_item.size(2), x_item.size(3))) for x_item in x]
        # print('max_pool2d x', len(x), [x_item.size() for x_item in x])

        # 将不同卷积核运算结果维度（batch，out_chanel,w,h=1）展平为（batch, outchanel*w*h）
        x = [x_item.view(x_item.size(0), -1) for x_item in x]
        # print('flatten x', len(x), [x_item.size() for x_item in x])

        # 将不同卷积核提取的特征组合起来,维度变为(batch, sum:outchanel*w*h)
        x = torch.cat(x, 1)
        # print('concat x', x.size()) torch.Size([320, 1024])

        # dropout层
        x = self.dropout(x)

        # 全连接层
        # representation = self.linear(x)
        output = self.block1(x)

        return output

    def trainModel(self, x):
        # with torch.no_grad():
        output = self.forward(x)

        return self.classification(output)

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # euclidean_distance: [128]
        # print('label.shape', label.shape)
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +  # calmp夹断用法
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive
