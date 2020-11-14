import torch.nn as nn
import torch.nn.functional as F
import torch
import random

from torch.autograd import Variable
from CNN_model import CNN


class BatchTreeEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, encode_dim, batch_size, use_gpu, pretrained_weight=None):
        super(BatchTreeEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dim = embedding_dim
        self.encode_dim = encode_dim
        self.W_c = nn.Linear(embedding_dim, encode_dim)
        self.activation = F.relu
        self.stop = -1
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.node_list = []
        self.th = torch.cuda if use_gpu else torch
        self.batch_node = None
        self.max_index = vocab_size
        self.cnn_model = CNN()
        # pretrained  embedding
        if pretrained_weight is not None:
            self.embedding.weight.data.copy_(
                torch.from_numpy(pretrained_weight))
            # self.embedding.weight.requires_grad = False

    def create_tensor(self, tensor):
        if self.use_gpu:
            return tensor.cuda()
        return tensor

    def traverse_mul(self, node, batch_index):
        size = len(node)
        if not size:
            return None
        batch_current = self.create_tensor(
            Variable(torch.zeros(size, self.embedding_dim)))

        index, children_index = [], []
        current_node, children = [], []
        for i in range(size):
            # if node[i][0] is not -1:
            index.append(i)
            current_node.append(node[i][0])
            temp = node[i][1:]
            c_num = len(temp)
            for j in range(c_num):
                if temp[j][0] is not -1:
                    if len(children_index) <= j:
                        children_index.append([i])
                        children.append([temp[j]])
                    else:
                        children_index[j].append(i)
                        children[j].append(temp[j])
            # else:
            #     batch_index[i] = -1

        batch_current = self.W_c(batch_current.index_copy(0, Variable(self.th.LongTensor(index)),
                                                          self.embedding(Variable(self.th.LongTensor(current_node)))))

        for c in range(len(children)):
            zeros = self.create_tensor(
                Variable(torch.zeros(size, self.encode_dim)))
            batch_children_index = [batch_index[i] for i in children_index[c]]
            tree = self.traverse_mul(children[c], batch_children_index)
            if tree is not None:
                batch_current += zeros.index_copy(0, Variable(
                    self.th.LongTensor(children_index[c])), tree)
        # batch_index = [i for i in batch_index if i is not -1]
        b_in = Variable(self.th.LongTensor(batch_index))
        self.node_list.append(
            self.batch_node.index_copy(0, b_in, batch_current))
        return batch_current

    def forward(self, x, bs):
        self.batch_size = bs
        self.batch_node = self.create_tensor(
            Variable(torch.zeros(self.batch_size, self.encode_dim)))
        self.node_list = []
        self.traverse_mul(x, list(range(self.batch_size)))
        self.node_list = torch.stack(self.node_list)
        return torch.max(self.node_list, 0)[0]


class BatchProgramCC(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, encode_dim, label_size, batch_size, use_gpu=True, pretrained_weight=None):
        super(BatchProgramCC, self).__init__()
        self.stop = [vocab_size-1]
        self.hidden_dim = hidden_dim
        self.num_layers = 1
        self.gpu = use_gpu
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.encode_dim = encode_dim
        self.label_size = label_size
        self.encoder = BatchTreeEncoder(self.vocab_size, self.embedding_dim, self.encode_dim,
                                        self.batch_size, self.gpu, pretrained_weight)
        self.root2label = nn.Linear(self.encode_dim, self.label_size)
        # gru
        self.bigru = nn.GRU(self.encode_dim, self.hidden_dim, num_layers=self.num_layers, bidirectional=True,
                            batch_first=True)
        # linear
        self.hidden2label = nn.Linear(
            self.hidden_dim * 2 * 30, self.label_size)

        self.cnn2label = nn.Linear(self.hidden_dim, self.label_size)

        # hidden
        self.hidden = self.init_hidden()
        self.dropout = nn.Dropout(0.2)

        # 关于attention的
        self.W_s1 = nn.Linear(2 * self.hidden_dim, 350)
        self.W_s2 = nn.Linear(350, 30)
        # self.fc_layer = nn.Linear(30*2*hidden_size, 2000)

        # 关于CNN的

    def init_hidden(self):
        if self.gpu is True:
            if isinstance(self.bigru, nn.LSTM):
                h0 = Variable(torch.zeros(self.num_layers * 2,
                                          self.batch_size, self.hidden_dim).cuda())
                c0 = Variable(torch.zeros(self.num_layers * 2,
                                          self.batch_size, self.hidden_dim).cuda())
                return h0, c0
            return Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim)).cuda()
        else:
            return Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim))

    def get_zeros(self, num):
        zeros = Variable(torch.zeros(num, self.encode_dim))
        if self.gpu:
            return zeros.cuda()
        return zeros

    def attention_net(self, bigru_out):
        """
        Now we will use self attention mechanism to produce a matrix embedding of the input sentence in which every row represents an
                encoding of the input sentence but giving an attention to a specific part of the sentence. 

        We will use 30 such embedding of the input sentence and then finally we will concatenate all the 30 sentence embedding vectors and connect it to a fully 
                connected layer of size 2000 which will be connected to the output layer of size 2 returning logits for our two classes i.e., 
                pos & neg.

                Arguments
                ---------
                lstm_output = A tensor containing hidden states corresponding to each time step of the LSTM network.
                ---------
                Returns : Final Attention weight matrix for all the 30 different sentence embedding in which each of 30 embeddings give
                                  attention to different parts of the input sentence.
                Tensor size : lstm_output.size() = (batch_size, Fnum_seq, 2*hidden_size)
                                          attn_weight_matrix.size() = (batch_size, 30, num_seq)
                """

        attn_weight_matrix = self.W_s2(F.tanh(self.W_s1(bigru_out)))
        attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)
        attn_weight_matrix = F.softmax(attn_weight_matrix, dim=2)

        return attn_weight_matrix

    def encode(self, x):
        # 取一个batch中的所有ast的 最大语句树个数
        lens = [len(item) for item in x]
        max_len = max(lens)

        encodes = []
        # 对每一个样本i
        for i in range(self.batch_size):
            # 取样本i的每一个语句树j
            for j in range(lens[i]):
                encodes.append(x[i][j])

        encodes = self.encoder(encodes, sum(lens))
        seq, start, end = [], 0, 0
        for i in range(self.batch_size):
            end += lens[i]
            if max_len-lens[i]:
                seq.append(self.get_zeros(max_len-lens[i]))
            seq.append(encodes[start:end])
            start = end
        encodes = torch.cat(seq)
        encodes = encodes.view(self.batch_size, max_len, -1)
        # return encodes
        gru_out, hidden = self.bigru(encodes, self.hidden)
        # print(gru_out.size())
        # (batch_size, num_seq, 2*encode_dim) 是输入到attention的表示
        # print(hidden.size())
        # (2, batch_size, encode_dim)

        # pooling
        #gru_out = F.max_pool1d(gru_out, gru_out.size(2)).squeeze(2)
        # gru_out = gru_out[:,-1]
        # gru_out.size() [atch_szie, 2* embedding_dim]

        # attention
        # attn_weight_matrix = self.attention_net(gru_out)
        # hidden_matrix = torch.bmm(attn_weight_matrix, gru_out)
        # hidden_matrix.size():[batch_size, 30, 2*embedding_dim]

        # fully connected
        # gru_with_attn_out = hidden_matrix.view(-1, hidden_matrix.size()[1]*hidden_matrix.size()[2])
        # hidden_matrix.size(): [batch_size, 30*2*embedding_dim]

        # cnn
        gru_cnn_out = self.cnn_model(bigru_out)

        # return gru_with_attn_out
        return gru_cnn_out
    # x1和x2都是输入的一个batch，每个batch包含：32个样本，每个样本是由ast拆分得到的所有语句树序列组成的
    # 即：每个样本是一个完整的ast树拆分得到的语句树组成，这些语句树的每个结点都被word2vec嵌入表示
    # ast的某个语句树的表示形式: [1, [21, [34, [50]], [138]]]
    # 同一个list中的结点是兄弟结点, 临近的在不同list的结点是父母-孩子结点关系, 如21是1的孩子结点, 34是21的孩子结点

    def forward(self, x1, x2):
        lvec, rvec = self.encode(x1), self.encode(x2)
        # print(lvec.size(), rvec.size())
        # gru_with_attn:(batch_size,3*2*embedding_dim)
        # gru_cnn_out： (batch_size, num_kernels*out_channels)

        # 一维范数计算两个编码的距离
        abs_dist = torch.abs(torch.add(lvec, -rvec))
        # print(abs_dist.size())
        # (batch_size,3*2*embedding_dim)

        # gru_with_attn：
        #y = torch.sigmoid(self.hidden2label(abs_dist))

        # gru_cnn_out：

        y = torch.sigmoid(self.cnn2label(abs_dist))
        return y
