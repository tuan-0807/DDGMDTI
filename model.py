from MCA import *
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math  # 新增导入，解决GraphConvolution中math.log报错


def binary_cross_entropy(pred_output, labels):
    loss_fct = torch.nn.BCELoss()
    m = nn.Sigmoid()
    n = torch.squeeze(m(pred_output), 1)
    loss = loss_fct(n, labels)
    return n, loss


def cross_entropy_logits(linear_output, label, weights=None):
    class_output = F.log_softmax(linear_output, dim=1)
    n = F.softmax(linear_output, dim=1)[:, 1]
    max_class = class_output.max(1)
    y_hat = max_class[1]  # get the index of the max log-probability
    if weights is None:
        loss = nn.NLLLoss()(class_output, label.type_as(y_hat).view(label.size(0)))
    else:
        losses = nn.NLLLoss(reduction="none")(class_output, label.type_as(y_hat).view(label.size(0)))
        loss = torch.sum(weights * losses) / torch.sum(weights)
    return n, loss



class DDGMDTI(nn.Module):
    def __init__(self, hyperparam_dict, args):
        super(DDGMDTI, self).__init__()


        self.pro_max_nodes = hyperparam_dict['PROTEIN_MAX_NODES']

        mlp_in_dim = hyperparam_dict['DECODER_IN_DIM']  # 1024
        mlp_hidden_dim = hyperparam_dict['DECODER_HIDDEN_DIM']  # 512
        mlp_out_dim = hyperparam_dict['DECODER_OUT_DIM']  # 128
        out_binary = hyperparam_dict['DECODER_BINARY']  # 1


        self.drug_seq_extractor1 = Dynamic_conv1d(in_planes=768, out_planes=512, kernel_size=3, ratio=0.25, padding=1)
        self.drug_seq_extractor2 = Dynamic_conv1d(in_planes=512, out_planes=512, kernel_size=3, ratio=0.25, padding=1)
        self.gru_drug = nn.GRU(input_size=512, hidden_size=256, batch_first=True, bidirectional=True)

        # 修改蛋白质特征提取器的输入维度以适应ProtT5
        # prott5_dim = 1024 ProtT5的特征维度
        self.protein_seq_extractor = deepGCN(nlayers = 3, nfeat = 1024, nhidden = 512, nclass = 512,#1024+531
                                dropout = 0.1, lamda = 1.5, alpha = 0.7, variant = False)
        #self.protein_seq_extractor2 = deepGCN(nlayers = nlayers, nfeat = 512, nhidden = nhidden, nclass = nclass,
        #                       dropout = dropout, lamda = lamda, alpha = alpha, variant = variant)


        self.mca = MCA_ED(args)

        self.mlp_classifier = MLPDecoder(mlp_in_dim, mlp_hidden_dim, mlp_out_dim, binary=out_binary)


    def forward(self, mol_emd, prott5_emd, adj, mode="train"):
        mol_emd = mol_emd.permute(0, 2, 1)
        v_d_seq = self.drug_seq_extractor1(mol_emd)  # v_d_seq大小为(64,512,290)
        v_d_seq = self.drug_seq_extractor2(v_d_seq)  # (64,512,290)
        v_d_seq = v_d_seq.permute(0, 2, 1)  # 调整维度以适应GRU
        v_d_seq, _ = self.gru_drug(v_d_seq)
        v_d = v_d_seq

        #prott5_emd=torch.squeeze(prott5_emd,dim=0)
        v_p = self.protein_seq_extractor(prott5_emd, adj)
        # prott5_emd = prott5_emd.permute(0, 2, 1)
        # v_p_seq = self.protein_seq_extractor1(prott5_emd)  # v_p_seq大小为(64,512,600)
        # v_p_seq = self.protein_seq_extractor2(v_p_seq)
        # v_p_seq = v_p_seq.permute(0, 2, 1)  # 调整维度以适应GRU
        # v_p_seq, _ = self.gru_protein(v_p_seq)
        # v_p = v_p_seq

        f = self.mca(v_d, v_p, None, None)  # f的大小为(64,1024)
        score = self.mlp_classifier(f)  # score大小为(64,1)

        if mode == "train":
            return v_d, v_p, f, score  # v_d, v_p表示未放入BAN网络中的药物和靶点的特征向量 f表示经过BAN后的特征向量
        elif mode == "eval":
            return v_d, v_p, score



class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):  # in_dim=1024; hidden_dim=512; outdim=128
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        #self.dropout1 = nn.Dropout(0.5)#此处为新加部分
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        #self.dropout2 = nn.Dropout(0.5)#此处为新加部分
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        #x = self.dropout1(x)#此处为新加部分
        x = self.bn2(F.relu(self.fc2(x)))
        #x = self.dropout2(x)#此处为新加部分
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x

class attention1d(nn.Module):
    def __init__(self, in_planes, ratios, K, temperature, init_weight=True):
        super(attention1d, self).__init__()
        assert temperature%3==1
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        if in_planes!=3:
            hidden_planes = int(in_planes*ratios)+1
        else:
            hidden_planes = K
        self.fc1 = nn.Conv1d(in_planes, hidden_planes, 1, bias=False)
        # self.bn = nn.BatchNorm2d(hidden_planes)
        self.fc2 = nn.Conv1d(hidden_planes, K, 1, bias=True)
        self.temperature = temperature
        if init_weight:
            self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def updata_temperature(self):
        if self.temperature!=1:
            self.temperature -=3
            print('Change temperature to:', str(self.temperature))


    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).reshape(x.size(0), -1)
        return F.softmax(x/self.temperature, 1)


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, residual=False, variant=False):
        super(GraphConvolution, self).__init__()
        self.variant = variant
        if self.variant:
            self.in_features = 2*in_features
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        # 将参数转换为整数
        self.weight = Parameter(torch.FloatTensor(int(self.in_features),int(self.out_features)))
        #self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, prott5_emd, adj , h0 , lamda, alpha, l):
        theta = min(1, math.log(lamda/l+1))

        if adj.dtype != prott5_emd.dtype:
            adj = adj.to(prott5_emd.dtype)

        batch_size = prott5_emd.size(0)
        outputs = []

        # 对批次中的每个样本单独应用图卷积操作
        for i in range(batch_size):
            # 处理单个样本
            hi = torch.spmm(adj, prott5_emd[i])  # adj: [num_nodes, num_nodes], prott5_emd[i]: [num_nodes, feature_dim]

            if self.variant:
                support = torch.cat([hi, h0[i]], 1)  # 确保h0也按批次处理
                r = (1 - alpha) * hi + alpha * h0[i]
            else:
                support = (1 - alpha) * hi + alpha * h0[i]
                r = support

            output = theta * torch.mm(support, self.weight) + (1 - theta) * r

            if self.residual:
                output = output + prott5_emd[i]

            outputs.append(output)

        # 将处理后的样本重新组合成批次
        return torch.stack(outputs, dim=0)

class deepGCN(nn.Module):
    def __init__(self, nlayers, nfeat, nhidden, nclass, dropout, lamda, alpha, variant):
        super(deepGCN, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GraphConvolution(nhidden, nhidden, variant=variant, residual=True))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda
        #self.layer_drop = nn.Dropout(0.3)  # 新增层间Dropout

    def forward(self, x, adj):
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        for i, con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(con(layer_inner, adj, _layers[0], self.lamda, self.alpha, i + 1))
            #layer_inner = self.layer_drop(layer_inner)  # 此处为新加部分，应用层间Dropout
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)

        return layer_inner

class Dynamic_conv1d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, ratio=0.25, stride=1, padding=0, dilation=1, groups=1, bias=True, K=4,temperature=34, init_weight=True):
        super(Dynamic_conv1d, self).__init__()
        assert in_planes%groups==0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.attention = attention1d(in_planes, ratio, K, temperature)

        self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes//groups, kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.zeros(K, out_planes))
        else:
            self.bias = None
        if init_weight:
            self._initialize_weights()

        #TODO 初始化
    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])


    def update_temperature(self):
        self.attention.updata_temperature()

    def forward(self, x):#将batch视作维度变量，进行组卷积，因为组卷积的权重是不同的，动态卷积的权重也是不同的
        softmax_attention = self.attention(x)
        batch_size, in_planes, height = x.size()
        # x = x.view(1, -1, height, )# 变化成一个维度进行组卷积
        x = x.reshape(1, -1, height)

        weight = self.weight.reshape(self.K, -1)

        # 动态卷积的权重的生成， 生成的是batch_size个卷积参数（每个参数不同）
        aggregate_weight = torch.mm(softmax_attention, weight).reshape(batch_size*self.out_planes, self.in_planes//self.groups, self.kernel_size,)
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).reshape(-1)
            output = F.conv1d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups*batch_size)
        else:
            output = F.conv1d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)

        output = output.reshape(batch_size, self.out_planes, output.size(-1))
        return output
