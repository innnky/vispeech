# source from https://github.com/KinglittleQ/GST-Tacotron/blob/master/GST.py

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class Hyperparameters():

    data = '/d/blizzard/lessac_cathy5/wavn'
    #data = '../../../data/data_thchs30'

    max_Ty = max_iter = 200

    # gpu = 2
    device = 'cuda:0'
    # device = 'cpu'

    lr = 0.001
    batch_size = 16   # !!!
    num_epochs = 100  # !!!
    eval_size = 1
    save_per_epoch = 1
    log_per_batch = 20
    log_dir = './log/train{}'

    model_path = None
    optimizer_path = None

    # eval_text = '''er2 dui4 lou2 shi4 cheng2 jiao1 yi4 zhi4 zuo4 yong4 zui4 da4 de5 xian4 gou4'''
    #eval_text = '''chua1n pu3 zo3ng to3ng shuo1 ta1 ce2ng ji1ng xia4ng me3i guo2 re2n mi2n che2ng nuo4 jia1ng yo3u yi1 ge4 me3i ha3o de she4ng da4n li3 wu4  sui2 zhe zhe4 yi1 jia3n shui4 fa3 a4n to1ng guo4  ta1 ye3 dui4 xia4n le zhe4 yi1 che2ng nuo4'''
    eval_text = 'it took me a long time to develop a brain . now that i have it i\'m not going to be silent !'
    ref_wav = '/d/blizzard/lessac_cathy5/wavn/PP_309_093.wav'

    lr_step = [500000, 1000000, 2000000]

    vocab = "PE abcdefghijklmnopqrstuvwxyz'.?"  # english
    #vocab = "PE abcdefghijklmnopqrstuvwxyz12345.?"  # chinese
    char2idx = {char: idx for idx, char in enumerate(vocab)}

    E = 256

    # reference encoder
    ref_enc_filters = [32, 32, 64, 64, 128, 128]
    ref_enc_size = [3, 3]
    ref_enc_strides = [2, 2]
    ref_enc_pad = [1, 1]
    ref_enc_gru_size = E // 2

    # style token layer
    token_num = 25
    # token_emb_size = 256
    num_heads = 8
    # multihead_attn_num_unit = 256
    # style_att_type = 'mlp_attention'
    # attn_normalize = True

    K = 16
    decoder_K = 8
    embedded_size = E
    dropout_p = 0.5
    num_banks = 15
    num_highways = 4

    # sr = 22050  # Sample rate.
    sr = 16000  # keda, thchs30, aishell
    n_fft = 1024  # fft points (samples) - ALE changed this from 2048
    frame_shift = 0.0125  # seconds
    frame_length = 0.05  # seconds
    hop_length = int(sr * frame_shift)  # samples.
    win_length = int(sr * frame_length)  # samples.
    n_mels = 80  # Number of Mel banks to generate
    power = 1.2  # Exponent for amplifying the predicted magnitude
    n_iter = 50  # Number of inversion iterations
    preemphasis = .97  # or None
    max_db = 100
    ref_db = 20

    n_priority_freq = int(3000 / (sr * 0.5) * (n_fft / 2))

    r = 5

    use_gpu = torch.cuda.is_available()


class ReferenceEncoder(nn.Module):
    '''
    inputs --- [N, Ty/r, n_mels*r]  mels
    outputs --- [N, ref_enc_gru_size]
    '''

    def __init__(self, hp):

        super().__init__()
        K = len(hp.ref_enc_filters)
        filters = [1] + hp.ref_enc_filters

        convs = [nn.Conv2d(in_channels=filters[i],
                           out_channels=filters[i + 1],
                           kernel_size=(3, 3),
                           stride=(2, 2),
                           padding=(1, 1)) for i in range(K)]
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList([nn.BatchNorm2d(num_features=hp.ref_enc_filters[i]) for i in range(K)])

        out_channels = self.calculate_channels(hp.n_mels, 3, 2, 1, K)
        self.gru = nn.GRU(input_size=hp.ref_enc_filters[-1] * out_channels,
                          hidden_size=hp.E // 2,
                          batch_first=True)
        self.n_mels = hp.n_mels

    def forward(self, inputs):
        N = inputs.size(0)
        out = inputs.view(N, 1, -1, self.n_mels)  # [N, 1, Ty, n_mels]
        for conv, bn in zip(self.convs, self.bns):
            out = conv(out)
            out = bn(out)
            out = F.relu(out)  # [N, 128, Ty//2^K, n_mels//2^K]

        out = out.transpose(1, 2)  # [N, Ty//2^K, 128, n_mels//2^K]
        T = out.size(1)
        N = out.size(0)
        out = out.contiguous().view(N, T, -1)  # [N, Ty//2^K, 128*n_mels//2^K]

        _, out = self.gru(out)  # out --- [1, N, E//2]

        return out.squeeze(0)

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for _ in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L


class STL(nn.Module):
    '''
    inputs --- [N, E//2]
    '''
    def __init__(self, hp):
        super().__init__()
        self.embed = nn.Parameter(torch.FloatTensor(hp.token_num, hp.E // hp.num_heads))
        d_q = hp.E // 2
        d_k = hp.E // hp.num_heads
        self.attention = MultiHeadAttention(query_dim=d_q, key_dim=d_k, num_units=hp.E, num_heads=hp.num_heads)

        init.normal_(self.embed, mean=0, std=0.5)

    def forward(self, inputs):
        N = inputs.size(0)
        query = inputs.unsqueeze(1)  # [N, 1, E//2]
        keys = F.tanh(self.embed).unsqueeze(0).expand(N, -1, -1)  # [N, token_num, E // num_heads]
        style_embed = self.attention(query, keys)

        return style_embed


class MultiHeadAttention(nn.Module):
    '''
    input:
        query --- [N, T_q, query_dim]
        key --- [N, T_k, key_dim]
    output:
        out --- [N, T_q, num_units]
    '''
    def __init__(self, query_dim, key_dim, num_units, num_heads):
        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim

        self.W_query = nn.Linear(in_features=query_dim, out_features=num_units, bias=False)
        self.W_key = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
        self.W_value = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)

    def forward(self, query, key):
        querys = self.W_query(query)  # [N, T_q, num_units]
        keys = self.W_key(key)  # [N, T_k, num_units]
        values = self.W_value(key)

        split_size = self.num_units // self.num_heads
        querys = torch.stack(torch.split(querys, split_size, dim=2), dim=0)  # [h, N, T_q, num_units/h]
        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]

        # score = softmax(QK^T / (d_k ** 0.5))
        scores = torch.matmul(querys, keys.transpose(2, 3))  # [h, N, T_q, T_k]
        scores = scores / (self.key_dim ** 0.5)
        scores = F.softmax(scores, dim=3)

        # out = score * V
        out = torch.matmul(scores, values)  # [h, N, T_q, num_units/h]
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]

        return out


class GST(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = ReferenceEncoder(Hyperparameters)
        self.stl = STL(Hyperparameters)

    def forward(self, inputs):
        enc_out = self.encoder(inputs)
        style_embed = self.stl(enc_out)

        return style_embed


if __name__ == '__main__':
    spec = torch.zeros([8, 1025, 454])
    gst = GST()
    prenet = nn.Conv1d(1025, 80, 3, 1)
    mel = prenet(spec)
    print(mel.shape)
    print(gst(mel).transpose(1,2).shape)

    torch.save(gst.state_dict(), "gst.pth")

