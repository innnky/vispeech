import copy
import math
import torch
from torch import nn
from torch.nn import functional as F

import commons
import modules
import attentions

from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from commons import init_weights, get_padding
import utils
from frame_prior_network import VariancePredictor, EnergyPredictor, FramePitchPredictor
# from vdecoder.hifigan.models import Generator

class Generator(torch.nn.Module):
    def __init__(self, initial_channel, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates,
                 upsample_initial_channel, upsample_kernel_sizes, gin_channels=0):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        resblock = modules.ResBlock1 if resblock == '1' else modules.ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(upsample_initial_channel // (2 ** i), upsample_initial_channel // (2 ** (i + 1)),
                                k, u, padding=(k - u) // 2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, g=None):
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()



class StochasticDurationPredictor(nn.Module):
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, n_flows=4, gin_channels=0):
        super().__init__()
        filter_channels = in_channels  # it needs to be removed from future version.
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.log_flow = modules.Log()
        self.flows = nn.ModuleList()
        self.flows.append(modules.ElementwiseAffine(2))
        for i in range(n_flows):
            self.flows.append(modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3))
            self.flows.append(modules.Flip())

        self.post_pre = nn.Conv1d(1, filter_channels, 1)
        self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.post_convs = modules.DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
        self.post_flows = nn.ModuleList()
        self.post_flows.append(modules.ElementwiseAffine(2))
        for i in range(4):
            self.post_flows.append(modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3))
            self.post_flows.append(modules.Flip())

        self.pre = nn.Conv1d(in_channels, filter_channels, 1)
        self.proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.convs = modules.DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, filter_channels, 1)

    def forward(self, x, x_mask, w=None, g=None, reverse=False, noise_scale=1.0):
        x = torch.detach(x)
        x = self.pre(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        x = self.convs(x, x_mask)
        x = self.proj(x) * x_mask

        if not reverse:
            flows = self.flows
            assert w is not None

            logdet_tot_q = 0
            h_w = self.post_pre(w)
            h_w = self.post_convs(h_w, x_mask)
            h_w = self.post_proj(h_w) * x_mask
            e_q = torch.randn(w.size(0), 2, w.size(2)).to(device=x.device, dtype=x.dtype) * x_mask
            z_q = e_q
            for flow in self.post_flows:
                z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w))
                logdet_tot_q += logdet_q
            z_u, z1 = torch.split(z_q, [1, 1], 1)
            u = torch.sigmoid(z_u) * x_mask
            z0 = (w - u) * x_mask
            logdet_tot_q += torch.sum((F.logsigmoid(z_u) + F.logsigmoid(-z_u)) * x_mask, [1, 2])
            logq = torch.sum(-0.5 * (math.log(2 * math.pi) + (e_q ** 2)) * x_mask, [1, 2]) - logdet_tot_q

            logdet_tot = 0
            z0, logdet = self.log_flow(z0, x_mask)
            logdet_tot += logdet
            z = torch.cat([z0, z1], 1)
            for flow in flows:
                z, logdet = flow(z, x_mask, g=x, reverse=reverse)
                logdet_tot = logdet_tot + logdet
            nll = torch.sum(0.5 * (math.log(2 * math.pi) + (z ** 2)) * x_mask, [1, 2]) - logdet_tot
            return nll + logq  # [b]
        else:
            flows = list(reversed(self.flows))
            flows = flows[:-2] + [flows[-1]]  # remove a useless vflow
            z = torch.randn(x.size(0), 2, x.size(2)).to(device=x.device, dtype=x.dtype) * noise_scale
            for flow in flows:
                z = flow(z, x_mask, g=x, reverse=reverse)
            z0, z1 = torch.split(z, [1, 1], 1)
            logw = z0
            return logw


class DurationPredictor(nn.Module):
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0):
        super().__init__()

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_1 = modules.LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_2 = modules.LayerNorm(filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)

    def forward(self, x, x_mask, g=None):
        x = torch.detach(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return x * x_mask


class TextEncoder(nn.Module):
    def __init__(self,
                 n_vocab,
                 out_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout):
        super().__init__()
        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.symbol_emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.symbol_emb.weight, 0.0, hidden_channels ** -0.5)

        self.encoder = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths):
        x = self.symbol_emb(x) * math.sqrt(self.hidden_channels)  # [b, t, h]
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

        x = self.encoder(x * x_mask, x_mask)
        return x, x_mask


class ResidualCouplingBlock(nn.Module):
    def __init__(self,
                 channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 n_flows=4,
                 gin_channels=0):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(
                modules.ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers,
                                              gin_channels=gin_channels, mean_only=True))
            self.flows.append(modules.Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x


class PosteriorEncoder(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 gin_channels=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(get_padding(kernel_size, 1), 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 16, 15, 1, padding=7)),
            norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(MultiPeriodDiscriminator, self).__init__()
        periods = [2, 3, 5, 7, 11]

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class LengthRegulator(nn.Module):
    """Length Regulator"""

    def __init__(self, hoplen, sr):
        super(LengthRegulator, self).__init__()
        self.hoplen = hoplen
        self.sr = sr

    def LR(self, x, duration, x_lengths):
        output = list()
        mel_len = list()
        x = torch.transpose(x, 1, -1)
        frame_lengths = list()

        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            frame_lengths.append(expanded.shape[0])

        max_len = max(frame_lengths)
        output_padded = torch.FloatTensor(x.size(0), max_len, x.size(2))
        output_padded.zero_()
        for i in range(output_padded.size(0)):
            output_padded[i, :frame_lengths[i], :] = output[i]
        output_padded = torch.transpose(output_padded, 1, -1)

        return output_padded, torch.LongTensor(frame_lengths)

    def expand(self, batch, predicted):
        out = list()
        predicted = predicted.squeeze()
        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            vec_expand = vec.expand(max(int(expand_size), 0), -1)
            out.append(vec_expand)

        out = torch.cat(out, 0)
        return out

    def forward(self, x, duration, x_lengths):

        output, x_lengths = self.LR(x, duration, x_lengths)
        return output, x_lengths


class FramePriorNet(nn.Module):
    def __init__(self,
                 n_vocab,
                 out_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout):
        super().__init__()

        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.emb = nn.Embedding(121, hidden_channels)

        self.fft_block = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            4,
            kernel_size,
            p_dropout)

    def forward(self, x_frame, x_mask):
        x = x_frame
        x = self.fft_block(x * x_mask, x_mask)
        x = x.transpose(1, 2)
        return x


class PitchPredictor(nn.Module):
    def __init__(self,
                 mean,
                 std,
                 n_vocab,
                 gin_channels,
                 out_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout):
        super().__init__()
        self.n_vocab = n_vocab  # 音素的个数，中文和英文不同
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.mean = mean
        self.std = std
        self.emb = nn.Embedding(256, hidden_channels)

        self.pitch_net = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout)
        self.proj_f0 = nn.Conv1d(hidden_channels, 1, 1)
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, hidden_channels, 1)

    def normalize(self, f0):
        return (f0 - self.mean) / self.std

    def denormalize(self, norm_f0):
        return norm_f0 * self.std + self.mean

    def forward(self, x, x_mask, f0=None, shift=None, g=None):
        x = torch.detach(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)

        x = self.pitch_net(x * x_mask, x_mask)
        x = x * x_mask
        pred_norm_f0 = self.proj_f0(x).squeeze(1)
        pred_f0 = self.denormalize(pred_norm_f0)
        if f0 is not None:
            embedding = self.emb(utils.f0_to_coarse(f0))
        else:
            shift = 1 if shift is None else shift
            embedding = self.emb(
                utils.f0_to_coarse((pred_f0 * shift))
            )
        return pred_norm_f0, pred_f0, embedding.transpose(1, 2)


class Projection(nn.Module):
    def __init__(self,
                 hidden_channels,
                 out_channels):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_mask):
        stats = self.proj(x) * x_mask
        m_p, logs_p = torch.split(stats, self.out_channels, dim=1)
        return m_p, logs_p


class SynthesizerTrn(nn.Module):
    """
  Synthesizer for Training
  """

    def __init__(self,
                 n_vocab,
                 spec_channels,
                 hop_length,
                 sampling_rate,
                 segment_size,
                 inter_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout,
                 resblock,
                 resblock_kernel_sizes,
                 resblock_dilation_sizes,
                 upsample_rates,
                 upsample_initial_channel,
                 upsample_kernel_sizes,
                 n_speakers=0,
                 gin_channels=0,
                 use_sdp=False,
                 f0_mean=171.21,
                 f0_std=128.9,
                 energy_min=-1.6306,
                 energy_max=10,
                 energy_mean=59.412674114165,
                 energy_std=36.625710,
                 **kwargs):

        super().__init__()
        self.n_vocab = n_vocab
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels

        self.use_sdp = use_sdp

        self.enc_p = TextEncoder(n_vocab,
                                 inter_channels,
                                 hidden_channels,
                                 filter_channels,
                                 n_heads,
                                 n_layers,
                                 kernel_size,
                                 p_dropout)
        hps = {
            "sampling_rate": 44100,
            "inter_channels": 192,
            "resblock": "1",
            "resblock_kernel_sizes": [3, 7, 11],
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "upsample_rates": [8, 8, 4, 2],
            "upsample_initial_channel": 512,
            "upsample_kernel_sizes": [16, 16, 4, 4],
            "gin_channels": 256,
        }
        self.nsf_dec = Generator(h=hps)
        self.enc_q = PosteriorEncoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16,
                                      gin_channels=gin_channels)
        self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels)

        self.dp = DurationPredictor(hidden_channels, 256, 3, 0.5, gin_channels=gin_channels)

        self.lr = LengthRegulator(hop_length, sampling_rate)
        self.frame_prior_net = FramePriorNet(n_vocab, inter_channels, hidden_channels, filter_channels, n_heads,
                                             n_layers, kernel_size, p_dropout)
        self.pitch_net = PitchPredictor(f0_mean, f0_std, n_vocab, gin_channels, inter_channels, hidden_channels,
                                        filter_channels,
                                        n_heads,
                                        n_layers,
                                        kernel_size, p_dropout)
        self.energy_predictor = EnergyPredictor(hidden_channels, gin_channels, energy_min, energy_max, energy_mean,
                                                energy_std)
        self.frame_pitch_emb = nn.Embedding(256, hidden_channels)
        self.project = Projection(hidden_channels, inter_channels)

        if n_speakers > 1:
            self.emb_g = nn.Embedding(n_speakers, gin_channels)

    def forward(self, phonemes, phonemes_lengths, f0, frame_f0, energy, phndur, spec, spec_lengths, sid=None):
        if self.n_speakers > 0:
            g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
        else:
            g = None

        # 文本编码
        x, x_mask = self.enc_p(phonemes, phonemes_lengths)

        # 时长预测
        logw_ = torch.log(phndur.detach().float() + 1).unsqueeze(1) * x_mask
        logw = self.dp(x, x_mask, g=g)
        l_loss = torch.sum((logw - logw_) ** 2, [1, 2])
        x_mask_sum = torch.sum(x_mask)
        l_length = l_loss / x_mask_sum

        # f0预测
        pred_norm_f0, pred_f0, pitch_embedding = self.pitch_net(x, x_mask, f0=f0, g=g)
        l_pitch_ph = F.mse_loss(pred_norm_f0, self.pitch_net.normalize(f0))
        x += pitch_embedding
        # energy预测
        pred_norm_energy, norm_energy, embedding, l_energy = self.energy_predictor(x, energy, g)
        x += embedding

        # 音素级别转换成帧级
        x_frame, x_lengths = self.lr(x, phndur, phonemes_lengths)

        # 补零对齐spec和帧级别输入
        spec_padded = torch.zeros(spec.shape[0], spec.shape[1], x_frame.shape[-1]).float().to(spec.device)
        spec_padded[:, :, :spec.shape[-1]] = spec
        spec = spec_padded.detach()
        x_frame = x_frame.to(x.device)
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x_frame.size(2)), 1).to(x.dtype)  # 更新x_mask矩阵
        x_mask = x_mask.to(x.device)

        x_frame += self.frame_pitch_emb(utils.f0_to_coarse(frame_f0)).transpose(1, 2)
        l_pitch = l_pitch_ph

        # 帧优先级网络
        x_frame = self.frame_prior_net(x_frame, x_mask)
        x_frame = x_frame.transpose(1, 2)
        m_p, logs_p = self.project(x_frame, x_mask)
        z, m_q, logs_q, y_mask = self.enc_q(spec, spec_lengths, g=g)
        z_p = self.flow(z, y_mask, g=g)
        z_slice, pitch_slice, ids_slice = commons.rand_slice_segments_with_pitch(z, frame_f0, spec_lengths,
                                                                                 self.segment_size)
        o = self.nsf_dec(z_slice, g=g, f0=pitch_slice)
        return o, l_length, l_pitch, l_energy, ids_slice, x_mask, y_mask, (
            z, z_p, m_p, logs_p, m_q, logs_q), pred_f0, f0, pred_norm_energy, norm_energy

    def infer(self, phonemes, phonemes_lengths,
              sid=None, noise_scale=1, length_scale=1, noise_scale_w=1., max_len=None,
              shift=None, energy_control=None, pitch_control=None,
              manual_duration=None, manual_f0=None):
        if self.n_speakers > 0:
            g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
        else:
            g = None
        x, x_mask = self.enc_p(phonemes, phonemes_lengths)
        # 时长预测
        logw = self.dp(x, x_mask, g=g)
        w = (torch.exp(logw) * x_mask - 1) * length_scale
        w = torch.ceil(w)
        if pitch_control is None:
            # f0预测
            pred_norm_f0, pred_f0, pitch_embedding = self.pitch_net(x, x_mask, shift=shift, g=g)

        else:
            pred_norm_f0, pred_f0, pitch_embedding = self.pitch_net(x, x_mask, f0=pitch_control, g=g)
            pred_f0 = pitch_control
        x += pitch_embedding
        # energy预测
        energy_emb = self.energy_predictor.infer(x, g, energy_control)
        x += energy_emb
        if manual_duration is not None:
            w = manual_duration.unsqueeze(0)
        # 扩帧
        x_frame, x_lengths = self.lr(x, w, phonemes_lengths)
        x_frame = x_frame.to(x.device)
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x_frame.size(2)), 1).to(x.dtype)
        x_mask = x_mask.to(x.device)


        max_len = x_frame.size(2)
        x_frame += self.frame_pitch_emb(utils.f0_to_coarse(manual_f0)).transpose(1, 2)


        x_frame = self.frame_prior_net(x_frame, x_mask)
        x_frame = x_frame.transpose(1, 2)
        m_p, logs_p = self.project(x_frame, x_mask)
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, x_mask, g=g, reverse=True)
        o = self.nsf_dec((z * x_mask)[:, :, :max_len], g=g, f0=manual_f0*0.5)

        return o, x_mask, (z, z_p, m_p, logs_p), pred_f0
    # 
    # def voice_conversion(self, y, y_lengths, sid_src, sid_tgt):
    #     assert self.n_speakers > 0, "n_speakers have to be larger than 0."
    #     g_src = self.emb_g(sid_src).unsqueeze(-1)
    #     g_tgt = self.emb_g(sid_tgt).unsqueeze(-1)
    #     z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g_src)
    #     z_p = self.flow(z, y_mask, g=g_src)
    #     z_hat = self.flow(z_p, y_mask, g=g_tgt, reverse=True)
    #     o_hat = self.nsf_dec(z_hat * y_mask, g=g_tgt)
    #     return o_hat, y_mask, (z, z_p, z_hat)
