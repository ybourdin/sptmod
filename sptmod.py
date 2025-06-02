from math import ceil, floor
from typing import Sequence
import torch
from torch import nn
from torch.nn import functional as F
from rnn import RNN

class CachedPadding1d(torch.nn.Module):
    """
    Cached Padding implementation, replace zero padding with the end of the previous tensor.
    https://github.com/acids-ircam/cached_conv/blob/master/cached_conv/convs.py
    """
    NoPadding = 0
    ZeroPadding = 1
    CachedPadding = 2

    def __init__(self, padding, max_batch_size=256):
        super().__init__()
        self.initialized = 0
        self.padding = padding
        self.max_batch_size = max_batch_size

    @torch.jit.unused
    @torch.no_grad()
    def init_cache(self, x):
        b, c, _ = x.shape
        self.register_buffer(
            "pad",
            torch.zeros(self.max_batch_size, c, self.padding).to(x),
            persistent=False
        )
        self.initialized += 1

    def forward(self, x, paddingmode=CachedPadding):
        if not self.initialized:
            self.init_cache(x)

        if self.padding:
            match paddingmode:
                case CachedPadding1d.ZeroPadding:
                    x = torch.nn.functional.pad(x, (self.padding, 0))
                case CachedPadding1d.CachedPadding:
                    x = torch.cat([self.pad[:x.shape[0]], x], -1)
                case CachedPadding1d.NoPadding:
                    pass
            self.pad[:x.shape[0]].copy_(x[..., -self.padding:].detach())

        return x

class CachedConv1d(nn.Module):
    """
    Implementation of a Conv1d **with stride 1** operation with cached padding (same).
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        assert kwargs.get("padding", 0) == 0 and kwargs.get("stride", 1) == 1
        self.conv = nn.Conv1d(*args, **kwargs)

        padding = (self.conv.kernel_size[0] - 1) * self.conv.dilation[0]

        self.cache = CachedPadding1d(padding)

    def forward(self, x, paddingmode=CachedPadding1d.CachedPadding):
        x = self.cache(x, paddingmode=paddingmode)
        return self.conv(
            x,
        )
    
class LinearInterp(torch.nn.Module):
    def __init__(self, channels, size_factor):
        super().__init__()
        self.convt = nn.ConvTranspose1d(channels, channels, 2 * size_factor, stride=size_factor, padding=size_factor, groups=channels, bias=False)
        self.convt.weight.data[:, :, :] = torch.concat(
            [
                torch.arange(0, size_factor) / size_factor,
                1 - torch.arange(0, size_factor) / size_factor
            ], 0).view(1, 1, 2 * size_factor)
        self.convt.weight.requires_grad = False

    def forward(self, *args, **kwargs):
        return self.convt(*args, **kwargs)

def film_op(x, ab, channels):
    a, b = torch.split(ab, channels, dim=-1)
    return a[..., None] * x + b[..., None]

def lookback_length(kernel_size_list, dilation_list, stride_list, pooling_list):
    n = 1
    for k, d, s, p in list(zip(kernel_size_list, dilation_list, stride_list, pooling_list, strict=True))[::-1]:
        n = 1 + d * (k - 1) + s * (p * n - 1)
    return n

class LookbackNetwork(torch.nn.Module):

    def __init__(self, in_out_model_channels, state_sizes, linear_end: bool, cond_dim, out_channels_list, kernel_size_list, dilation_list, stride_list, pooling_list, pooling_type, film_hidden_neurons_list):
        super().__init__()
        assert len(out_channels_list) == len(kernel_size_list) == len(dilation_list) == len(stride_list)

        self.in_out_model_channels = in_out_model_channels
        self.state_sizes = state_sizes
        self.cond_dim = cond_dim
        self.film_hidden_neurons_list = film_hidden_neurons_list
        self.linear_end = linear_end
        self.out_channels_list = out_channels_list
        self.kernel_size_list = kernel_size_list
        self.dilation_list = dilation_list
        self.stride_list = stride_list
        self.pooling_list = pooling_list
        assert pooling_type in ("max", "avg")
        self.pooling_type = pooling_type

        self.expected_input_length = self.needed_samples
        _L = self.expected_input_length
        for _d, _k, _s, _p in zip(dilation_list,  kernel_size_list, stride_list, pooling_list):
            _L = floor((_L + 2 * 0 - _d * (_k - 1) - 1) / _s + 1)   # conv
            _L = floor((_L + 2 * 0 - 1 * (_p - 1) - 1) / _p + 1)    # pool
        assert _L == 1

        if linear_end:
            self.linear_end_module = nn.Linear(out_channels_list[-1], sum(state_sizes))
        else:
            assert self.out_channels_list[-1] == sum(state_sizes)

        self.convs = nn.ModuleList([
            torch.nn.Conv1d(_i, _o, _k, stride=_s, dilation=_d) 
            for _i, _o, _k, _s, _d in zip(
                [in_out_model_channels] + out_channels_list[:-1],
                out_channels_list,
                kernel_size_list,
                stride_list,
                dilation_list
            )
        ])
        if cond_dim > 0:
            assert isinstance(film_hidden_neurons_list, Sequence)
            self.films = []
            for c in out_channels_list:
                layers = []
                if len(film_hidden_neurons_list) == 0 or not film_hidden_neurons_list[0]:
                    h = cond_dim
                else:
                    for i, h in enumerate(film_hidden_neurons_list):
                        layers.append(nn.Sequential(
                            nn.Linear(cond_dim if i == 0 else film_hidden_neurons_list[i - 1], h),
                            nn.PReLU(h)
                        ))
                self.films.append(nn.Sequential(*layers, nn.Linear(h, 2 * c)))
            self.films = nn.ModuleList(self.films)
        else: 
            self.films = len(out_channels_list) * [None]
        self.acts = nn.ModuleList([
            torch.nn.PReLU(_o) for _o in out_channels_list
        ])
        self.pools = torch.nn.ModuleList([
            torch.nn.MaxPool1d(bk)
            if self.pooling_type == "max" else
            torch.nn.AvgPool1d(bk)
            for bk in self.pooling_list
        ])

    @property
    def needed_samples(self):
        n = 1
        for k, d, s, p in list(zip(self.kernel_size_list, self.dilation_list, self.stride_list, self.pooling_list, strict=True))[::-1]:
            n = 1 + d * (k - 1) + s * (p * n - 1)
        return n

    def forward(self, lookback, p):
        assert lookback.shape[2] == self.expected_input_length
        assert lookback.shape[1] == self.convs[0].in_channels, "the lookback tensor should have input_channels+sidechain_channels+input_channels channels"
        v = lookback
    
        for conv, act, film, pool, c in zip(self.convs, self.acts, self.films, self.pools, self.out_channels_list, strict=True):
            v = conv(v)
            v = act(v)
            if film is not None:
                ab = film(p)
                v = film_op(v, ab, c)
            v = pool(v)

        # v has shape (B, out_channels_list[-1], 1)
        v = v.squeeze(-1)

        if self.linear_end:
            v = self.linear_end_module(v)

        return v
    
class TurboCachedTFiLM(torch.nn.Module):
    """
    maxpool
    film (optional)
    turboRNN
    upsample
    """
    def __init__(self, in_channels, block_size, cond_dim, pooling_type, rnn_cell, rnn_hidden_size, film_hidden_neurons_list):
        super().__init__()
        self.in_channels = in_channels
        self.block_size = block_size
        self.rnn_cell = rnn_cell
        self.rnn_hidden_size = rnn_hidden_size
        self.cond_dim = cond_dim
        self.film_hidden_neurons_list = film_hidden_neurons_list
        assert pooling_type in ("max", "avg")
        self.pooling_type = pooling_type
        self.maxpool = nn.MaxPool1d(block_size) if pooling_type == "max" else nn.AvgPool1d(block_size)
        self.rnn = RNN(in_channels, rnn_hidden_size, rnn_cell)
        self.state = None
        self.upsample = LinearInterp(rnn_hidden_size, block_size)
        self.cache = CachedPadding1d(1)

        self.film_nn = None

        if cond_dim > 0:
            if isinstance(film_hidden_neurons_list, Sequence):
                self.film_nn = nn.ModuleList()
                if len(film_hidden_neurons_list) == 0 or not film_hidden_neurons_list[0]:
                    h = cond_dim
                else:
                    for i, h in enumerate(film_hidden_neurons_list):
                        self.film_nn.append(nn.Sequential(
                            nn.Linear(cond_dim if i == 0 else film_hidden_neurons_list[i - 1], h),
                            nn.PReLU(h)
                        ))

                self.film_nn.append(nn.Linear(h, 2 * self.in_channels))
                self.film_nn = nn.Sequential(*self.film_nn)
    
    def forward(self, z, p, state=None, paddingmode=0):
        """ z [B, C, L] p [B, P] """
        assert z.shape[2] % self.block_size == 0

        if state is None:
            if self.state is None:
                self.state = torch.zeros((z.size(0), (2 if self.rnn_cell == "LSTM" else 1) * self.rnn_hidden_size), device=z.device, requires_grad=True).detach()
        else:
            self.state = state

        z2 = self.maxpool(z)
        if self.film_nn is not None:
            ab = self.film_nn(p)
            z2 = film_op(z2, ab, self.in_channels)
        z2, self.state = self.rnn(z2, self.state)
        z2 = self.cache(z2, paddingmode=paddingmode)
        z2 = self.upsample(z2)
        return z2
    
class Model(nn.Module):
    def __init__(
            self, /, input_channels, sidechain_channels, n_params,
            s_channels_list, s_kernel_size_list, s_dilation_list, tfilm_block_size, tfilm_pooling_type,
            cond_size, film_hidden_neurons_list, rnn_cell, rnn_hidden_size,
            lbnet, lbnet_dropout, lbnet_dropout_full_items,
            lbnet_linear_end, lbnet_out_channels_list, lbnet_kernel_size_list, lbnet_dilation_list, lbnet_stride_list, lbnet_pooling_list, lbnet_pooling_type,
            lbnet_film_hidden_neurons_list,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.input_channels = input_channels
        self.sidechain_channels = sidechain_channels
        self.n_params = n_params
        self.tfilm_block_size = tfilm_block_size
        self.lbnet_dropout = lbnet_dropout
        self.lbnet_dropout_full_items = lbnet_dropout_full_items

        self.sc = sidechain_channels > 0
        self.scc = sidechain_channels if self.sc else input_channels

        self.s_convs = nn.ModuleList(
            CachedConv1d(self.scc if i==0 else s_channels_list[i-1], c, k, dilation=d, padding=0)
            for i, (c, k, d) in enumerate(zip(s_channels_list, s_kernel_size_list, s_dilation_list))
        )

        self.s_conv11f = nn.ModuleList(
            nn.Conv1d(rnn_hidden_size, 2 * input_channels, 1)
            for c in s_channels_list
        )

        self.s_conv11s = nn.ModuleList(
            nn.Conv1d(rnn_hidden_size, 2 * c, 1)
            for c in s_channels_list
        )

        self.condnet = nn.Sequential(
            nn.Linear(n_params, cond_size),
            nn.PReLU(cond_size),
            nn.Linear(cond_size, cond_size),
            nn.PReLU(cond_size),
            nn.Linear(cond_size, cond_size),
            nn.PReLU(cond_size),
        ) if n_params > 0 and cond_size is not None else nn.Identity()

        cond_dim = cond_size if cond_size is not None else n_params

        self.s_tfilms = nn.ModuleList(
            TurboCachedTFiLM(c, tfilm_block_size, cond_dim, tfilm_pooling_type, rnn_cell, rnn_hidden_size, film_hidden_neurons_list)
            for c in s_channels_list
        )

        self.s_acts = nn.ModuleList(
            nn.PReLU(conv.conv.out_channels) for conv in self.s_convs
        )

        self.lookback_net = LookbackNetwork(
            input_channels + sidechain_channels + input_channels, 
            len(s_channels_list) * [(2 if rnn_cell == "LSTM" else 1) * rnn_hidden_size],
            lbnet_linear_end,
            cond_dim,
            lbnet_out_channels_list,
            lbnet_kernel_size_list,
            lbnet_dilation_list,
            lbnet_stride_list,
            lbnet_pooling_list,
            lbnet_pooling_type,
            lbnet_film_hidden_neurons_list
        )   if lbnet else None

        if lbnet_dropout > 0 and not lbnet_dropout_full_items:
            self.lbnet_dropout_layer = nn.Dropout(p=lbnet_dropout)
        else:
            self.lbnet_dropout_layer = None
    
    def reset_caches(self):
        for module in self.modules():
            if isinstance(module, CachedPadding1d):
                module.initialized = False

    def reset_states(self):
        for tfilm in self.s_tfilms:
            tfilm.state = None

    def detach_states(self):
        for tfilm in self.s_tfilms:
            tfilm.state = tfilm.state.detach()
    
    def calc_indices(self, target_length):
        """
        returns (
            expected input length,
            starting index for modulation path,
            starting index for audio path,
            starting index for lbnet,
            cropping size list for modulation path,
            cropping size list for audio path
        )
        """
        L = target_length
        N = len(self.s_convs)
        P = self.tfilm_block_size
        sa = [
            0
            for conv in self.s_convs
        ]
        sm = [
            (conv.conv.kernel_size[0] - 1) * conv.conv.dilation[0]
            for conv in self.s_convs
        ]
        km = [0 for _ in range(N)]
        cmm = [0 for _ in range(N)]
        cma = [0 for _ in range(N)]
        km[-1] = 1 + ceil(L / P)
        cmm[-1] = - L + (km[-1] - 1) * P
        cma[-1] = - L + (km[-1] - 1) * P
        for j in range(N - 2, -1, -1):
            sum_sm = sum(sm[j + 1 :])
            sum_sa = sum(sa[j + 1 :])
            sum_cmm = sum(cmm[j + 1 :])
            km[j] = max(N - j + ceil((L + sum_sm + sum_cmm) / P), 1 + ceil((L + sum_sa) / P))
            cmm[j] = - L + (km[j] - (N - j)) * P - sum_sm - sum_cmm
            cma[j] = - L - sum_sa + (km[j] - 1) * P
        tm0 = target_length - sm[0] - km[0] * P
        ta0 = - sum(sa)
        tl0 = - self.lookback_net.expected_input_length if self.lookback_net is not None else 0
        input_length = max(target_length - tm0, target_length - ta0, target_length - tl0)
        i0m = input_length - (target_length - tm0)
        i0a = input_length - (target_length - ta0)
        i0l = input_length - (target_length - tl0)
        return input_length, i0m, i0a, i0l, cmm, cma
    
    def set_target_length(self, target_length):
        self.input_length, self.i0m, self.i0a, self.i0l, self.cmm, self.cma = self.calc_indices(target_length)

    def forward(self, x, p, y_true=None, *, use_spn=False, paddingmode=CachedPadding1d.NoPadding):
        """
        x: tensor of shape [B, C, Lin] (batch size, input_channels + sidechain_channels, input length)
        p: tensor of shape [B, P] (batch size, number of parameters)

        output:
            y: tensor of shape [B, C, Lout]
        """

        # separate input and sidechain
        if self.sc:
            s = x[:, self.input_channels:]
            x = x[:, :self.input_channels]
            assert x.shape[1] == self.input_channels
            assert s.shape[1] == self.sidechain_channels
        else:
            s = x
        
        N = len(self.s_convs)

        if self.n_params > 0:
            zp = self.condnet(p)
        else:
            zp = p

        lbstate = None
        if use_spn and self.lookback_net is not None:
            _il0, _il1 = self.i0l, self.i0l + self.lookback_net.expected_input_length
            lookback = torch.concat((
                    x[:, :, _il0 : _il1],
                    y_true[:, :, _il0 : _il1]
                ) if not self.sc else (
                    x[:, :, _il0 : _il1],
                    s[:, :, _il0 : _il1],
                    y_true[:, :, _il0 : _il1]
                ), 
                dim=1
            )
            lbstate = self.lookback_net(lookback, zp)
            # apply dropout
            if self.lbnet_dropout > 0:
                if self.lbnet_dropout_full_items:
                    lbstate[:int(self.lbnet_dropout * lbstate.size(0))] = 0.
                else:
                    lbstate = self.lbnet_dropout_layer(lbstate)

        if paddingmode == CachedPadding1d.NoPadding:
            vm = s[:, :, self.i0m:]
            va = x[:, :, self.i0a:]
        else:
            vm = s
            va = x

        for j in range(N):
            vm = self.s_convs[j](vm, paddingmode=paddingmode)
            vm = self.s_acts[j](vm)
            if lbstate is None:
                initial_state = None
            else:
                state_size = (2 if self.hparams.rnn_cell == "LSTM" else 1) * self.hparams.rnn_hidden_size
                initial_state = lbstate[:, j * state_size : (j + 1) * state_size]
            tfilm = self.s_tfilms[j](vm, zp, state=initial_state, paddingmode=paddingmode)  # on obtient alors (2 * input_channels) channels
            if paddingmode == CachedPadding1d.NoPadding:
                tfilm_f = tfilm[:, :, self.cma[j]:]
            else:
                tfilm_f = tfilm
            ksi_mul, ksi_plus = torch.split(self.s_conv11f[j](tfilm_f), self.hparams.input_channels, dim=1)

            va = va * ksi_mul + ksi_plus

            if j < N-1 :
                tfilm_s_mul, tfilm_s_plus = torch.split(self.s_conv11s[j](tfilm), self.s_convs[j].conv.out_channels, dim=1)
                if paddingmode == CachedPadding1d.NoPadding:
                    vm = vm[:, :, self.tfilm_block_size:]
                vm = vm * tfilm_s_mul + tfilm_s_plus
                if paddingmode == CachedPadding1d.NoPadding:
                    vm = vm[:, :, self.cmm[j]:]
        
        y = va

        return y