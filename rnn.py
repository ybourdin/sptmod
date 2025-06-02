"""
Functional versions of LSTM and GRU for hypernetworking
"""
import torch
from torch import Tensor as T
from torch.func import functional_call
from math import prod
from typing import Literal

class RNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, cell: Literal["LSTM", "GRU"]):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell = cell

        if cell == "LSTM":
            self.rnn = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=False)
        elif cell == "GRU":
            self.rnn = torch.nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=False)
        else:
            raise ValueError(f"Invalid cell type: {cell}")

    def forward(self, x: T, state: T, weights: T | None = None) -> tuple[T, T]:
        """
        Expects:
        - x (B, C, L)
        - state (B, 2 * H) if LSTM else (B, H)
        - weights (B, 4 * H * (C + H + 2)) if LSTM else (B, 3 * H * (C + H + 2))
        """
        C = x.size(1)
        x = x.permute(2, 0, 1)  # -> (L, B, C)
        if weights is not None and self.cell == "LSTM":
            H = state.size(1) // 2
            assert weights.size(1) == 4 * H * (C + H + 2)
            lstm_shapes = {
                "weight_ih_l0": (4 * H, C),
                "weight_hh_l0": (4 * H, H),
                "bias_ih_l0": (4 * H,),
                "bias_hh_l0": (4 * H,),
            }
            out_items, h0_items, c0_items = [], [], []
            weights_splits = torch.split(weights, [prod(s) for s in lstm_shapes.values()], dim=-1)
            for i in range(x.shape[1]):
                xi = x[:, i : i+1]
                statei = (
                    state[i : i+1, :H].unsqueeze(0), 
                    state[i : i+1, H:].unsqueeze(0)
                )
                lstm_params = {
                    param_name: split[i].reshape(shape).contiguous()
                    for (param_name, shape), split in zip(lstm_shapes.items(), weights_splits, strict=True)
                }
                out, state_item = functional_call(self.rnn, lstm_params, (xi, statei), strict=True)
                out_items.append(out)
                h0_items.append(state_item[0])
                c0_items.append(state_item[1])
            state = torch.concat((
                torch.concat(h0_items, dim=1).squeeze(0), 
                torch.concat(c0_items, dim=1).squeeze(0)
                ), dim=-1
            )
            outputs = torch.concat(out_items, dim=1)
        elif weights is not None and self.cell == "GRU":
            H = state.size(1)
            assert weights.size(1) == 3 * H * (C + H + 2)
            gru_shapes = {
                "weight_ih_l0": (3 * H, C),
                "weight_hh_l0": (3 * H, H),
                "bias_ih_l0": (3 * H,),
                "bias_hh_l0": (3 * H,),
            }
            out_items, h0_items = [], []
            weights_splits = torch.split(weights, [prod(s) for s in gru_shapes.values()], dim=-1)
            for i in range(x.shape[1]):
                xi = x[:, i : i+1]
                statei = state[i : i+1, :H].unsqueeze(0)
                gru_params = {
                    param_name: split[i].reshape(shape).contiguous()
                    for (param_name, shape), split in zip(gru_shapes.items(), weights_splits, strict=True)
                }
                out, state_item = functional_call(self.rnn, gru_params, (xi, statei), strict=True)
                out_items.append(out)
                h0_items.append(state_item)
            state = torch.concat(h0_items, dim=1).squeeze(0)
            outputs = torch.concat(out_items, dim=1)
        elif self.cell == "LSTM":
            state = state.unsqueeze(0).split(state.size(1) // 2, dim=-1)
            outputs, state = self.rnn(x, (state[0].contiguous(), state[1].contiguous()))
            state = torch.concat(state, dim=-1).squeeze(0)
        elif self.cell == "GRU":
            state = state.unsqueeze(0)
            outputs, state = self.rnn(x, state.contiguous())
            state = state.squeeze(0)
        y = outputs.permute(1, 2, 0)
        return y, state

if __name__ == "__main__":
    # test if TurboRNN works for cell="LSTM" and cell="GRU" by checking their output shapes with random inputs

    B, C, L, H = 30, 31, 16384, 32
    x = torch.randn(B, C, L)

    for cell in ["LSTM", "GRU"]:
        h0 = torch.randn(B, 2*H if cell == "LSTM" else H)
        weights = torch.randn(B, 4 * H * (C + H + 2) if cell == "LSTM" else 3 * H * (C + H + 2))
        model = RNN(input_size=C, hidden_size=H, cell=cell)
        for w in (weights, None):
            y, state = model(x, h0, w)
            assert y.shape == (B, H, L)
            if cell == "LSTM":
                assert state.shape == (B, 2 * H)
            else:
                assert state.shape == (B, H)