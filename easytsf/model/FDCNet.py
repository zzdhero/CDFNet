import torch
import torch.nn as nn

from easytsf.model.kanlayer import KANInterface


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()

        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)

        elif mode == 'denorm':
            x = self._denormalize(x)

        else:
            raise NotImplementedError

        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias

        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + self.mean

        return x


class MultiKANLayer(nn.Module):
    """
    [("TaylorKAN", 4), ("TaylorKAN", 4), ("WavKAN", None), ("WavKAN", None)]

    """

    def __init__(self, in_len, out_len, layer_config):
        super(MultiKANLayer, self).__init__()
        self.layers = nn.ModuleList([])
        for layer_type, layer_param in layer_config:
            if layer_type == "WavKAN":
                self.layers.append(KANInterface(in_len, out_len, layer_type="WavKAN"))
            elif layer_type == "KAN":
                self.layers.append(KANInterface(in_len, out_len, layer_type="KAN", n_grid=layer_param))
            elif layer_type == "TaylorKAN":
                self.layers.append(KANInterface(in_len, out_len, layer_type="TaylorKAN", order=layer_param))
            elif layer_type == "JacobiKAN":
                self.layers.append(KANInterface(in_len, out_len, layer_type="JacobiKAN", degree=layer_param))
            elif layer_type == "Linear":
                self.layers.append(torch.Linear(in_len, out_len))
            else:
                raise NotImplemented


class FDCNet(nn.Module):
    def __init__(self, hist_len, pred_len, var_num, use_repr, hidden_dim, layer_config, construction_loss_weight,
                 dropout=0.1):
        super(FDCNet, self).__init__()
        self.hist_len = hist_len
        self.pred_len = pred_len

        self.alpha = construction_loss_weight

        self.dropout = nn.Dropout(dropout)
        self.rev = RevIN(var_num)
        self.use_repr = use_repr

        self.deco_layers = MultiKANLayer(hist_len, hist_len, layer_config=layer_config)
        if use_repr:
            self.repr_layers = MultiKANLayer(hist_len, hidden_dim, layer_config=layer_config)
            self.pred_layers = MultiKANLayer(hidden_dim, pred_len, layer_config=layer_config)
        else:
            self.pred_layers = MultiKANLayer(hist_len, pred_len, layer_config=layer_config)

        self.construction_loss_fn = nn.MSELoss(reduction='mean')

    def forward(self, var_x, marker_x):
        var_x = self.rev(var_x, 'norm').permute(0, 2, 1)
        var_x = self.dropout(var_x)

        # decomposition
        deco_results = []
        for d_layer in self.deco_layers.layers:
            deco_results.append(d_layer(var_x))

        # representation
        if self.use_repr:
            repr_results = []
            for i, r_layer in enumerate(self.repr_layers.layers):
                repr_results.append(r_layer(deco_results[i]))
        else:
            repr_results = deco_results

        # forecasting
        pred_results = []
        for i, p_layer in enumerate(self.pred_layers.layers):
            pred_results.append(p_layer(repr_results[i]))
        pred = torch.sum(torch.stack(pred_results, dim=-1), dim=-1).permute(0, 2, 1)
        pred = self.rev(pred, 'denorm')

        # construction
        construction = torch.sum(torch.stack(deco_results, dim=-1), dim=-1)
        c_loss = self.construction_loss_fn(construction, var_x)
        return pred, c_loss * self.alpha
