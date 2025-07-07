import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim, z_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 300),
            nn.ReLU(),
            nn.Linear(300, z_dim),
            nn.BatchNorm1d(z_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, input_dim, z_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 300),
            nn.ReLU(),
            nn.Linear(300, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim),
        )

    def forward(self, x):
        return self.decoder(x)


class AutoEncoders(nn.Module):
    def __init__(self, view_num, view_dims, z_dim):
        super(AutoEncoders, self).__init__()
        self.view_num = view_num
        self.view_dims = view_dims
        self.encoders = nn.ModuleList([
            Encoder(view_dims[v], z_dim)
            for v in range(view_num)
        ])
        self.decoders = nn.ModuleList([
            Decoder(view_dims[v], z_dim)
            for v in range(view_num)
        ])

    def forward(self, xs, miss_vecs):
        z_spec, xr_spec, xs_com = [], [], []
        for v in range(self.view_num):
            z = self.encoders[v](xs[v])
            xr = self.decoders[v](z)
            z_spec.append((torch.mul(z.t(), miss_vecs[v]).t()).float())
            xr_spec.append((torch.mul(xr.t(), miss_vecs[v]).t()).float())
            xs_com.append((torch.mul(xs[v].t(), miss_vecs[v]).t()).float())
        z_com = (torch.mul(sum(z_spec).t(), (1 / sum(miss_vecs))).t()).float()  # [batchsize, z_dim]
        xr_com = []
        for v in range(self.view_num):
            xr_com.append((torch.mul(self.decoders[v](z_com).t(), miss_vecs[v]).t()).float())

        return xr_spec, xr_com, z_spec, z_com, xs_com


    def forward_commonZ(self, xs, miss_vecs):
        z_spec = []
        for v in range(self.view_num):
            z_spec.append((torch.mul((self.encoders[v](xs[v])).t(), miss_vecs[v]).t()).float())
        z_com = (torch.mul(sum(z_spec).t(), (1 / sum(miss_vecs))).t()).float()
        return z_com


    def decode_for_recover(self, zr_cur_sel_recover, cur_miss_id):
        xr_cur_sel_recover = self.decoders[cur_miss_id](zr_cur_sel_recover)
        return xr_cur_sel_recover


    def forward_singleZ(self, xs):
        z_spec = []
        for v in range(self.view_num):
            z_spec.append(self.encoders[v](xs[v]))
        return z_spec


    def forward_cur_com(self, zs, com_view_ids, cur_miss_vec, cur_com_id):
        z_sel_spec = []
        for v in com_view_ids:
            z_sel_spec.append((torch.mul(zs[v].t(), cur_miss_vec).t())[cur_com_id].float())
        z_sel_common = (torch.mul(sum(z_sel_spec).t(), (1 / len(com_view_ids))).t()).float()
        return z_sel_common, z_sel_spec




