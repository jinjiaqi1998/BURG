import torch
from torch import nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Gaussian(torch.distributions.Distribution):
    def __init__(self):
        super().__init__()
        self.normal = torch.distributions.Normal(0, 1)
        self.log_2pi = torch.log(torch.tensor(2 * torch.pi)).to(device)

    def log_prob(self, x):
        return -0.5 * (self.log_2pi + x ** 2)

    def sample(self, size):
        return self.normal.sample(size).to(device)


class Coupling(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_layer, odd_flag):
        super().__init__()
        self.odd_flag = odd_flag % 2
        self.half_dim = input_dim // 2

        self.net = nn.Sequential(
            nn.Linear(self.half_dim, hidden_dim),
            nn.ReLU(),
            *[nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ) for _ in range(hidden_layer - 1)]
        )

        self.scale_net = nn.Sequential(
            nn.Linear(hidden_dim, self.half_dim),
            nn.Tanh()
        )
        self.trans_net = nn.Linear(hidden_dim, self.half_dim)

    def forward(self, x, reverse):
        if self.odd_flag:
            x1, x2 = x[:, :self.half_dim], x[:, self.half_dim:]
        else:
            x2, x1 = x[:, :self.half_dim], x[:, self.half_dim:]

        h = self.net(x2)
        scale = self.scale_net(h)
        trans = self.trans_net(h)

        if reverse:
            x1 = (x1 - trans) * torch.exp(-scale)
        else:
            x1 = x1 * torch.exp(scale) + trans

        if self.odd_flag:
            return torch.cat((x1, x2), dim=1)
        else:
            return torch.cat((x2, x1), dim=1)


class Scale(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.s = nn.Parameter(torch.zeros(1, input_dim))

    def forward(self, x, reverse):
        if reverse:
            result = torch.exp(-self.s) * x
        else:
            result = torch.exp(self.s) * x
        return result, self.s


class Single_Flow(nn.Module):
    def __init__(self, input_dim=784, coupling_num=4, hidden_dim=1000, hidden_layers=5):
        super().__init__()
        self.prior = Gaussian()
        self.coupling = nn.ModuleList([
            Coupling(input_dim, hidden_dim, hidden_layers, odd_flag=i + 1)
            for i in range(coupling_num)
        ])
        self.scale = Scale(input_dim)

    def forward(self, x, reverse=False):
        for coupling in self.coupling:
            x = coupling(x, reverse)
        h, s = self.scale(x, reverse)

        log_det = torch.sum(s, dim=1)
        log_prob = self.prior.log_prob(h)
        log_prob = torch.sum(log_prob, dim=1)
        loss = -(log_prob + log_det)
        return loss, h

    def reverse_generate(self, z):
        h, s = self.scale(z, True)
        for coupling in reversed(self.coupling):
            h = coupling(h, True)
        return h


class FlowAllViews(nn.Module):
    def __init__(self, input_dim, coupling_num, hidden_dim, hidden_layers, view_num):
        super(FlowAllViews, self).__init__()
        self.view_num = view_num
        self.flows = nn.ModuleList([
            Single_Flow(input_dim, coupling_num, hidden_dim, hidden_layers)
            for _ in range(view_num)
        ])

    def get_ztilde_com(self, z_sel_spec):
        total_loss = 0
        zs_tilde_spec = []
        for v in range(len(z_sel_spec)):
            loss, h = self.flows[v](z_sel_spec[v])
            zs_tilde_spec.append(h)
            total_loss = total_loss + torch.mean(loss)
        normal_factor = 1 / torch.sqrt(torch.tensor(len(z_sel_spec)).to(device))
        z_tilde_com = sum(t * normal_factor for t in zs_tilde_spec)
        return z_tilde_com, total_loss

    def flow_SingleViews(self, zs, miss_vecs):
        loss_list_flow = []
        for view_id, (z, miss_vec) in enumerate(zip(zs, miss_vecs)):
            non_miss_ids = torch.where(miss_vec == 1)[0]
            if len(non_miss_ids) > 0:
                z_non_miss = z[non_miss_ids]
                loss, _ = self.flows[view_id](z_non_miss)
                loss_list_flow.append(torch.mean(loss) / len(non_miss_ids))
        total_loss = sum(loss_list_flow) / len(loss_list_flow)
        return total_loss

    def recover_SingleView(self, z_com, recover_view_id):
        recovered_z = self.flows[recover_view_id].reverse_generate(z_com)
        return recovered_z




