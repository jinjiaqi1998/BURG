import torch
import torch.nn as nn
from itertools import combinations
import torch.nn.functional as F
from AENets import *
from FlowNets import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class BURG(nn.Module):
    def __init__(self, args):
        super(BURG, self).__init__()
        self.args = args
        self.AEs = AutoEncoders(self.args.view_num, self.args.view_dims, self.args.z_dim)
        self.Flows = FlowAllViews(
            input_dim=self.args.z_dim,
            coupling_num=self.args.coupling_num,
            hidden_dim=self.args.hidden_dim,
            hidden_layers=self.args.hidden_layers,
            view_num=self.args.view_num
        )
        self.criterionAE = torch.nn.MSELoss()

    def _init_weights(self):
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        self.autoencoder.apply(init_weights)
        self.flow.apply(init_weights)

    def get_SingleZs(self, xs):
        return self.AEs.forward_singleZ(xs)

    def get_commonZ(self, xs, miss_vecs):
        return self.AEs.forward_commonZ(xs, miss_vecs)

    def train_AE(self, xs, miss_vecs):
        xr_spec, xr_com, z_spec, z_com, xs_com = self.AEs.forward(xs, miss_vecs)
        loss_list_spec_recon, loss_list_com_recon = [], []
        for v in range(self.args.view_num):
            loss_list_spec_recon.append(self.criterionAE(xr_spec[v], xs_com[v]))
            loss_list_com_recon.append(self.criterionAE(xr_com[v], xs_com[v]))
        loss_spec_recon = sum(loss_list_spec_recon)
        loss_com_recon = sum(loss_list_com_recon)
        loss_Recon = loss_spec_recon + self.args.para_recon * loss_com_recon
        return loss_Recon

    def train_Flow(self, xs, miss_vecs):
        z_spec = self.AEs.forward_singleZ(xs)
        loss_Flow = self.Flows.flow_SingleViews(z_spec, miss_vecs)
        return loss_Flow


    def train_Recover_observe(self, xs, miss_vecs):
        xr_spec, xr_com, z_spec, z_com, xs_com = self.AEs.forward(xs, miss_vecs)
        complete_samples = torch.ones_like(miss_vecs[0])
        for v in range(self.args.view_num):
            complete_samples = torch.mul(complete_samples, miss_vecs[v])
        complete_ids = torch.where(complete_samples > 0)[0]
        z_com_all = []
        x_com_all = []
        for v in range(self.args.view_num):
            z_com_all.append(z_spec[v][complete_ids])
            x_com_all.append(xs[v][complete_ids])

        loss_list_flow, loss_list_recover = [], []
        for miss_view in range(self.args.view_num):
            z_complete_spec = [z_com_all[v] for v in range(self.args.view_num) if v != miss_view]
            z_tilde_com, cur_loss_flow = self.Flows.get_ztilde_com(z_complete_spec)
            zr_recover = self.Flows.recover_SingleView(z_tilde_com, miss_view)
            z_target = z_com_all[miss_view]
            cur_loss_recover = self.criterionAE(zr_recover, z_target)

            loss_list_flow.append(cur_loss_flow)
            loss_list_recover.append(cur_loss_recover)

        loss_Flow = sum(loss_list_flow) / len(loss_list_flow)
        loss_Recover = sum(loss_list_recover) / len(loss_list_recover)

        return loss_Flow, loss_Recover


    def train_Recover_comb(self, xs, miss_vecs):
        xr_spec, xr_com, z_spec, z_com, xs_com = self.AEs.forward(xs, miss_vecs)
        loss_list_flow, loss_list_recover_recon = [], []
        all_view_ids = [v for v in range(self.args.view_num)]
        count = 0
        for r in range(1, self.args.view_num):
            combs = combinations(range(1, self.args.view_num + 1), r)
            for ori_comb in combs:
                com_view_ids = [x - 1 for x in list(ori_comb)]
                miss_view_ids = [ele for ele in all_view_ids if ele not in com_view_ids]
                for cur_miss_id in miss_view_ids:
                    count += 1
                    cur_miss_vec = miss_vecs[cur_miss_id]
                    for v in com_view_ids:
                        cur_miss_vec = torch.mul(cur_miss_vec, miss_vecs[v])
                    cur_com_id = torch.where(cur_miss_vec > 0)[0]
                    z_sel_common, z_sel_spec = self.AEs.forward_cur_com(z_spec, com_view_ids, cur_miss_vec, cur_com_id)
                    z_tilde_com, cur_loss_flow = self.Flows.get_ztilde_com(z_sel_spec)
                    zr_cur_sel_recover = self.Flows.recover_SingleView(z_tilde_com, cur_miss_id)
                    xr_cur_sel_recover = self.AEs.decode_for_recover(zr_cur_sel_recover, cur_miss_id)
                    z_cur_miss_sel = z_spec[cur_miss_id][cur_com_id]
                    x_cur_miss_sel = xs[cur_miss_id][cur_com_id]
                    loss_list_recover_recon.append(self.criterionAE(zr_cur_sel_recover, z_cur_miss_sel) + self.criterionAE(xr_cur_sel_recover, x_cur_miss_sel))
                    loss_list_flow.append(cur_loss_flow)

        loss_Flow = sum(loss_list_flow) / len(loss_list_flow)
        loss_Recover = sum(loss_list_recover_recon) / len(loss_list_recover_recon)

        return loss_Flow, loss_Recover


    def train_nac_vec(self, xs, miss_vecs):
        xr_spec, xr_com, z_spec, z_com, xs_com = self.AEs.forward(xs, miss_vecs)
        cur_N = xs[0].shape[0]
        z_tilde_com, _ = self.Flows.get_ztilde_com(z_spec)
        miss_wInf_vec = [torch.where(mv == 0, float('inf'), mv) for mv in miss_vecs]
        diag_mask = torch.eye(cur_N, device=xs[0].device) * float('inf')
        Sim_wInf = [
            torch.mul(
                torch.cdist(z_spec[v], z_spec[v]) + diag_mask,
                torch.outer(miss_wInf_vec[v], miss_wInf_vec[v])
            ) for v in range(self.args.view_num)
        ]

        loss_list_nac = []
        missing_samples_all = [(miss_vecs[v] == 0).nonzero().squeeze() for v in range(self.args.view_num)]
        missing_samples_all = [ms.unsqueeze(0) if ms.dim() == 0 and ms.numel() > 0 else ms
                               for ms in missing_samples_all]
        z_recovered_all = torch.stack([
            self.Flows.recover_SingleView(z_tilde_com, v)
            for v in range(self.args.view_num)
        ])
        for v in range(self.args.view_num):
            missing_samples = missing_samples_all[v]
            if missing_samples.numel() == 0:
                continue

            temp_Sim = torch.zeros((len(missing_samples), self.args.view_num, cur_N), device=xs[0].device)
            for j in range(self.args.view_num):
                temp_Sim[:, j] = torch.mul(Sim_wInf[j][missing_samples], miss_wInf_vec[v])
            temp_Sim[:, v] = float('inf')
            min_indices = torch.argmin(temp_Sim.reshape(len(missing_samples), -1), dim=1)
            min_cols = min_indices % cur_N
            anchors = z_recovered_all[v][missing_samples]
            targets = z_spec[v][min_cols]
            dists = torch.cdist(
                anchors.unsqueeze(1),
                targets.unsqueeze(1),
                p=2
            ).squeeze()
            loss_list_nac.append(dists.mean())

        loss_NAC = torch.mean(torch.stack(loss_list_nac)) if loss_list_nac else torch.tensor(0.0).to(xs[0].device)

        return loss_NAC


    def train_pc_vec(self, xs, miss_vecs, centroids_com):
        xr_spec, xr_com, z_spec, z_com, xs_com = self.AEs.forward(xs, miss_vecs)
        z_tilde_com, _ = self.Flows.get_ztilde_com(z_spec)
        cur_N = xs[0].shape[0]
        miss_stack = torch.stack(miss_vecs).unsqueeze(-1)
        p_views = []
        for v in range(self.args.view_num):
            sim = -torch.cdist(z_spec[v], centroids_com)
            p = F.softmax(sim / self.args.para_tau, dim=1)
            p = p * miss_vecs[v].unsqueeze(-1)
            p_views.append(p)

        p_stack = torch.stack(p_views)
        p_sum = torch.sum(p_stack, dim=0)
        valid_views = torch.sum(miss_stack, dim=0)
        valid_views = torch.clamp(valid_views, min=1.0)
        p_avg = p_sum / valid_views

        y_pred = torch.argmax(p_avg, dim=1)
        y_onehot = F.one_hot(y_pred, num_classes=self.args.cluster_num)

        z_recovered = torch.stack([self.Flows.recover_SingleView(z_tilde_com, v) for v in range(self.args.view_num)])

        sim_recovered = -torch.cdist(z_recovered.view(-1, z_recovered.size(-1)), centroids_com)
        p_recovered = F.softmax(sim_recovered / self.args.para_tau, dim=1).view(self.args.view_num, cur_N, -1)

        sim_original = torch.stack(
            [-torch.cdist(z_spec[v], centroids_com) for v in range(self.args.view_num)])
        p_original = F.softmax(sim_original / self.args.para_tau, dim=2)

        miss_stack_expanded = miss_stack.expand(-1, -1, self.args.cluster_num)
        p_final = torch.where(miss_stack_expanded == 1, p_original, p_recovered)

        log_p = torch.log(p_final + 1e-10)
        y_onehot_expanded = y_onehot.unsqueeze(0).expand(self.args.view_num, -1, -1)

        align_loss = -torch.sum(y_onehot_expanded * log_p)
        entropy = -self.args.para_gamma * torch.sum(p_final * log_p)

        loss_pc = (align_loss + entropy) / (cur_N * self.args.view_num)

        return loss_pc




