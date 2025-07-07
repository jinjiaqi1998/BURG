import torch
import argparse
import os
from tqdm import tqdm
import pandas as pd
from torch_kmeans import TorchKMeans
from dataloader import *
from Nmetrics import evaluate
from BURGNets import *
import warnings
warnings.filterwarnings("ignore")


def seed_everything(SEED=42):
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def pd_toExcel(my_dic, fileName):
    Mrs, P_nacs, P_pcs, ACCs, NMIs, ARIs = [], [], [], [], [], []
    for i in range(len(my_dic)):
        Mrs.append(my_dic[i]["Missrate"])
        ACCs.append(my_dic[i]["ACC"])
        NMIs.append(my_dic[i]["NMI"])
        ARIs.append(my_dic[i]["ARI"])

    dfData = {
        'Missrate': Mrs,
        'ACC': ACCs,
        'NMI': NMIs,
        'ARI': ARIs,
    }
    df = pd.DataFrame(dfData)
    df.to_excel(fileName, index=False)


def get_SingleZs():
    singleZs = [[] for _ in range(args.view_num)]
    with torch.no_grad():
        for batch_idx, (xs, _, _, _) in enumerate(Cluster_loader):
            xs = [item.to(device) for item in xs]
            zs = model.get_SingleZs(xs)
            for v in range(args.view_num):
                singleZs[v] += zs[v].cpu().tolist()
    return singleZs


def get_CommonZ(cur_loader):
    commonZ = []
    with torch.no_grad():
        for xs, _, miss_vecs, cur_idx in cur_loader:
            for v in range(args.view_num):
                xs[v] = xs[v].to(device)
                miss_vecs[v] = miss_vecs[v].to(device)
            z = model.get_commonZ(xs, miss_vecs)
            commonZ.append(z)

    Z_mat = torch.cat(commonZ, dim=0)
    return Z_mat


def get_TotalResult(fea_tensor, y_truth):
    estimator = TorchKMeans(n_clusters=args.cluster_num)
    estimator.fit(fea_tensor)
    y_pred = estimator.labels_
    acc, nmi, purity, fscore, precision, recall, ari = evaluate(y_truth, y_pred)
    result_dic = dict({'ACC': acc, 'NMI': nmi, 'PUR': purity, 'Fscore': fscore,
                       'Prec': precision, 'Recall': recall, 'ARI': ari})
    return result_dic


def pretrain(Epoch_AE, Epoch_Flow):
    optimizer_AE = torch.optim.Adam(model.AEs.parameters(), lr=args.lr)
    optimizer_Flow = torch.optim.Adam(model.Flows.parameters(), lr=args.lr)

    t_progress1 = tqdm(range(Epoch_AE), desc='Pretraining AE')
    for epoch in t_progress1:
        tot_l_recon = 0.0
        for batch_idx, (xs, _, miss_vecs, _) in enumerate(AE_loader):
            for v in range(args.view_num):
                xs[v] = xs[v].to(device)
                miss_vecs[v] = miss_vecs[v].to(device)

            loss_Recon = model.train_AE(xs, miss_vecs)

            optimizer_AE.zero_grad()
            loss_Recon.backward()
            optimizer_AE.step()

            tot_l_recon += loss_Recon.item()
        # print('Epoch {}'.format(epoch + 1), 'loss_recon:{:.6f}'.format(tot_l_recon / len(AE_loader)))

    t_progress2 = tqdm(range(Epoch_Flow), desc='Pretraining Flow')
    for epoch in t_progress2:
        tot_l_flow = 0.0
        for batch_idx, (xs, _, miss_vecs, _) in enumerate(AE_loader):
            for v in range(args.view_num):
                xs[v] = xs[v].to(device)
                miss_vecs[v] = miss_vecs[v].to(device)

            with torch.no_grad():
                z_spec = model.AEs.forward_singleZ(xs)
            loss_Flow = model.Flows.flow_SingleViews(z_spec, miss_vecs)

            optimizer_Flow.zero_grad()
            loss_Flow.backward()
            optimizer_Flow.step()

            tot_l_flow += loss_Flow.item()
        # print('Epoch {}'.format(epoch + 1), 'loss_flow:{:.6f}'.format(tot_l_flow / len(AE_loader)))


def train_recover(Epochs):
    optimizer_AE = torch.optim.Adam(model.AEs.parameters(), lr=args.lr)
    optimizer_Flow = torch.optim.Adam(model.Flows.parameters(), lr=args.lr)

    t_progress = tqdm(range(Epochs), desc='Training Recovering')
    for epoch in t_progress:
        tot_l_recon, tot_l_flow, tot_l_recover = 0.0, 0.0, 0.0
        for batch_idx, (xs, _, miss_vecs, _) in enumerate(AE_loader):
            for v in range(args.view_num):
                xs[v] = xs[v].to(device)
                miss_vecs[v] = miss_vecs[v].to(device)

            loss_Recon = model.train_AE(xs, miss_vecs)
            optimizer_AE.zero_grad()
            loss_Recon.backward()
            optimizer_AE.step()

            loss_Flow, loss_Recover = model.train_Recover_observe(xs, miss_vecs)  # 全完整
            loss_Flow_total = args.para_flow * loss_Flow + args.para_recover * loss_Recover

            optimizer_Flow.zero_grad()
            loss_Flow_total.backward()
            optimizer_Flow.step()

            # tot_l_recon += loss_Recon.item()
            # tot_l_flow += args.para_flow * loss_Flow.item()
            # tot_l_recover += args.para_recover * loss_Recover.item()

        # print('Epoch {}'.format(epoch + 1),
        #       'loss_recon:{:.6f}'.format(tot_l_recon / len(AE_loader)),
        #       'loss_flow:{:.6f}'.format(tot_l_flow / len(AE_loader)),
        #       'loss_recover:{:.6f}'.format(tot_l_recover / len(AE_loader)))


def train_DualCon(Epochs):
    optimizer_AE = torch.optim.Adam(model.AEs.parameters(), lr=args.lr)
    optimizer_Flow = torch.optim.Adam(model.Flows.parameters(), lr=args.lr)
    optimizer_All = torch.optim.Adam(model.parameters(), lr=args.lr)

    Dual_loader = torch.utils.data.DataLoader(dataset=AE_dataset, batch_size=args.batch_dual, shuffle=True, drop_last=True)

    base_estimator = TorchKMeans(n_clusters=args.cluster_num)
    t_progress = tqdm(range(Epochs), desc='Training Dual Consistency')
    for epoch in t_progress:
        if epoch % args.update_interval == 0:
            Z_mat = get_CommonZ(Cluster_loader)
            base_estimator.fit(Z_mat)
            centroids_com = base_estimator.cluster_centers_
            centroids_com = centroids_com.to(device)

        tot_l_recon, tot_l_flow, tot_l_recover, tot_l_nac, tot_l_pc = 0.0, 0.0, 0.0, 0.0, 0.0

        for batch_idx, (xs, _, miss_vecs, _) in enumerate(Dual_loader):
            for v in range(args.view_num):
                xs[v] = xs[v].to(device)
                miss_vecs[v] = miss_vecs[v].to(device)

            loss_Recon = model.train_AE(xs, miss_vecs)
            optimizer_AE.zero_grad()
            loss_Recon.backward()
            optimizer_AE.step()

            loss_Flow, loss_Recover = model.train_Recover_observe(xs, miss_vecs)  # 全完整
            loss_Flow_total = args.para_flow * loss_Flow + args.para_recover * loss_Recover

            optimizer_Flow.zero_grad()
            loss_Flow_total.backward()
            optimizer_Flow.step()

            loss_NAC = model.train_nac_vec(xs, miss_vecs)
            loss_PC = model.train_pc_vec(xs, miss_vecs, centroids_com)
            loss_Dual = args.para_nac * loss_NAC + args.para_pc * loss_PC

            optimizer_All.zero_grad()
            loss_Dual.backward()
            optimizer_All.step()

            # tot_l_recon += loss_Recon.item()
            # tot_l_flow += args.para_flow * loss_Flow.item()
            # tot_l_recover += args.para_recover * loss_Recover.item()
            # tot_l_nac += args.para_nac * loss_NAC.item()
            # tot_l_pc += args.para_pc * loss_PC.item()

        # print('Epoch {}'.format(epoch + 1),
        #       'loss_recon:{:.6f}'.format(tot_l_recon / len(Dual_loader)),
        #       'loss_flow:{:.6f}'.format(tot_l_flow / len(Dual_loader)),
        #       'loss_recover:{:.6f}'.format(tot_l_recover / len(Dual_loader)),
        #       'loss_nac:{:.6f}'.format(tot_l_nac / len(Dual_loader)),
        #       'loss_pc:{:.6f}'.format(tot_l_pc / len(Dual_loader)))


def Recovery():
    final_tensor = torch.empty((args.data_num, args.view_num * args.z_dim), device='cpu')
    with torch.no_grad():
        batch_start = 0
        for batch_idx, (xs, _, miss_vecs, _) in enumerate(Cluster_loader):
            xs = [x.to(device) for x in xs]
            miss_vecs = [m.to(device) for m in miss_vecs]
            batch_size = xs[0].shape[0]
            xr_spec, xr_com, z_spec, z_com, xs_com = model.AEs.forward(xs, miss_vecs)
            z_tilde_com, _ = model.Flows.get_ztilde_com(z_spec)

            z_rec = [z.clone() for z in z_spec]
            for v in range(args.view_num):
                missing_samples = (miss_vecs[v] == 0).nonzero().squeeze()
                if len(missing_samples.shape) == 0:
                    missing_samples = missing_samples.unsqueeze(0)
                if len(missing_samples) > 0:
                    z_recovered = model.Flows.recover_SingleView(z_tilde_com, v)
                    z_rec[v][missing_samples] = z_recovered[missing_samples]

            batch_tensor = torch.cat(z_rec, dim=1)

            batch_end = batch_start + batch_size
            final_tensor[batch_start:batch_end] = batch_tensor.cpu()
            batch_start = batch_end

    return final_tensor


if __name__ == '__main__':
    dataset = {
        0: "CUB",
        1: "HW_6Views",
        2: "CiteSeer",
        3: "Animal",
        4: "Reuters_5V_dim10",
        5: "YouTubeFace10_4Views",
    }
    for data_id in dataset:
        All_Metrics = []
        file_name = "./All_Benchmarks/" + dataset[data_id] + ".xlsx"
        for mr in [0.1, 0.3, 0.5, 0.7]:
            print(f'{"=" * 20} {dataset[data_id]}, MR = {mr} {"=" * 20}')

            seed_everything(SEED=42)
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

            X, Y, Miss_vecs, cluster_num, data_num, view_num, view_dims = load_data(dataset[data_id], mr)
            idxs = np.array([i for i in range(data_num)])

            parser = argparse.ArgumentParser(description='train')
            parser.add_argument('--dataset', default=str(data_id), help='name of dataset')
            parser.add_argument('--missrate', default=mr, help='missing rate of multi-view data')
            parser.add_argument('--batch_size', default=128, help='batch size of pretraining stage')
            parser.add_argument('--batch_dual', default=512, help='batch size of dual consistency stage')
            parser.add_argument('--lr', default=0.0003, help='learning rate')
            parser.add_argument('--Epochs_PreAE', default=100, help='epoches of Pretraining AEs')
            parser.add_argument('--Epochs_PreFlow', default=100, help='epoches of Pretraining Flows')
            parser.add_argument('--Epochs_Recover', default=30, help='epoches of Training Recovery')
            parser.add_argument('--Epochs_DualCon', default=20, help='epoches of Training Dual Consistency')
            parser.add_argument('--z_dim', default=128, help='dimension of embedding from encoders')
            parser.add_argument('--hidden_dim', default=200, help='dim of hidden features in normalizing flow')
            parser.add_argument('--coupling_num', default=4, help='number of coupling layers')
            parser.add_argument('--hidden_layers', default=5, help='number of hidden layers in coupling layer')
            parser.add_argument('--update_interval', default=1, type=int, help='number of epochs before updating the centroids')

            parser.add_argument('--para_nac', default=1.0)
            parser.add_argument('--para_pc', default=1.0)

            parser.add_argument('--para_recon', default=1.0)
            parser.add_argument('--para_flow', default=1.0)
            parser.add_argument('--para_recover', default=1.0)
            parser.add_argument('--para_gamma', default=1.0)
            parser.add_argument('--para_tau', default=1.0)
            parser.add_argument('--cluster_num', default=cluster_num, help='number of clusters')
            parser.add_argument('--data_num', default=data_num, help='number of samples')
            parser.add_argument('--view_num', default=view_num, help='number of views')
            parser.add_argument('--view_dims', default=view_dims, help='dimension of views')
            args = parser.parse_args()

            AE_dataset = TrainDataset_All(X, Y, Miss_vecs, idxs)
            AE_loader = torch.utils.data.DataLoader(dataset=AE_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
            Clu_dataset = TrainDataset_All(X, Y, Miss_vecs, idxs)
            Cluster_loader = torch.utils.data.DataLoader(dataset=Clu_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

            model = BURG(args).to(device)

            pretrain(args.Epochs_PreAE, args.Epochs_PreFlow)

            train_recover(args.Epochs_Recover)

            train_DualCon(args.Epochs_DualCon)

            fea_l_afterfill = Recovery()
            res_dic = get_TotalResult(fea_l_afterfill, Y[0])
            print('Final Result: ACC=%.4f, NMI=%.4f, ARI=%.4f' % (res_dic['ACC'], res_dic['NMI'], res_dic['ARI']))

            write_dic = dict({'Missrate': mr, 'ACC': res_dic['ACC'], 'NMI': res_dic['NMI'], 'ARI': res_dic['ARI']})
            All_Metrics.append(write_dic)
            # pd_toExcel(All_Metrics, file_name)

