import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from hashlib import sha256

class BinaryDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, pos_lab, neg_lab):
        self.dataset = dataset
        self.used_ids = []
        self.pos_lab, self.neg_lab = pos_lab, neg_lab
        for i, (X, y) in enumerate(dataset):
            if y == pos_lab or y == neg_lab:
                self.used_ids.append(i)

    def __len__(self,):
        return len(self.used_ids)

    def __getitem__(self, i):
        X, y = self.dataset[self.used_ids[i]]
        if y == self.pos_lab:
            y_new = 1
        else:
            assert y == self.neg_lab
            y_new = 0
        return X, y_new

class SmoothedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, sigma):
        self.dataset = dataset
        self.sigma = sigma

        data_shape = dataset[0][0].shape
        self.perturbs = torch.FloatTensor(len(dataset), *data_shape).normal_(0, sigma)

    def __len__(self,):
        return len(self.dataset)

    def __getitem__(self, i):
        X, y = self.dataset[i]
        X_new = X + self.perturbs[i]
        #pert = torch.FloatTensor(*X.shape).normal_(0, self.sigma)
        #X_new = X + pert
        return X_new, y



def get_sigma(model, batch, lr_sig, sig_0, iters, device="cuda:0", ret_radius=False):
    sig = torch.autograd.Variable(sig_0, requires_grad=True).view(batch.shape[0], 1, 1, 1)
    m = torch.distributions.normal.Normal(torch.zeros(batch.shape[0]).to(device),torch.ones(batch.shape[0]).to(device))

    for _ in range(iters):
        sig = sig.to(device)
        batch = batch.to(device)

        eps = torch.randn_like(batch) * sig
        #out = model(batch + eps)
        eps = eps.to(device)
        model = model.to(device)
        out = torch.sigmoid(model(batch + eps).squeeze(1)).detach().cpu().numpy()
        #print(out)
        #print(out.shape,'out.shape')

        vals = np.stack((1-out, out), axis=1)
        vals = torch.from_numpy(vals)
        #print(vals.shape)

        vals.transpose_(0, 1)
        #print(vals.shape)
        vals = vals.to(device)
        gap = m.icdf(vals[0].clamp_(0.02, 0.98)) - m.icdf(vals[1].clamp_(0.02, 0.98))
        gap= torch.abs(gap)
        sig = sig.to(device)
        gap = gap.to(device)
        radius = sig.reshape(-1) / 2 * gap  # The radius formula
        grad = torch.autograd.grad(torch.relu(radius).sum(), sig)
        sig = sig + lr_sig * grad[0]  # Gradient Ascent step
    batch = batch.to(device)
    sig = sig.to(device)
    eps = torch.randn_like(batch) * sig
    if ret_radius:
        return sig.reshape(-1), batch + eps, radius
    return sig.reshape(-1), batch + eps


def train_model(model, dataloader, lr, sigma_0, epoch, iters_sig, dldp_setting=(0.0,5.0), verbose=True, testloader=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if dldp_setting[0] != 0:
        from torchdp import PrivacyEngine
        privacy_engine = PrivacyEngine(model, dataloader, alphas=0.0, noise_multiplier=dldp_setting[0], max_grad_norm=dldp_setting[1])
        privacy_engine.attach(optimizer)

    cum_loss = 0.0
    cum_acc = 0.0
    cum_pred = []
    cum_lab = []
    tot = 0.0
    for i,(x_in, y_in,idx) in enumerate(dataloader):
        B = x_in.size()[0]
        sigma, batch_cor = get_sigma(model, x_in, 0.0001, sigma_0[idx], iters_sig)
        sigma = sigma.to("cuda:0")
        sigma_0 = sigma_0.to("cuda:0")
        sigma_0[idx] = sigma
        model = model.to("cuda:0")
        batch_cor = batch_cor.to("cuda:0")
        y_in =y_in.to("cuda:0")
        pred = model(batch_cor).squeeze(1)
        loss = model.loss(pred, y_in)
        y_in = y_in.to('cpu')

        model.zero_grad()
        loss.backward()
        optimizer.step()

        cum_loss += loss.item() * B
        cum_acc += ((pred>0).cpu().long().eq(y_in)).sum().item()
        cum_pred = cum_pred + list(pred.detach().cpu().numpy())
        cum_lab = cum_lab + list(y_in.numpy())
        tot = tot + B
    print("Epoch %d, loss = %.4f, acc = %.4f" % (epoch, cum_loss / tot, cum_acc / tot))

    if verbose:
        print ("Epoch %d, loss = %.4f, acc = %.4f"%(epoch, cum_loss/tot, cum_acc/tot))
        if testloader is not None:
            print (eval_binary_model(model, testloader))
            model.train()
    return sigma_0

def eval_model(model, dataloader, sig, iters_sig, ret_auc=False):
    model.eval()
    cum_acc = 0.0
    cum_pred = []
    cum_lab = []
    tot = 0.0
    for i,(x_in, y_in,idx) in enumerate(dataloader):
        B = x_in.size()[0]
        x_in = x_in.to("cuda:0")
        sigma, batch_cor = get_sigma(model, x_in, 0.0001, sig[idx], iters_sig, device='cuda:0')
        sigma = sigma.to("cuda:0")
        sig = sig.to("cuda:0")
        sig[idx] = sigma
        #pred = model(batch_cor).squeeze(1)
        pred = model(x_in).squeeze(1) #对输入数据进行模型预测，并将结果压缩以去掉单维度条目（通常是不必要的批处理维度）

        cum_acc += ((pred>0).cpu().long().eq(y_in)).sum().item()
        cum_pred = cum_pred + list(pred.detach().cpu().numpy())
        cum_lab = cum_lab + list(y_in.numpy())
        tot = tot + B
    if ret_auc:
        return cum_acc / tot, roc_auc_score(cum_lab, cum_pred), sig
    else:
        return cum_acc / tot , sig

def certificate_over_dataset(model, dataloader, PREFIX, N_m, sigma):
    model_preds = []
    labs = []
    print('_______________________',len(dataloader))
    for _ in tqdm(range(N_m)):
        model.load_state_dict(torch.load(PREFIX+'/smoothed_%d.model'%_))
        hashval = int(sha256(open(PREFIX+'/smoothed_%d.model'%_, 'rb').read()).hexdigest(), 16) % (2**32)
        model.fix_pert(sigma=sigma, hash_num=hashval)
        all_pred = np.zeros((0,2))
        for x_in, y_in, idx in dataloader:
            pred = torch.sigmoid(model(x_in).squeeze(1)).detach().cpu().numpy()
            pred = np.stack((1-pred, pred), axis=1)
            if (_ == 0):
                labs = labs + list(y_in.numpy())
            all_pred = np.concatenate([all_pred, pred], axis=0)
        model_preds.append(all_pred)
        model.unfix_pert()

    gx = np.array(model_preds).mean(0)
    print(gx,'____________gx_________________')
    labs = np.array(labs)
    pa = gx.max(1)
    pred_c = gx.argmax(1)
    print(pred_c,'____________pred_c_________________')

    gx[np.arange(len(pred_c)), pred_c] = -1
    pb = gx.max(1)
    is_acc = (pred_c==labs)
    return pa, pb, is_acc,pred_c,labs
