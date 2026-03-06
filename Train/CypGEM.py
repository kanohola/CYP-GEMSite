# -*- coding: utf-8 -*-
import os, time, json, pickle, warnings, argparse
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch, to_dense_adj, softmax
from torch_geometric.nn import TransformerConv, global_mean_pool, global_max_pool
from torch_geometric.nn.norm import BatchNorm as PGBatchNorm, LayerNorm as PGLayerNorm
from torch_geometric.nn.aggr import SumAggregation
from sklearn.metrics import (
    roc_auc_score, average_precision_score, matthews_corrcoef, 
    jaccard_score, precision_score, recall_score, precision_recall_curve, auc
)
from torch.optim.lr_scheduler import LambdaLR

warnings.filterwarnings("ignore")

# ================= 1. Config =================
NODE_IN_DIM = 47        
EXPECTED_EDGE_DIM = 15  

CONFIG = {
    'data_dir': Path("/data1/zyx/new_test_no_threshold/enzyme_type_with_multi_bone"), 
    'external_test_pkl': "/data1/zyx/new_test_no_threshold/processed/external_test_3d.pkl", 
    
    'epochs': 300,          
    'warmup_epochs': 20,
    'patience': 10,
    'batch_size': 96,      
    
    'lr': 7e-5,             
    'min_lr': 1e-6,
    
    'weight_decay': 0.001,   
    'grad_clip': 2.0,
    
    'alpha': 0.75,           
    
    'pos_weight': 1.0,      
    
    'hidden': 768,         
    'heads': 8,
    'n_layers': 6,          
    
    'd_e': 128,              
    'attn_heads': 4,
    'cls_hidden': 128,
    
    'dropout': 0.35,        
    'feat_drop': 0.2,       
    'droppath': 0.1,
    'norm_type': 'batch',
    'max_degree': 64,
    'max_spd': 32,
    'num_tasks': 9,
    
    'use_ema': True,
    'ema_decay': 0.995
}

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience, self.verbose, self.delta = patience, verbose, delta
        self.path, self.trace_func = path, trace_func
        self.counter, self.best_score, self.early_stop = 0, None, False
        self.val_score_min = -np.Inf

    def __call__(self, score, model, ema=None):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model, ema) 
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience: self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model, ema) 
            self.counter = 0

    def save_checkpoint(self, score, model, ema=None):
        if self.verbose:
            self.trace_func(f'Validation metric improved ({self.val_score_min:.4f} --> {score:.4f}).  Saving model ...')
        
        if ema:
            ema.apply_shadow(model)
            torch.save(model.state_dict(), self.path)
            ema.restore(model)
        else:
            torch.save(model.state_dict(), self.path)
            
        self.val_score_min = score

# ================= 2. EMA & Scheduler =================

class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {}
        self.original = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.original[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.original[name]

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    return LambdaLR(optimizer, lr_lambda, last_epoch)

# ================= 3. Loss Function & Evaluation =================

def mixed_loss_function(logits, targets, alpha_task=0.8, pos_weight=1.0, rank_weight=0.5):
    
    target_som = (targets.sum(dim=1) > 0).float()
    
    
    log_prob_none = -F.softplus(logits).sum(dim=1)
    pred_som = 1.0 - torch.exp(log_prob_none)
    
    
    loss_main_raw = F.binary_cross_entropy(pred_som, target_som, reduction='none')
    weights = torch.ones_like(target_som)
    weights[target_som > 0.5] = pos_weight
    loss_main = (loss_main_raw * weights).mean()
    
    
    mask_pos = target_som > 0.5
    if mask_pos.sum() > 0:
        loss_aux = F.binary_cross_entropy_with_logits(logits[mask_pos], targets[mask_pos])
    else:
        loss_aux = torch.tensor(0.0, device=logits.device)
        
    if mask_pos.sum() > 0 and (~mask_pos).sum() > 0:
        pos_scores = pred_som[mask_pos]
        neg_scores = pred_som[~mask_pos]
        
        n_pairs = min(len(pos_scores), len(neg_scores))

        p_idx = torch.randperm(len(pos_scores))[:n_pairs]
        n_idx = torch.randperm(len(neg_scores))[:n_pairs]
        
        loss_rank = F.margin_ranking_loss(
            pos_scores[p_idx], 
            neg_scores[n_idx], 
            target=torch.ones(n_pairs, device=logits.device), 
            margin=0.1
        )
    else:
        loss_rank = torch.tensor(0.0, device=logits.device)

   
    total_loss = alpha_task * loss_main + (1 - alpha_task) * loss_aux + rank_weight * loss_rank
    
    return total_loss, loss_main, loss_aux

def calculate_strict_metrics(pred_som, true_som, batch_index, threshold=0.5):
    y_true = true_som.cpu().numpy()
    y_scores = pred_som.cpu().numpy()
    if len(np.unique(y_true)) < 2: return {}

    auc_score = roc_auc_score(y_true, y_scores)
    auprc_score = average_precision_score(y_true, y_scores)
    y_pred = (y_scores >= threshold).astype(int)
    mcc = matthews_corrcoef(y_true, y_pred)
    recall = recall_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    jaccard = jaccard_score(y_true, y_pred, zero_division=0)

    batch_np = batch_index.cpu().numpy()
    num_mols = batch_np.max() + 1
    hits_k = {1: 0, 2: 0, 3: 0}
    total_valid_mols = 0
    
    for i in range(num_mols):
        mol_mask = (batch_np == i)
        if not np.any(mol_mask): continue
        mol_scores = y_scores[mol_mask]
        mol_labels = y_true[mol_mask]
        # if np.sum(mol_labels) == 0: continue
        
        total_valid_mols += 1

        sorted_indices = np.argsort(mol_scores)[::-1]
        

        for k in [1, 2, 3]:
            top_k_indices = sorted_indices[:min(k, len(sorted_indices))]
            is_hit = False
            for idx in top_k_indices:
                if mol_labels[idx] == 1 and mol_scores[idx] >= threshold: 
                    is_hit = True
                    break
            if is_hit: hits_k[k] += 1
                
    metrics = {
        'AUC': auc_score, 
        'AUPRC': auprc_score,  
        'MCC': mcc, 
        'Jaccard': jaccard,    
        'Recall': recall, 
        'Precision': precision,
        'Top1': hits_k[1]/total_valid_mols if total_valid_mols>0 else 0,
        'Top2': hits_k[2]/total_valid_mols if total_valid_mols>0 else 0,
        'Top3': hits_k[3]/total_valid_mols if total_valid_mols>0 else 0
    }
    return metrics

def print_metrics_report(metrics, name="Eval"):
    if not metrics: print(f"[{name}] No metrics."); return
    print(f"\n{'='*20} {name} Report {'='*20}")
    print(f"{'Metric':<15} | {'Value':<10}")
    print("-" * 35)
    for k in ['AUC', 'AUPRC', 'MCC', 'Jaccard', 'Precision', 'Recall', 'Top1', 'Top2', 'Top3']:
        if k in metrics: print(f"{k:<15} | {metrics[k]:.4f}")
    print("-" * 35)
# ================= 4. Model definition  =================

class DropPath(nn.Module):
    def __init__(self, p: float = 0.0): super().__init__(); self.p = p
    def forward(self, x):
        if self.p==0. or not self.training: return x
        keep=1-self.p; shape=(x.shape[0],)+(1,)*(x.ndim-1); rand=keep+torch.rand(shape,dtype=x.dtype,device=x.device)
        return x.div(keep)*torch.floor(rand)

def make_norm(norm_type: str, dim: int):
    t=(norm_type or "none").lower()
    if t=="batch": return PGBatchNorm(dim)
    if t=="layer": return PGLayerNorm(dim, mode="node")
    return nn.Identity()

class NodeProjector(nn.Module):
    def __init__(self, in_dim, hidden, feat_drop): super().__init__(); self.proj=nn.Sequential(nn.Linear(in_dim,hidden),nn.ReLU(),nn.BatchNorm1d(hidden),nn.Dropout(feat_drop))
    def forward(self, x): return self.proj(x)

class EdgeProjector(nn.Module):
    def __init__(self, in_dim, d_e, feat_drop): super().__init__(); self.proj=nn.Sequential(nn.Linear(in_dim,d_e),nn.ReLU(),nn.BatchNorm1d(d_e),nn.Dropout(feat_drop))
    def forward(self, e): return self.proj(e)

class EdgeFusion(nn.Module):
    def __init__(self, d_e):
        super().__init__(); self.gate_mlp=nn.Sequential(nn.Linear(d_e*2,d_e),nn.Sigmoid()); self.norm=nn.LayerNorm(d_e)
    def forward(self, handcraft_feat, spd_feat):
        gate=self.gate_mlp(torch.cat([handcraft_feat,spd_feat],dim=-1)); fused=gate*handcraft_feat+(1.0-gate)*spd_feat; return self.norm(fused)

class EdgeUpdate(nn.Module):
    def __init__(self, node_dim, edge_dim): 
        super().__init__()
        self.node_norm = PGLayerNorm(node_dim, mode="node")
        self.mlp = nn.Sequential(nn.Linear(node_dim * 2 + edge_dim, edge_dim * 2), nn.ReLU())
        self.edge_norm = nn.LayerNorm(edge_dim)
    def forward(self, x, edge_index, edge_attr): 
        src, dst = edge_index
        xn = self.node_norm(x)
        m = torch.cat([xn[src], xn[dst], edge_attr], dim=-1)
        raw_output = self.mlp(m)
        delta, gate_logit = torch.chunk(raw_output, 2, dim=-1)
        gate = torch.sigmoid(gate_logit)
        return self.edge_norm(edge_attr + gate * delta)

class GaussianSmearing(nn.Module):
    def __init__(self, start=0.0, stop=20.0, num_gaussians=32):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / ((stop - start) / (num_gaussians - 1))**2
        self.register_buffer('offset', offset)
    def forward(self, dist):
        dist = dist.unsqueeze(-1) - self.offset
        return torch.exp(self.coeff * torch.pow(dist, 2))

class GraphormerBlock(nn.Module):
    def __init__(self, dim, edge_dim, heads, dropout, norm_type, droppath, max_degree):
        super().__init__(); assert dim%heads==0; self.norm=make_norm(norm_type,dim)
        self.attn=TransformerConv(dim,dim//heads,heads,edge_dim=edge_dim,dropout=dropout)
        self.ffn=nn.Sequential(nn.Linear(dim,dim*4),nn.GELU(),nn.Dropout(dropout),nn.Linear(dim*4,dim),nn.Dropout(dropout))
        self.stoch=DropPath(droppath); self.max_deg=int(max_degree); self.deg_emb=nn.Embedding(self.max_deg,dim)
        self.edge_upd = EdgeUpdate(node_dim=dim, edge_dim=edge_dim)
    def _degree_embed(self, num_nodes, edge_index, device):
        deg=torch.zeros(num_nodes,dtype=torch.long,device=device).scatter_add_(0,edge_index[0],torch.ones_like(edge_index[0]))
        return self.deg_emb(deg.clamp_max(self.max_deg-1))
    def forward(self, x, edge_index, edge_attr):
        x_in = x + self._degree_embed(x.size(0), edge_index, x.device)
        h = self.norm(x_in)
        h_attn = self.attn(h, edge_index, edge_attr) 
        x_attn_res = x_in + self.stoch(h_attn)
        h_ffn = self.norm(x_attn_res)
        h_ffn_out = self.ffn(h_ffn)
        x_new = x_attn_res + self.stoch(h_ffn_out)
        ea_new = self.edge_upd(x_new, edge_index, edge_attr)
        return x_new, ea_new

class GlobalTransformerLayer(nn.Module):
    def __init__(self, dim, heads, max_dist=128, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True, dropout=dropout)
        self.dist_bias = nn.Embedding(max_dist, heads)
        self.gaussian = GaussianSmearing(start=0.0, stop=20.0, num_gaussians=32)
        self.rbf_proj = nn.Linear(32, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim), nn.Dropout(dropout))
        self.num_heads = heads

    def forward(self, x, batch, dist_matrix, dist_3d):
        x_dense, mask = to_dense_batch(x, batch)
        batch_size, max_n = x_dense.size(0), x_dense.size(1)
        dist_clamped = dist_matrix.clamp(0, self.dist_bias.num_embeddings - 1).long()
        bias_spd = self.dist_bias(dist_clamped).permute(0, 3, 1, 2)
        rbf_feat = self.gaussian(dist_3d)
        bias_3d = self.rbf_proj(rbf_feat).permute(0, 3, 1, 2)
        attn_bias = (bias_spd + bias_3d).reshape(batch_size * self.num_heads, max_n, max_n)
        padding_mask = ~mask
        padding_mask_expanded = padding_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.num_heads, max_n, max_n)
        attn_bias = attn_bias.masked_fill(padding_mask_expanded.reshape(batch_size * self.num_heads, max_n, max_n), float("-inf"))
        h = self.norm1(x_dense)
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_bias)
        h = h + attn_out
        h = h + self.ffn(self.norm2(h))
        return h[mask]

class NodeEdgeCrossAttention(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        self.dim, self.heads, self.d_head = dim, heads, dim // heads
        self.scale = self.d_head ** -0.5
        self.q_proj, self.k_proj, self.v_proj, self.out_proj = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.aggr = SumAggregation()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q_nodes, k_edges, v_edges, edge_index):
        N, E = q_nodes.size(0), k_edges.size(0)
        dst, src = edge_index
        if E == 0: return torch.zeros((N, self.dim), device=q_nodes.device)
        q = self.q_proj(q_nodes).view(N, self.heads, self.d_head)
        k = self.k_proj(k_edges).view(E, self.heads, self.d_head)
        v = self.v_proj(v_edges).view(E, self.heads, self.d_head)
        attn_score = (q[dst] * k).sum(dim=-1) * self.scale
        attn_weights = self.dropout(softmax(attn_score, dst, num_nodes=N))
        weighted_v = (v * attn_weights.unsqueeze(-1)).view(E, self.dim)
        return self.out_proj(self.aggr(weighted_v, dst, dim_size=N))

class AdvancedGraphormer(nn.Module):
    def __init__(self, node_in, edge_in, hidden, heads, n_layers, d_e, num_tasks, 
                 max_spd=32, max_degree=64, dropout=0.1, **kwargs):
        super().__init__()
        self.max_spd = max_spd
        self.node_proj = NodeProjector(node_in, hidden, kwargs.get('feat_drop', 0.1))
        self.edge_proj = EdgeProjector(edge_in, d_e, kwargs.get('feat_drop', 0.1))
        self.spatial_emb = nn.Embedding(max_spd, d_e)
        self.edge_fusion = EdgeFusion(d_e)
        self.layers = nn.ModuleList([
            GraphormerBlock(hidden, d_e, heads, dropout, kwargs.get('norm_type', 'batch'), 
                            kwargs.get('droppath', 0.1), max_degree) 
            for _ in range(n_layers)
        ])
        self.global_layer = GlobalTransformerLayer(hidden, heads, max_spd, dropout)
        self.edge_aggr = NodeEdgeCrossAttention(d_e, heads=kwargs.get('attn_heads', 4), dropout=dropout)
        self.node_readout_proj = nn.Sequential(nn.Linear(hidden * 2, d_e), nn.ReLU(), nn.Dropout(dropout))
        self.global_pool_proj = nn.Sequential(nn.Linear(hidden * 2, d_e), nn.ReLU(), nn.Dropout(dropout))
        self.global_edge_proj = nn.Sequential(nn.Linear(d_e, d_e), nn.ReLU(), nn.Dropout(dropout))
        cls_hidden = kwargs.get('cls_hidden', 128)
        self.head = nn.Sequential(
            nn.Linear(4 * d_e, cls_hidden * 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(cls_hidden * 2, cls_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(cls_hidden, num_tasks)
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, data):
        x, ei, ea, spd = data.x, data.edge_index, data.edge_attr, data.spd
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x = self.node_proj(x)
        ea_handcraft = self.edge_proj(ea)
        spd_embedding = self.spatial_emb(spd.clamp_max(self.max_spd-1))
        ea = self.edge_fusion(ea_handcraft, spd_embedding)
        x0 = x.clone()
        x_hist, ea_hist = [], []
        for blk in self.layers:
            x, ea = blk(x, ei, ea)
            x_hist.append(x)
            ea_hist.append(ea)
        x_combined = sum(x_hist[-3:]) if len(x_hist) >= 3 else x_hist[-1]
        ea_combined = sum(ea_hist[-3:]) if len(ea_hist) >= 3 else ea_hist[-1]
        pos_dense, mask = to_dense_batch(data.pos, batch) 
        dist_3d = torch.cdist(pos_dense, pos_dense, p=2.0)
        dist_spd_dense = to_dense_adj(ei, batch, edge_attr=spd, max_num_nodes=mask.size(1))
        x_global = self.global_layer(x_combined, batch, dist_spd_dense, dist_3d)
        x_combined = x_combined + x_global
        h_node = torch.cat([x_combined, x0], dim=-1)
        h_node_proj = self.node_readout_proj(h_node)
        ea_smart = self.edge_aggr(h_node_proj, ea_combined, ea_combined, ei)
        g_feat = torch.cat([global_mean_pool(x_combined, batch), global_max_pool(x_combined, batch)], dim=-1)
        g_feat_lifted = self.global_pool_proj(g_feat)[batch]
        edge_batch = batch[ei[0]]
        g_edge_mean = global_mean_pool(ea_combined, edge_batch)
        g_edge_lifted = self.global_edge_proj(g_edge_mean)[batch]
        x_final = torch.cat([h_node_proj, ea_smart, g_feat_lifted, g_edge_lifted], dim=-1)
        logits = self.head(self.drop(x_final))
        return logits

# ================= 5. Dataset Class =================

class PickleDataset(torch.utils.data.Dataset):
    def __init__(self, pkl_path):
        self.pkl_path = Path(pkl_path)
        if not self.pkl_path.exists(): 
            print(f"[Warning] File not found: {self.pkl_path}. Returning empty.")
            self.data = []
        else:
            with open(self.pkl_path, 'rb') as f: self.data = pickle.load(f)
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

# ================= 6. Training process =================

def train_one_epoch(model, loader, optimizer, device, alpha, pos_weight, scheduler, ema):
    model.train()
    meters = {'total':0, 'som':0, 'aux':0}
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        logits = model(batch)
        loss, l_som, l_aux = mixed_loss_function(logits, batch.y, alpha, pos_weight)
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
        optimizer.step()
        
        # [新增] EMA 更新
        if ema: ema.update(model)
        if scheduler: scheduler.step()
            
        meters['total'] += loss.item()
        meters['som'] += l_som.item()
        meters['aux'] += l_aux.item()
        
    return {k: v/len(loader) for k,v in meters.items()}

@torch.no_grad()
def evaluate_wrapper(model, loader, device, name="Eval", ema=None):
    if ema: ema.apply_shadow(model)
    model.eval()
    
    if len(loader) == 0: 
        if ema: ema.restore(model)
        return {}
        
    all_pred_som, all_true_som, all_batch = [], [], []
    mol_offset = 0
    
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        probs = torch.sigmoid(logits)

        log_prob_none = -F.softplus(logits).sum(dim=1)
        pred_som = 1.0 - torch.exp(log_prob_none)
        
        true_som = (batch.y.sum(dim=1) > 0).float()
        
        global_batch = batch.batch + mol_offset
        mol_offset += (batch.batch.max().item() + 1)
        
        all_pred_som.append(pred_som)
        all_true_som.append(true_som)
        all_batch.append(global_batch)
        
    full_pred = torch.cat(all_pred_som)
    full_true = torch.cat(all_true_som)
    full_batch = torch.cat(all_batch)
    
    metrics = calculate_strict_metrics(full_pred, full_true, full_batch)
    
    if metrics:
        print(f"[{name:<8}] MCC:{metrics['MCC']:.4f} | AUC:{metrics['AUC']:.4f} | Rec:{metrics['Recall']:.4f} | Pre:{metrics['Precision']:.4f} | Top2:{metrics['Top2']:.4f}")
    
    if ema: ema.restore(model)
    
    return metrics

# ================= 7. main function ================= 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=3)
    args = parser.parse_args()
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader = DataLoader(PickleDataset(CONFIG['data_dir']/"train.pkl"), batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4)
    valid_loader = DataLoader(PickleDataset(CONFIG['data_dir']/"valid.pkl"), batch_size=CONFIG['batch_size'], num_workers=2)
    test_loader  = DataLoader(PickleDataset(CONFIG['data_dir']/"test.pkl"),  batch_size=CONFIG['batch_size'], num_workers=2)
    ext_loader = DataLoader(PickleDataset(CONFIG['external_test_pkl']), batch_size=CONFIG['batch_size'], num_workers=2)

    model = AdvancedGraphormer(
        node_in=NODE_IN_DIM, edge_in=EXPECTED_EDGE_DIM, 
        hidden=CONFIG['hidden'], heads=CONFIG['heads'], n_layers=CONFIG['n_layers'],
        d_e=CONFIG['d_e'], num_tasks=CONFIG['num_tasks'], max_spd=CONFIG['max_spd'],
        max_degree=CONFIG['max_degree'], dropout=CONFIG['dropout'], 
        attn_heads=CONFIG['attn_heads'], cls_hidden=CONFIG['cls_hidden'],
        feat_drop=CONFIG['feat_drop'], droppath=CONFIG['droppath'], 
        norm_type=CONFIG['norm_type']
    ).to(device)
    
    print(f"Model Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    
    total_steps = len(train_loader) * CONFIG['epochs']
    warmup_steps = len(train_loader) * CONFIG['warmup_epochs']
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    ema = EMA(model, CONFIG['ema_decay']) if CONFIG['use_ema'] else None
    
    early_stopping = EarlyStopping(patience=CONFIG['patience'], verbose=True, path="./best_model_ema.pt")

    print("Starting Training...")
    for epoch in range(1, CONFIG['epochs'] + 1):
        start = time.time()
        losses = train_one_epoch(model, train_loader, optimizer, device, CONFIG['alpha'], CONFIG['pos_weight'], scheduler, ema)
        
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:03d} | LR: {lr:.2e} | Loss: {losses['total']:.4f} (Main: {losses['som']:.4f}) | Time: {time.time()-start:.1f}s")
        
        val_metrics = evaluate_wrapper(model, valid_loader, device, "VAL", ema)
        
        evaluate_wrapper(model, ext_loader, device, "EXT", ema)
        
        curr_mcc = val_metrics.get('MCC', 0)
        early_stopping(curr_mcc, model, ema)
        
        if early_stopping.early_stop:
            print("Early stopping.")
            break
            
    print("\nLoading Best Model...")
    
    model.load_state_dict(torch.load("./best_model_ema.pt"))
    test_metrics = evaluate_wrapper(model, test_loader, device, "FINAL_TEST", ema=None)
    print_metrics_report(test_metrics, "Internal Test Set")

    ext_metrics = evaluate_wrapper(model, ext_loader, device, "FINAL_EXT", ema=None)
    print_metrics_report(ext_metrics, "External Test Set")