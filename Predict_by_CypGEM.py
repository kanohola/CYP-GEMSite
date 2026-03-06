# -*- coding: utf-8 -*-
import os
import argparse
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.rdchem import HybridizationType, ChiralType

from torch_geometric.data import Data
from torch_geometric.utils import softmax, to_dense_batch, to_dense_adj
from torch_geometric.nn import TransformerConv, global_mean_pool, global_max_pool
from torch_geometric.nn.norm import BatchNorm as PGBatchNorm, LayerNorm as PGLayerNorm
from torch_geometric.nn.aggr import SumAggregation

# ==============================================================================
# ==============================================================================
NODE_IN_DIM = 47        
EXPECTED_EDGE_DIM = 15  
SUBTYPES = ['1A2', '2A6', '2B6', '2C8', '2C9', '2C19', '2D6', '2E1', '3A4']
# ==============================================================================
# ==============================================================================

class DropPath(nn.Module):
    def __init__(self, p: float = 0.0): super().__init__(); self.p = p
    def forward(self, x): return x 

def make_norm(norm_type: str, dim: int):
    t=(norm_type or "none").lower()
    if t=="batch": return PGBatchNorm(dim)
    if t=="layer": return PGLayerNorm(dim, mode="node")
    return nn.Identity()

class NodeProjector(nn.Module):
    def __init__(self, in_dim, hidden, feat_drop): 
        super().__init__()
        self.proj=nn.Sequential(nn.Linear(in_dim,hidden),nn.ReLU(),nn.BatchNorm1d(hidden),nn.Dropout(feat_drop))
    def forward(self, x): return self.proj(x)

class EdgeProjector(nn.Module):
    def __init__(self, in_dim, d_e, feat_drop): 
        super().__init__()
        self.proj=nn.Sequential(nn.Linear(in_dim,d_e),nn.ReLU(),nn.BatchNorm1d(d_e),nn.Dropout(feat_drop))
    def forward(self, e): return self.proj(e)

class EdgeFusion(nn.Module):
    def __init__(self, d_e):
        super().__init__(); self.gate_mlp=nn.Sequential(nn.Linear(d_e*2,d_e),nn.Sigmoid()); self.norm=nn.LayerNorm(d_e)
    def forward(self, handcraft_feat, spd_feat):
        gate = self.gate_mlp(torch.cat([handcraft_feat, spd_feat], dim=-1))
        return self.norm(gate * handcraft_feat + (1.0 - gate) * spd_feat)

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
        return self.edge_norm(edge_attr + torch.sigmoid(gate_logit) * delta)

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
        super().__init__(); self.norm=make_norm(norm_type,dim)
        self.attn=TransformerConv(dim,dim//heads,heads,edge_dim=edge_dim,dropout=dropout)
        self.ffn=nn.Sequential(nn.Linear(dim,dim*4),nn.GELU(),nn.Dropout(dropout),nn.Linear(dim*4,dim),nn.Dropout(dropout))
        self.stoch=DropPath(droppath); self.deg_emb=nn.Embedding(int(max_degree),dim)
        self.edge_upd = EdgeUpdate(node_dim=dim, edge_dim=edge_dim)
    def _degree_embed(self, num_nodes, edge_index, device):
        deg=torch.zeros(num_nodes,dtype=torch.long,device=device).scatter_add_(0,edge_index[0],torch.ones_like(edge_index[0]))
        return self.deg_emb(deg.clamp_max(self.deg_emb.num_embeddings-1))
    def forward(self, x, edge_index, edge_attr):
        x_in = x + self._degree_embed(x.size(0), edge_index, x.device)
        h = self.norm(x_in)
        h_attn = self.attn(h, edge_index, edge_attr) 
        x_attn_res = x_in + self.stoch(h_attn)
        h_ffn = self.norm(x_attn_res)
        x_new = x_attn_res + self.stoch(self.ffn(h_ffn))
        return x_new, self.edge_upd(x_new, edge_index, edge_attr)

class GlobalTransformerLayer(nn.Module):
    def __init__(self, dim, heads, max_dist=128, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim); self.num_heads = heads
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True, dropout=dropout)
        self.dist_bias = nn.Embedding(max_dist, heads)
        self.gaussian = GaussianSmearing(start=0.0, stop=20.0, num_gaussians=32)
        self.rbf_proj = nn.Linear(32, heads); self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim), nn.Dropout(dropout))

    def forward(self, x, batch, dist_matrix, dist_3d):
        x_dense, mask = to_dense_batch(x, batch)
        batch_size, max_n = x_dense.size(0), x_dense.size(1)
        dist_clamped = dist_matrix.clamp(0, self.dist_bias.num_embeddings - 1).long()
        bias_spd = self.dist_bias(dist_clamped).permute(0, 3, 1, 2)
        rbf_feat = self.gaussian(dist_3d)
        bias_3d = self.rbf_proj(rbf_feat).permute(0, 3, 1, 2)
        attn_bias = (bias_spd + bias_3d).reshape(batch_size * self.num_heads, max_n, max_n)
        padding_mask_expanded = (~mask).unsqueeze(1).unsqueeze(1).expand(batch_size, self.num_heads, max_n, max_n)
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
        self.aggr = SumAggregation(); self.dropout = nn.Dropout(dropout)
    def forward(self, q_nodes, k_edges, v_edges, edge_index):
        N, E = q_nodes.size(0), k_edges.size(0)
        dst, src = edge_index
        if E == 0: return torch.zeros((N, self.dim), device=q_nodes.device)
        q = self.q_proj(q_nodes).view(N, self.heads, self.d_head)
        k = self.k_proj(k_edges).view(E, self.heads, self.d_head)
        v = self.v_proj(v_edges).view(E, self.heads, self.d_head)
        attn_score = (q[dst] * k).sum(dim=-1) * self.scale
        attn_weights = self.dropout(softmax(attn_score, dst, num_nodes=N))
        return self.out_proj(self.aggr((v * attn_weights.unsqueeze(-1)).view(E, self.dim), dst, dim_size=N))

class AdvancedGraphormer(nn.Module):
    def __init__(self, node_in, edge_in, hidden, heads, n_layers, d_e, num_tasks, 
                 max_spd=32, max_degree=64, dropout=0.1, **kwargs):
        super().__init__()
        self.max_spd = max_spd
        self.node_proj = NodeProjector(node_in, hidden, kwargs.get('feat_drop', 0.1))
        self.edge_proj = EdgeProjector(edge_in, d_e, kwargs.get('feat_drop', 0.1))
        self.spatial_emb = nn.Embedding(max_spd, d_e)
        self.edge_fusion = EdgeFusion(d_e)
        self.layers = nn.ModuleList([GraphormerBlock(hidden, d_e, heads, dropout, kwargs.get('norm_type', 'batch'), kwargs.get('droppath', 0.1), max_degree) for _ in range(n_layers)])
        self.global_layer = GlobalTransformerLayer(hidden, heads, max_spd, dropout)
        self.edge_aggr = NodeEdgeCrossAttention(d_e, heads=kwargs.get('attn_heads', 4), dropout=dropout)
        self.node_readout_proj = nn.Sequential(nn.Linear(hidden * 2, d_e), nn.ReLU(), nn.Dropout(dropout))
        self.global_pool_proj = nn.Sequential(nn.Linear(hidden * 2, d_e), nn.ReLU(), nn.Dropout(dropout))
        self.global_edge_proj = nn.Sequential(nn.Linear(d_e, d_e), nn.ReLU(), nn.Dropout(dropout))
        cls_hidden = kwargs.get('cls_hidden', 128)
        self.head = nn.Sequential(nn.Linear(4 * d_e, cls_hidden * 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(cls_hidden * 2, cls_hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(cls_hidden, num_tasks))
        self.drop = nn.Dropout(dropout)

    def forward(self, data):
        x, ei, ea, spd = data.x.float(), data.edge_index, data.edge_attr.float(), data.spd
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        x = self.node_proj(x); ea_handcraft = self.edge_proj(ea)
        ea = self.edge_fusion(ea_handcraft, self.spatial_emb(spd.clamp_max(self.max_spd-1)))
        
        x0 = x.clone(); x_hist, ea_hist = [], []
        for blk in self.layers:
            x, ea = blk(x, ei, ea)
            x_hist.append(x); ea_hist.append(ea)
            
        x_combined = sum(x_hist[-3:]) if len(x_hist) >= 3 else x_hist[-1]
        ea_combined = sum(ea_hist[-3:]) if len(ea_hist) >= 3 else ea_hist[-1]
        
        pos_dense, mask = to_dense_batch(data.pos, batch) 
        x_global = self.global_layer(x_combined, batch, to_dense_adj(ei, batch, edge_attr=spd, max_num_nodes=mask.size(1)), torch.cdist(pos_dense, pos_dense, p=2.0))
        x_combined = x_combined + x_global
        
        h_node_proj = self.node_readout_proj(torch.cat([x_combined, x0], dim=-1))
        x_final = torch.cat([
            h_node_proj, 
            self.edge_aggr(h_node_proj, ea_combined, ea_combined, ei), 
            self.global_pool_proj(torch.cat([global_mean_pool(x_combined, batch), global_max_pool(x_combined, batch)], dim=-1))[batch], 
            self.global_edge_proj(global_mean_pool(ea_combined, batch[ei[0]]))[batch]
        ], dim=-1)
        return self.head(self.drop(x_final))

# ==============================================================================
# ==============================================================================

def _one_hot_index(idx: int, num_classes: int) -> torch.Tensor:
    v = torch.zeros(num_classes, dtype=torch.float32)
    if 0 <= idx < num_classes: v[idx] = 1.0
    return v

class FeatureGenerator:
    def __init__(self, max_spd=64):
        self.max_spd = max_spd

    def process_molecule(self, smiles_or_mol):
        mol = Chem.MolFromSmiles(smiles_or_mol) if isinstance(smiles_or_mol, str) else smiles_or_mol
        if mol is None: return None, None  
        AllChem.ComputeGasteigerCharges(mol)
        edge_index = self._get_adjacency_info(mol)
        data = Data(
            x=self._get_node_features(mol), 
            edge_index=edge_index, 
            edge_attr=self._get_edge_features(mol), 
            spd=self._get_spd_matrix(mol, edge_index), 
            pos=self._get_optimized_3d_coords(mol), 
            smiles=Chem.MolToSmiles(mol)
        )
        return data, mol  

    def _get_optimized_3d_coords(self, mol):
        try:
            mol_h = Chem.AddHs(mol)
            params = AllChem.ETKDGv3(); params.randomSeed = 42
            if AllChem.EmbedMolecule(mol_h, params) == -1: return torch.zeros((mol.GetNumAtoms(), 3))
            AllChem.MMFFOptimizeMolecule(mol_h)
            return torch.tensor(Chem.RemoveHs(mol_h).GetConformer().GetPositions(), dtype=torch.float32)
        except: return torch.zeros((mol.GetNumAtoms(), 3))

    def _get_spd_matrix(self, mol, edge_index):
        num_atoms = mol.GetNumAtoms()
        if num_atoms == 0 or edge_index.size(1) == 0: return torch.zeros((0,), dtype=torch.long)
        G = nx.Graph(); G.add_nodes_from(range(num_atoms))
        for i, j in edge_index.t().tolist(): G.add_edge(i, j)
        spd = nx.floyd_warshall_numpy(G, nodelist=range(num_atoms))
        spd[np.isinf(spd)] = self.max_spd + 1
        return torch.tensor(np.clip(spd[edge_index[0].numpy(), edge_index[1].numpy()], 0, self.max_spd), dtype=torch.long)

    def _get_node_features(self, mol):
        try: logp, tpsa, mw, rings = Descriptors.MolLogP(mol), Descriptors.TPSA(mol), Descriptors.MolWt(mol), Descriptors.RingCount(mol)
        except: logp, tpsa, mw, rings = 0.0, 0.0, 0.0, 0.0
        g_feats = torch.tensor([np.clip(logp/10, -2, 2), np.clip(tpsa/150, 0, 2), np.clip(mw/500, 0, 2), np.clip(rings/5, 0, 2)])
        
        feats = []
        for atom in mol.GetAtoms():
            sym = _one_hot_index(['C','N','O','S','P','F','Cl','Br','I'].index(atom.GetSymbol()) if atom.GetSymbol() in ['C','N','O','S','P','F','Cl','Br','I'] else 9, 10)
            hyb = _one_hot_index([HybridizationType.SP, HybridizationType.SP2, HybridizationType.SP3].index(atom.GetHybridization()) if atom.GetHybridization() in [HybridizationType.SP, HybridizationType.SP2, HybridizationType.SP3] else 3, 4)
            deg = _one_hot_index(atom.GetDegree() if atom.GetDegree() <= 4 else 5, 6)
            hs = _one_hot_index(atom.GetTotalNumHs() if atom.GetTotalNumHs() <= 3 else 4, 5)
            arom = torch.tensor([1.0 if atom.GetIsAromatic() else 0.0])
            r_idx = mol.GetRingInfo().NumAtomRings(atom.GetIdx()); r_oh = _one_hot_index(r_idx if r_idx <= 2 else 3, 4)
            imp = _one_hot_index(atom.GetImplicitValence() if atom.GetImplicitValence() <= 4 else 5, 6)
            gast = torch.tensor([float(atom.GetProp('_GasteigerCharge')) if atom.HasProp('_GasteigerCharge') and np.isfinite(float(atom.GetProp('_GasteigerCharge'))) else 0.0])
            f_chg = torch.tensor([float(atom.GetFormalCharge())])
            r_elec = torch.tensor([float(atom.GetNumRadicalElectrons())])
            chiral = _one_hot_index({ChiralType.CHI_TETRAHEDRAL_CCW: 0, ChiralType.CHI_TETRAHEDRAL_CW: 1, ChiralType.CHI_OTHER: 2}.get(atom.GetChiralTag(), 3), 4)
            
            feats.append(torch.cat([sym, hyb, deg, hs, arom, r_oh, imp, gast, f_chg, r_elec, chiral, g_feats]))
        return torch.stack(feats)

    def _get_edge_features(self, mol):
        feats = []
        hybrid_map = {HybridizationType.SP: 0, HybridizationType.SP2: 1, HybridizationType.SP3: 2}
        combo_map = {(2, 2): 0, (1, 2): 1, (1, 1): 2, (0, 2): 3, (0, 1): 4}
        for b in mol.GetBonds():
            bt = b.GetBondType(); b_oh = _one_hot_index(0 if bt==Chem.BondType.SINGLE else 1 if bt==Chem.BondType.DOUBLE else 2 if bt==Chem.BondType.TRIPLE else 3 if (bt==Chem.BondType.AROMATIC or b.GetIsAromatic()) else 0, 4)
            ai, aj = mol.GetAtomWithIdx(b.GetBeginAtomIdx()), mol.GetAtomWithIdx(b.GetEndAtomIdx())
            charge_i = float(ai.GetProp('_GasteigerCharge') if ai.HasProp('_GasteigerCharge') else 0)
            charge_j = float(aj.GetProp('_GasteigerCharge') if aj.HasProp('_GasteigerCharge') else 0)
            pol = torch.tensor([abs(charge_i - charge_j)])
            h_idx_i, h_idx_j = hybrid_map.get(ai.GetHybridization(), 99), hybrid_map.get(aj.GetHybridization(), 99)
            h_combo = _one_hot_index(combo_map.get(tuple(sorted((h_idx_i, h_idx_j))), 5), 6)
            r_oh = torch.tensor([1., 0.] if b.IsInRing() else [0., 1.])
            c_oh = torch.tensor([1., 0.] if b.GetIsConjugated() else [0., 1.])
            e = torch.cat([b_oh, pol, h_combo, r_oh, c_oh]); feats.extend([e, e])
        return torch.stack(feats) if feats else torch.zeros((0, 15))

    def _get_adjacency_info(self, mol):
        edges = []
        for b in mol.GetBonds(): i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx(); edges.extend([[i,j],[j,i]])
        return torch.tensor(edges).t().contiguous() if edges else torch.empty((2,0), dtype=torch.long)

# ==============================================================================
# ==============================================================================

class CypGEMPredictor:
    def __init__(self, model_path, threshold=0.5, device='cuda'):
        self.threshold = threshold
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = AdvancedGraphormer(node_in=47, edge_in=15, hidden=768, heads=8, n_layers=6, d_e=128, num_tasks=9).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=False))
        self.model.eval()
        self.feat_gen = FeatureGenerator()

    def predict(self, smiles):
        data, mol = self.feat_gen.process_molecule(smiles)
        if data is None: return None, None
        data = data.to(self.device)
        with torch.no_grad():
            logits = self.model(data)
            probs = (1.0 - torch.exp(-F.softplus(logits).sum(dim=1))).cpu().numpy()
        
        atom_probs = []
        for i in range(len(probs)):
            atom_probs.append({
                "idx": i, 
                "sym": mol.GetAtomWithIdx(i).GetSymbol(), 
                "prob": float(probs[i]),
                "is_som": bool(probs[i] > self.threshold)
            })
        return mol, atom_probs

    def visualize(self, mol, atom_probs, path):
        from rdkit.Chem import Draw
        soms = [p for p in atom_probs if p['is_som']]
        soms.sort(key=lambda x: x['prob'], reverse=True)
        
        highlights = [p['idx'] for p in soms]
        max_p = max([p['prob'] for p in soms]) if soms else 1.0
        
        colors = {p['idx']: plt.get_cmap('YlOrRd')(0.2 + 0.8 * (p['prob'] / max_p))[:3] for p in soms}
        
        d = Draw.rdMolDraw2D.MolDraw2DCairo(800, 800)
        d.drawOptions().addAtomIndices = True
        Draw.rdMolDraw2D.PrepareAndDrawMolecule(d, mol, highlightAtoms=highlights, highlightAtomColors=colors)
        d.FinishDrawing()
        d.WriteDrawingText(path + ".png")
        
        with open(path + ".csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Index", "Symbol", "Prob", "Is_SoM"])
            for p in atom_probs:
                writer.writerow([p['idx'], p['sym'], f"{p['prob']:.4f}", p['is_som']])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to CypGEM.pt")
    parser.add_argument("--input", type=str, required=True, help="SMILES string or file path")
    parser.add_argument("--out", type=str, default="./results", help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for SoM (default: 0.5)")
    args = parser.parse_args()
    
    os.makedirs(args.out, exist_ok=True)
    predictor = CypGEMPredictor(args.model, threshold=args.threshold)
    
    inputs = [args.input] if not os.path.isfile(args.input) else [line.strip() for line in open(args.input)]
    
    for i, s in enumerate(inputs):
        mol, res = predictor.predict(s)
        if mol: 
            predictor.visualize(mol, res, os.path.join(args.out, f"mol_{i}"))
            
            soms = [p for p in res if p['is_som']]
            print(f"Predicted mol_{i} successfully. Found {len(soms)} SoM(s) > {args.threshold}")
