import os
import argparse
import csv
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import networkx as nx 

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.rdchem import HybridizationType, ChiralType

from torch_geometric.data import Data
from torch_geometric.utils import dropout_edge, softmax, to_dense_batch, to_dense_adj
from torch_geometric.nn import TransformerConv
from torch_geometric.nn.norm import BatchNorm as PGBatchNorm, LayerNorm as PGLayerNorm
from torch_geometric.nn.aggr import SumAggregation
from torch_geometric.nn import global_mean_pool, global_max_pool

# ==============================================================================
# Part 0: config
# ==============================================================================
MODEL_CONFIG = {
    'attn_heads': 4,
    'cls_hidden': 128,
    'd_e': 128,
    'heads': 8,
    'hidden': 1024,
    'n_layers': 6,
    'norm_type': 'batch',
    'max_degree': 64,
    'max_spd': 32, 
    'dropout': 0.3,
    'droppath': 0.3, 
    'feat_drop': 0.1,
    'drop_edge_p': 0.0
}

NODE_IN_DIM = 47
EDGE_IN_DIM = 15

# ==============================================================================
# Part 1: Model Structure (CYP-GEMSite)
# ==============================================================================

class DropPath(nn.Module):
    def __init__(self, p: float = 0.0): super().__init__(); self.p = p
    def forward(self, x): return x

def make_norm(norm_type: str, dim: int):
    t=(norm_type or "none").lower()
    if t=="batch": return PGBatchNorm(dim)
    if t=="layer": return PGLayerNorm(dim, mode="node")
    return nn.Identity()

class GaussianSmearing(nn.Module):
    def __init__(self, start=0.0, stop=20.0, num_gaussians=32):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / ((stop - start) / (num_gaussians - 1))**2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.unsqueeze(-1) - self.offset
        return torch.exp(self.coeff * torch.pow(dist, 2))

class NodeProjector(nn.Module):
    def __init__(self, in_dim, hidden, feat_drop): super().__init__(); self.proj=nn.Sequential(nn.Linear(in_dim,hidden),nn.ReLU(),nn.BatchNorm1d(hidden),nn.Dropout(feat_drop))
    def forward(self, x): return self.proj(x)

class EdgeProjector(nn.Module):
    def __init__(self, in_dim, d_e, feat_drop): super().__init__(); self.proj=nn.Sequential(nn.Linear(in_dim,d_e),nn.ReLU(),nn.BatchNorm1d(d_e),nn.Dropout(feat_drop))
    def forward(self, e): return self.proj(e)

class EdgeFusion(nn.Module):
    def __init__(self, d_e): super().__init__(); self.gate_mlp=nn.Sequential(nn.Linear(d_e*2,d_e),nn.Sigmoid()); self.norm=nn.LayerNorm(d_e)
    def forward(self, handcraft_feat, spd_feat): 
        gate=self.gate_mlp(torch.cat([handcraft_feat,spd_feat],dim=-1))
        fused=gate*handcraft_feat+(1.0-gate)*spd_feat
        return self.norm(fused)

class EdgeUpdate(nn.Module):
    def __init__(self, node_dim, edge_dim, use_norm=True): 
        super().__init__(); self.node_norm = PGLayerNorm(node_dim, mode="node")
        self.mlp = nn.Sequential(nn.Linear(node_dim * 2 + edge_dim, edge_dim * 2), nn.ReLU())
        self.edge_norm = nn.LayerNorm(edge_dim) if use_norm else nn.Identity()
    def forward(self, x, edge_index, edge_attr): 
        src, dst = edge_index; xn = self.node_norm(x)
        m = torch.cat([xn[src], xn[dst], edge_attr], dim=-1)
        raw_output = self.mlp(m); delta, gate_logit = torch.chunk(raw_output, 2, dim=-1)
        return self.edge_norm(edge_attr + torch.sigmoid(gate_logit) * delta)

class NodeEdgeCrossAttention(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__(); assert dim % heads == 0; self.dim = dim; self.heads = heads
        self.d_head = dim // heads; self.scale = self.d_head ** -0.5
        self.q_proj = nn.Linear(dim, dim); self.k_proj = nn.Linear(dim, dim); self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim); self.aggr = SumAggregation(); self.dropout = nn.Dropout(dropout)
    def forward(self, q_nodes, k_edges, v_edges, edge_index):
        N = q_nodes.size(0); E = k_edges.size(0); dst, src = edge_index
        if E == 0: return torch.zeros((N, self.dim), device=q_nodes.device)
        q = self.q_proj(q_nodes).view(N, self.heads, self.d_head)
        k = self.k_proj(k_edges).view(E, self.heads, self.d_head); v = self.v_proj(v_edges).view(E, self.heads, self.d_head)
        attn_score = (q[dst] * k).sum(dim=-1) * self.scale
        attn_weights = self.dropout(softmax(attn_score, dst, num_nodes=N))
        weighted_v = (v * attn_weights.unsqueeze(-1)).view(E, self.dim)
        return self.out_proj(self.aggr(weighted_v, dst, dim_size=N))

class GlobalTransformerLayer(nn.Module):
    def __init__(self, dim, heads, max_dist=128, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True, dropout=dropout)
        self.dist_bias = nn.Embedding(max_dist, heads) # SPD Bias
        self.gaussian = GaussianSmearing(start=0.0, stop=20.0, num_gaussians=32)
        self.rbf_proj = nn.Linear(32, heads) # 3D RBF Bias
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim), nn.Dropout(dropout))
        self.num_heads = heads

    def forward(self, x, batch, dist_matrix, dist_3d=None):
        x_dense, mask = to_dense_batch(x, batch)
        batch_size, max_n = x_dense.size(0), x_dense.size(1)
        
        # A. SPD Bias
        dist_clamped = dist_matrix.clamp(0, self.dist_bias.num_embeddings - 1).long()
        bias_spd = self.dist_bias(dist_clamped).permute(0, 3, 1, 2)
        attn_bias = bias_spd
        
        # B. 3D RBF Bias
        if dist_3d is not None:
            rbf_feat = self.gaussian(dist_3d)
            bias_3d = self.rbf_proj(rbf_feat).permute(0, 3, 1, 2)
            attn_bias = attn_bias + bias_3d

        attn_bias = attn_bias.reshape(batch_size * self.num_heads, max_n, max_n)
        padding_mask = ~mask
        padding_mask_expanded = padding_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.num_heads, max_n, max_n)
        padding_mask_expanded = padding_mask_expanded.reshape(batch_size * self.num_heads, max_n, max_n)
        attn_bias = attn_bias.masked_fill(padding_mask_expanded, float("-inf"))
        
        h = self.norm1(x_dense)
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_bias)
        h = h + attn_out
        h = h + self.ffn(self.norm2(h))
        return h[mask]

class GraphormerBlock(nn.Module):
    def __init__(self, dim, edge_dim, heads, dropout, norm_type, droppath, max_degree):
        super().__init__(); self.norm=make_norm(norm_type,dim)
        self.attn=TransformerConv(dim,dim//heads,heads,edge_dim=edge_dim,dropout=dropout)
        self.ffn=nn.Sequential(nn.Linear(dim,dim*4),nn.GELU(),nn.Dropout(dropout),nn.Linear(dim*4,dim),nn.Dropout(dropout))
        self.stoch=DropPath(droppath); self.max_deg=int(max_degree); self.deg_emb=nn.Embedding(self.max_deg,dim)
        self.edge_upd = EdgeUpdate(node_dim=dim, edge_dim=edge_dim, use_norm=True)
    def _degree_embed(self, num_nodes, edge_index, device):
        deg=torch.zeros(num_nodes,dtype=torch.long,device=device).scatter_add_(0,edge_index[0],torch.ones_like(edge_index[0]))
        return self.deg_emb(deg.clamp_max(self.max_deg-1))
    def forward(self, x, edge_index, edge_attr, norm_type):
        x_in = x + self._degree_embed(x.size(0), edge_index, x.device); h = self.norm(x_in)
        h_attn = self.attn(h, edge_index, edge_attr); x_attn_res = x_in + self.stoch(h_attn)
        h_ffn = self.norm(x_attn_res) if norm_type in ["layer","batch"] else x_attn_res
        h_ffn_out = self.ffn(h_ffn); x_new = x_attn_res + self.stoch(h_ffn_out)
        ea_new = self.edge_upd(x_new, edge_index, edge_attr)
        return x_new, ea_new

class GraphormerModel(nn.Module):
    def __init__(self, node_in, edge_in, hidden, heads, n_layers, attn_heads=4, cls_hidden=128, dropout=0.3, d_e=128, norm_type="batch", droppath=0.1, feat_drop=0.2, max_degree=32, max_spd=64, drop_edge_p=0.0, **kwargs):
        super().__init__(); self.max_spd=max_spd; self.drop_edge_p=drop_edge_p
        self.node_proj=NodeProjector(node_in,hidden,feat_drop); self.edge_proj=EdgeProjector(edge_in,d_e,feat_drop)
        self.spatial_emb=nn.Embedding(max_spd, d_e); self.edge_fusion=EdgeFusion(d_e)
        self.layers=nn.ModuleList([GraphormerBlock(hidden,d_e,heads,dropout,norm_type,droppath*(i+1)/max(n_layers,1),max_degree) for i in range(n_layers)])
        
        self.global_layer = GlobalTransformerLayer(dim=hidden, heads=heads, max_dist=max_spd, dropout=dropout)
        
        self.edge_aggr = NodeEdgeCrossAttention(d_e, heads=attn_heads, dropout=dropout); self.d_e = d_e; self.drop=nn.Dropout(dropout)
        self.node_readout_proj = nn.Sequential(nn.Linear(hidden * 2, d_e), nn.ReLU(), nn.Dropout(dropout))
        self.global_pool_proj = nn.Sequential(nn.Linear(hidden * 2, d_e), nn.ReLU(), nn.Dropout(dropout))
        self.global_edge_proj = nn.Sequential(nn.Linear(d_e, d_e), nn.ReLU(), nn.Dropout(dropout))

        input_dim = int(4 * d_e)
        self.head = nn.Sequential(
            nn.Linear(input_dim, cls_hidden * 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(cls_hidden * 2, cls_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(cls_hidden, 1)
        )
    
    def forward_logits(self, data):
        x,ei,ea,spd = data.x,data.edge_index,data.edge_attr,data.spd
        if hasattr(data, 'batch') and data.batch is not None: batch = data.batch
        else: batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        x=self.node_proj(x); ea_handcraft=self.edge_proj(ea)
        
        spd_embedding=self.spatial_emb(spd.clamp_max(self.max_spd-1))
        ea=self.edge_fusion(ea_handcraft,spd_embedding)
        
        x0 = x.clone(); x_history = []; ea_history = []
        for blk in self.layers: 
            x, ea = blk(x, ei, ea, norm_type="layer"); x_history.append(x); ea_history.append(ea)
        
        x_combined = sum(x_history[-3:]) if len(x_history) >= 3 else sum(x_history)
        ea_combined = sum(ea_history[-3:]) if len(ea_history) >= 3 else sum(ea_history)
        
        dist_dense_spd = to_dense_adj(ei, batch, edge_attr=spd, max_num_nodes=None)
        
        if hasattr(data, 'pos') and data.pos is not None:
            pos_dense, _ = to_dense_batch(data.pos, batch)
            dist_3d_batch = torch.cdist(pos_dense, pos_dense, p=2.0)
        else:
            dist_3d_batch = None
            
        x_global_refined = self.global_layer(x_combined, batch, dist_dense_spd, dist_3d=dist_3d_batch)
        x_combined = x_combined + x_global_refined

        # --- Readout ---
        h_node = torch.cat([x_combined, x0], dim=-1)
        h_node_proj = self.node_readout_proj(h_node)
        ea_smart_aggr = self.edge_aggr(q_nodes=h_node_proj, k_edges=ea_combined, v_edges=ea_combined, edge_index=ei)
        g_feat_lifted = self.global_pool_proj(torch.cat([global_mean_pool(x_combined, batch), global_max_pool(x_combined, batch)], dim=-1))[batch]
        edge_batch = batch[ei[0]]
        g_edge_lifted = self.global_edge_proj(global_mean_pool(ea_combined, edge_batch))[batch]

        x_out = torch.cat([h_node_proj, ea_smart_aggr, g_feat_lifted, g_edge_lifted], dim=-1)
        h_dropped = self.drop(x_out)
        return self.head(h_dropped), None
    
    def forward(self, data):
        return self.forward_logits(data)[0]

# ==============================================================================
# Part 2: Feature Generator 
# ==============================================================================
class FeatureGenerator:
    def __init__(self, max_spd=64):
        self.max_spd = max_spd

    def _one_hot_index(self, idx: int, num_classes: int) -> torch.Tensor:
        v = torch.zeros(num_classes, dtype=torch.float32)
        if 0 <= idx < num_classes: v[idx] = 1.0
        return v

    def process_molecule(self, smiles_or_mol):
        if isinstance(smiles_or_mol, str):
            mol = Chem.MolFromSmiles(smiles_or_mol)
        else:
            mol = smiles_or_mol
        
        if mol is None: return None
        
        edge_index = self._get_adjacency_info(mol)
        
        spd = self._get_spd_matrix(mol, edge_index)
        
        pos = self._get_optimized_3d_coords(mol)

        x = self._get_node_features(mol)
        edge_attr = self._get_edge_features(mol)
        
        return Data(
            x=x, 
            edge_index=edge_index, 
            edge_attr=edge_attr, 
            spd=spd, 
            pos=pos,
            smiles=Chem.MolToSmiles(mol)
        )

    def _get_optimized_3d_coords(self, mol):
        num_atoms = mol.GetNumAtoms()
        try:
            mol_h = Chem.AddHs(mol)
            res = AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3())
            if res == -1: return torch.zeros((num_atoms, 3), dtype=torch.float32)
            
            AllChem.MMFFOptimizeMolecule(mol_h)
            mol_opt = Chem.RemoveHs(mol_h) 
            coords = mol_opt.GetConformer().GetPositions()
            return torch.tensor(coords, dtype=torch.float32)
        except:
            return torch.zeros((num_atoms, 3), dtype=torch.float32)

    def _get_spd_matrix(self, mol, edge_index):
        num_atoms = mol.GetNumAtoms()
        num_edges = edge_index.size(1)
        if num_atoms == 0 or num_edges == 0: return torch.zeros((0,), dtype=torch.long)
        
        G = nx.Graph()
        G.add_nodes_from(range(num_atoms))
        for bond in mol.GetBonds():
            G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            
        spd_matrix_np = nx.floyd_warshall_numpy(G, nodelist=range(num_atoms))
        spd_matrix_np[np.isinf(spd_matrix_np)] = self.max_spd + 1
        
        src = edge_index[0].numpy()
        dst = edge_index[1].numpy()
        spd_edge_values = spd_matrix_np[src, dst]
        
        spd_edge_values = np.clip(spd_edge_values, 0, self.max_spd)
        return torch.tensor(spd_edge_values, dtype=torch.long)

    def _get_node_features(self, mol):
        try:
            logp = Descriptors.MolLogP(mol); tpsa = Descriptors.TPSA(mol)
            mol_wt = Descriptors.MolWt(mol); num_rings = Descriptors.RingCount(mol)
        except: logp, tpsa, mol_wt, num_rings = 0.0, 0.0, 0.0, 0.0
        global_feats = torch.tensor([
            np.clip(logp / 10.0, -2.0, 2.0), np.clip(tpsa / 150.0, 0.0, 2.0),
            np.clip(mol_wt / 500.0, 0.0, 2.0), np.clip(num_rings / 5.0, 0.0, 2.0)
        ], dtype=torch.float32)
        
        atom_features = []
        for atom in mol.GetAtoms():
            symbol_list = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I']
            atom_type = self._one_hot_index(symbol_list.index(atom.GetSymbol()) if atom.GetSymbol() in symbol_list else 9, 10)
            hybrid_list = [HybridizationType.SP, HybridizationType.SP2, HybridizationType.SP3]
            hybrid = self._one_hot_index(hybrid_list.index(atom.GetHybridization()) if atom.GetHybridization() in hybrid_list else 3, 4)
            degree = self._one_hot_index(atom.GetDegree() if atom.GetDegree() <= 4 else 5, 6)
            h_count = self._one_hot_index(atom.GetTotalNumHs() if atom.GetTotalNumHs() <= 3 else 4, 5)
            aromatic = torch.tensor([1.0] if atom.GetIsAromatic() else [0.0], dtype=torch.float32)
            ring_count = self._one_hot_index(min(mol.GetRingInfo().NumAtomRings(atom.GetIdx()), 3), 4)
            implicit = self._one_hot_index(min(atom.GetImplicitValence(), 5), 6)
            gasteiger = atom.GetDoubleProp('_GasteigerCharge') if atom.HasProp('_GasteigerCharge') else 0.0
            if not np.isfinite(gasteiger): gasteiger = 0.0
            formal = torch.tensor([atom.GetFormalCharge()], dtype=torch.float32)
            radical = torch.tensor([atom.GetNumRadicalElectrons()], dtype=torch.float32)
            chiral_map = {ChiralType.CHI_TETRAHEDRAL_CCW: 0, ChiralType.CHI_TETRAHEDRAL_CW: 1, ChiralType.CHI_OTHER: 2}
            chiral = self._one_hot_index(chiral_map.get(atom.GetChiralTag(), 3), 4)
            feat = torch.cat([atom_type, hybrid, degree, h_count, aromatic, ring_count, implicit, 
                              torch.tensor([gasteiger]), formal, radical, chiral, global_feats])
            atom_features.append(feat)
        if len(atom_features) == 0: return torch.zeros((0, 47), dtype=torch.float32)
        return torch.stack(atom_features)

    def _get_edge_features(self, mol):
        feats = []
        def _oh_bool(flag): return torch.tensor([1., 0.]) if flag else torch.tensor([0., 1.])
        hybrid_map = {HybridizationType.SP: 0, HybridizationType.SP2: 1, HybridizationType.SP3: 2}
        combo_map = {(2, 2): 0, (1, 2): 1, (1, 1): 2, (0, 2): 3, (0, 1): 4}

        for b in mol.GetBonds():
            bt = b.GetBondType()
            if bt == Chem.BondType.SINGLE: b_type = self._one_hot_index(0, 4)
            elif bt == Chem.BondType.DOUBLE: b_type = self._one_hot_index(1, 4)
            elif bt == Chem.BondType.TRIPLE: b_type = self._one_hot_index(2, 4)
            elif bt == Chem.BondType.AROMATIC or b.GetIsAromatic(): b_type = self._one_hot_index(3, 4)
            else: b_type = self._one_hot_index(0, 4)
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            ai, aj = mol.GetAtomWithIdx(i), mol.GetAtomWithIdx(j)
            ci = ai.GetDoubleProp('_GasteigerCharge') if ai.HasProp('_GasteigerCharge') else 0.0
            cj = aj.GetDoubleProp('_GasteigerCharge') if aj.HasProp('_GasteigerCharge') else 0.0
            polarity = torch.tensor([abs(ci - cj)])
            hi, hj = ai.GetHybridization(), aj.GetHybridization()
            key = tuple(sorted((hybrid_map.get(hi, 99), hybrid_map.get(hj, 99))))
            combo = self._one_hot_index(combo_map.get(key, 5), 6)
            in_ring = _oh_bool(b.IsInRing())
            conjugated = _oh_bool(b.GetIsConjugated())
            e = torch.cat([b_type, polarity, combo, in_ring, conjugated]) # 15 dim
            feats.append(e); feats.append(e)
        if len(feats) == 0: return torch.zeros((0, 15), dtype=torch.float32)
        return torch.stack(feats)

    def _get_adjacency_info(self, mol):
        edges = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edges.append([i, j]); edges.append([j, i])
        if len(edges) == 0: return torch.empty((2, 0), dtype=torch.long)
        return torch.tensor(edges, dtype=torch.long).t().contiguous()

# ==============================================================================
# Part 3: Predictor and Visualization
# ==============================================================================

class SoMPredictor:
    def __init__(self, model_path, threshold=0.5, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"[Init] Loading CYP-GEMSite model from {model_path} to {self.device}...")
        
        self.model = GraphormerModel(node_in=NODE_IN_DIM, edge_in=EDGE_IN_DIM, **MODEL_CONFIG)
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
            self.model.load_state_dict(state_dict)
            if isinstance(checkpoint, dict) and 'threshold' in checkpoint:
                threshold = float(checkpoint['threshold'])
                print(f"[Info] Found saved threshold in checkpoint: {threshold:.4f}")
            print("[Success] Model weights loaded.")
        except Exception as e:
            print(f"[Error] Failed to load weights: {e}")
            raise e
            
        self.model.to(self.device)
        self.model.eval()
        self.threshold = threshold
        self.feat_gen = FeatureGenerator(max_spd=MODEL_CONFIG['max_spd'])

    def predict(self, smiles_or_mol):
        try:
            data = self.feat_gen.process_molecule(smiles_or_mol)
            if data is None: return None, None
            
            data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
            data = data.to(self.device)
            
            with torch.no_grad():
                logits = self.model(data)
                probs = torch.sigmoid(logits).cpu().numpy()
            
            atom_probs = []
            mol = Chem.MolFromSmiles(data.smiles) 
            for i, p in enumerate(probs):
                atom_probs.append({
                    "atom_idx": i, 
                    "symbol": mol.GetAtomWithIdx(i).GetSymbol(),
                    "prob": float(p), 
                    "is_som": bool(p >= self.threshold)
                })
            return mol, atom_probs
        except Exception as e:
            print(f"Prediction Error: {e}")
            return None, None

    def visualize_and_save(self, mol, atom_probs, output_base_path):
        
        csv_path = output_base_path + ".csv"
        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["Atom_Index", "Symbol", "Probability", "Is_SoM_Above_Thr"])
                for p in atom_probs:
                    writer.writerow([p['atom_idx'], p['symbol'], f"{p['prob']:.6f}", p['is_som']])
            print(f"  >> Saved Data to: {csv_path}")
        except Exception as e:
            print(f"  [Error] Failed to save CSV file: {e}")

        png_path = output_base_path + ".png"
        
        soms = [x for x in atom_probs if x['is_som']]
        soms.sort(key=lambda x: x['prob'], reverse=True)
        hit_ats = [x['atom_idx'] for x in soms]
        atom_colors = {}
        
        cmap = plt.get_cmap('Reds')
        n_hits = len(soms)
        for rank, item in enumerate(soms):
            idx = item['atom_idx']
            intensity = 1.0 - (0.6 * (rank / max(n_hits, 1)))
            rgba = cmap(intensity)
            atom_colors[idx] = (rgba[0], rgba[1], rgba[2])

        try:
            from rdkit.Chem import Draw
            width, height = 1500, 1500
            drawer = Draw.rdMolDraw2D.MolDraw2DCairo(width, height)
            dopts = drawer.drawOptions()
            dopts.useBWAtomPalette()
            dopts.padding = 0.1
            dopts.bondLineWidth = 2
            dopts.addAtomIndices = True  
            dopts.annotationFontScale = 0.8
            
            Draw.rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol, highlightAtoms=hit_ats, highlightAtomColors=atom_colors)
            drawer.FinishDrawing()
            drawer.WriteDrawingText(png_path)
            print(f"  >> Saved Image to: {png_path}")
        except Exception as e:
            print(f"  [Error] Visualization failed: {e}")

# ==============================================================================
# Part 4: Main Function
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="CYP-GEMSite Predictor (SDF Support & CSV/PNG Export)")
    parser.add_argument("--model", type=str, required=True, help="Path to best_model.pt")
    parser.add_argument("--input", type=str, required=True, help="SMILES/SDF file or single SMILES string")
    parser.add_argument("--output_dir", type=str, default="./predictions")
    parser.add_argument("--threshold", type=float, default=0.5, help="Prediction threshold")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    predictor = SoMPredictor(args.model, threshold=args.threshold)
    
    molecules_to_process = []
    input_path = args.input
    if os.path.isfile(input_path):
        ext = os.path.splitext(input_path)[1].lower()
        if ext == '.sdf':
            print(f"[Input] Reading SDF file: {input_path}")
            suppl = Chem.SDMolSupplier(input_path)
            molecules_to_process = [m for m in suppl if m is not None]
        else:
            print(f"[Input] Reading text file: {input_path}")
            with open(input_path, 'r') as f: 
                molecules_to_process = [line.strip().split()[0] for line in f if line.strip()]
    else:
        molecules_to_process = [args.input]

    print("\n" + "="*60 + f"\nStart Prediction (Thr={predictor.threshold}, Total Mols={len(molecules_to_process)})\n" + "="*60)

    for idx, item in enumerate(molecules_to_process):
        try:
            mol, probs = predictor.predict(item)
            if mol is None: 
                print(f"[Skip] Failed to process molecule index {idx}")
                continue

            try: inchi_key = Chem.MolToInchiKey(mol)
            except: inchi_key = f"Unknown_InChIKey_{idx}"
            if not inchi_key: inchi_key = f"Mol_{idx}"

            print(f"\nProcessing [{idx+1}]: {inchi_key}")
            soms = [p for p in probs if p['is_som']]
            soms.sort(key=lambda x: x['prob'], reverse=True)
            print(f"  > Found {len(soms)} potential SoMs")
            
            base_path = os.path.join(args.output_dir, inchi_key)
            predictor.visualize_and_save(mol, probs, base_path)

        except Exception as e:
            print(f"  [Error] Processing molecule {idx} failed: {e}")

if __name__ == "__main__":
    main()
