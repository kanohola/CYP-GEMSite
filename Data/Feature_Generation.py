import os
import re
import ast
import pickle
import networkx as nx 
from tqdm import tqdm 
import numpy as np
import torch

import torch.nn.functional as F  

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.rdchem import HybridizationType, ChiralType

from torch_geometric.data import Dataset, Data

# ==============================================================================
# Part 1: Functions
# ==============================================================================
def _one_hot_index(idx: int, num_classes: int) -> torch.Tensor:
    """创建指定索引为1的one-hot向量。"""
    v = torch.zeros(num_classes, dtype=torch.float32)
    if 0 <= idx < num_classes: v[idx] = 1.0
    return v

# ==============================================================================
# Part 2: PyTorch Geometric Dataset 
# ==============================================================================
class CYP_GEMSite_Dataset(Dataset): 
    
    def __init__(self, root, filename, transform=None, pre_transform=None, pre_filter=None):
        self.root = root
        self.filename = filename
        self.max_spd = 64 
        super().__init__(root, transform, pre_transform, pre_filter)
        with open(self.processed_paths[0], 'rb') as f:
            self._data_list = pickle.load(f)

    @property
    def raw_file_names(self):
        return self.filename

    @property
    def processed_file_names(self):
        
        return 'new.pkl'

    def download(self):
        pass

    def process(self):
        print("Processing raw data with V7-Graphormer-3D features...")
        print("  - 47-dim node features (x)")
        print("  - 15-dim edge features (edge_attr)")
        print("  - Optimized 3D coordinates (pos)")
        print("  - Edge-aligned Shortest Path Distance Feature (spd) [Shape: E]")
        print("  - [NEW] Pairwise 3D Distance Matrix (dist_3d) [Shape: N, N]")
        
        data_list = []
        mols = Chem.SDMolSupplier(self.raw_paths[0], removeHs=True)

        for i, mol in enumerate(tqdm(mols, desc="Processing Molecules")):
            if mol is None: continue
            
            smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
            try: AllChem.ComputeGasteigerCharges(mol)
            except: pass

            x = self._get_node_features(mol)
            edge_attr = self._get_edge_features(mol)
            edge_index = self._get_adjacency_info(mol)
            y = self._get_labels(mol)

            pos = self._get_optimized_3d_coords(mol, i) 
            
            spd = self._get_spd_matrix(mol, edge_index)

            data = Data(x=x, 
                        edge_index=edge_index, 
                        edge_attr=edge_attr, 
                        y=y,
                        pos=pos,
                        spd=spd,
                        smiles=smiles
                        )
            
            data_list.append(data)

        with open(self.processed_paths[0], 'wb') as f:
            pickle.dump(data_list, f)
        print(f"Processing finished. Data saved to {self.processed_paths[0]}")

    def _get_optimized_3d_coords(self, mol, mol_idx):
        num_atoms = mol.GetNumAtoms()
        try:
            mol_h = Chem.AddHs(mol)
            
            status = AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3())
            if status == -1: 
                # print(f"Warning: 3D embedding failed for mol {mol_idx}. Using 2D (zeros).")
                return torch.zeros((num_atoms, 3), dtype=torch.float32)

            
            AllChem.MMFFOptimizeMolecule(mol_h)
            
           
            mol_opt = Chem.RemoveHs(mol_h)
            
            
            coords = mol_opt.GetConformer().GetPositions()
            return torch.tensor(coords, dtype=torch.float32)
            
        except Exception as e:
            # print(f"Warning: 3D optimization failed for mol {mol_idx}. Error: {e}. Using 2D (zeros).")
            return torch.zeros((num_atoms, 3), dtype=torch.float32)

    def _get_spd_matrix(self, mol, edge_index):
        num_atoms = mol.GetNumAtoms()
        num_edges = edge_index.size(1)

        if num_atoms == 0:
            return torch.zeros((0,), dtype=torch.long)
        
        if num_edges == 0:
            return torch.zeros((0,), dtype=torch.long)
            
        G = nx.Graph()
        G.add_nodes_from(range(num_atoms))
        for bond in mol.GetBonds():
            G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

        spd_matrix_np = nx.floyd_warshall_numpy(G, nodelist=range(num_atoms))
        
        spd_matrix_np[np.isinf(spd_matrix_np)] = self.max_spd + 1 
        
        src = edge_index[0].numpy()
        dst = edge_index[1].numpy()
        
        spd_edge_values_np = spd_matrix_np[src, dst]

        spd_edge_values_np = np.clip(spd_edge_values_np, 0, self.max_spd)
        
        return torch.tensor(spd_edge_values_np, dtype=torch.long)

    def _get_node_features(self, mol):
        
        try:
            logp = Descriptors.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            mol_wt = Descriptors.MolWt(mol)
            num_rings = Descriptors.RingCount(mol)
        except Exception as e: 
            logp, tpsa, mol_wt, num_rings = 0.0, 0.0, 0.0, 0.0

        global_feats = torch.tensor([
            np.clip(logp / 10.0, -2.0, 2.0),
            np.clip(tpsa / 150.0, 0.0, 2.0),
            np.clip(mol_wt / 500.0, 0.0, 2.0),
            np.clip(num_rings / 5.0, 0.0, 2.0)
        ], dtype=torch.float32) # [4 dim]
        
        atom_features = []
            
        for i, atom in enumerate(mol.GetAtoms()):
            symbol_list = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I']; atom_type_oh = _one_hot_index(symbol_list.index(atom.GetSymbol()) if atom.GetSymbol() in symbol_list else 9, 10)
            hybrid_list = [HybridizationType.SP, HybridizationType.SP2, HybridizationType.SP3]; hybrid_oh = _one_hot_index(hybrid_list.index(atom.GetHybridization()) if atom.GetHybridization() in hybrid_list else 3, 4)
            degree_idx = atom.GetDegree() if atom.GetDegree() <= 4 else 5; degree_oh = _one_hot_index(degree_idx, 6)
            h_count_idx = atom.GetTotalNumHs() if atom.GetTotalNumHs() <= 3 else 4; h_count_oh = _one_hot_index(h_count_idx, 5) # (隐式H)
            aromatic_feat = torch.tensor([1.0] if atom.GetIsAromatic() else [0.0], dtype=torch.float32)
            ring_count_idx = mol.GetRingInfo().NumAtomRings(atom.GetIdx()); ring_count_idx = ring_count_idx if ring_count_idx <= 2 else 3; ring_count_oh = _one_hot_index(ring_count_idx, 4)
            implicit_valence = atom.GetImplicitValence(); implicit_valence_idx = implicit_valence if implicit_valence <= 4 else 5; implicit_valence_oh = _one_hot_index(implicit_valence_idx, 6)
            gasteiger_charge = atom.GetDoubleProp('_GasteigerCharge') if atom.HasProp('_GasteigerCharge') else 0.0
            if not np.isfinite(gasteiger_charge): gasteiger_charge = 0.0 
            gasteiger_charge_feat = torch.tensor([gasteiger_charge], dtype=torch.float32)
            formal_charge_feat = torch.tensor([atom.GetFormalCharge()], dtype=torch.float32)
            radical_electrons_feat = torch.tensor([atom.GetNumRadicalElectrons()], dtype=torch.float32)
            chiral_tag = atom.GetChiralTag()
            chiral_map = {
                ChiralType.CHI_TETRAHEDRAL_CCW: 0,
                ChiralType.CHI_TETRAHEDRAL_CW: 1,
                ChiralType.CHI_OTHER: 2,
            }
            chiral_oh = _one_hot_index(chiral_map.get(chiral_tag, 3), 4)

            final_atom_feature = torch.cat([
                atom_type_oh,         # 10
                hybrid_oh,            # 4
                degree_oh,            # 6
                h_count_oh,           # 5
                aromatic_feat,        # 1
                ring_count_oh,        # 4
                implicit_valence_oh,  # 6
                gasteiger_charge_feat,# 1
                formal_charge_feat,   # 1
                radical_electrons_feat, # 1
                chiral_oh,            # 4
                global_feats          # 4
            ], dim=0) # Total: 47
            atom_features.append(final_atom_feature)
        
        if len(atom_features) == 0: return torch.zeros((0, 47), dtype=torch.float32)
        return torch.stack(atom_features, dim=0)

    def _get_edge_features(self, mol):
        
        feats=[]

        def _one_hot_bool(flag: bool) -> torch.Tensor:
            return torch.tensor([1., 0.], dtype=torch.float32) if flag else torch.tensor([0., 1.], dtype=torch.float32)

        hybrid_map = {HybridizationType.SP: 0, HybridizationType.SP2: 1, HybridizationType.SP3: 2}
        combo_map = {(2, 2): 0, (1, 2): 1, (1, 1): 2, (0, 2): 3, (0, 1): 4}

        for b in mol.GetBonds():
            i,j=b.GetBeginAtomIdx(),b.GetEndAtomIdx(); ai,aj=mol.GetAtomWithIdx(i),mol.GetAtomWithIdx(j); bt=b.GetBondType()
            
            if bt==Chem.BondType.SINGLE:bond_type_oh=_one_hot_index(0,4)
            elif bt==Chem.BondType.DOUBLE:bond_type_oh=_one_hot_index(1,4)
            elif bt==Chem.BondType.TRIPLE:bond_type_oh=_one_hot_index(2,4)
            elif bt==Chem.BondType.AROMATIC or b.GetIsAromatic():bond_type_oh=_one_hot_index(3,4)
            else:bond_type_oh=_one_hot_index(0,4)
            
            charge_i = ai.GetDoubleProp('_GasteigerCharge') if ai.HasProp('_GasteigerCharge') else 0.0
            charge_j = aj.GetDoubleProp('_GasteigerCharge') if aj.HasProp('_GasteigerCharge') else 0.0
            bond_polarity_feat = torch.tensor([abs(charge_i - charge_j)], dtype=torch.float32)
            
            h_i = ai.GetHybridization(); h_j = aj.GetHybridization()
            key = tuple(sorted((hybrid_map.get(h_i, 99), hybrid_map.get(h_j, 99))))
            hybrid_combo_oh = _one_hot_index(combo_map.get(key, 5), 6)
            
            is_in_ring_oh = _one_hot_bool(b.IsInRing())
            is_conjugated_oh = _one_hot_bool(b.GetIsConjugated())

            e=torch.cat([
                bond_type_oh,         # 4
                bond_polarity_feat,   # 1
                hybrid_combo_oh,      # 6
                is_in_ring_oh,        # 2
                is_conjugated_oh,     # 2
            ], dim=0) # Total: 15
            
            feats.append(e)
            feats.append(e)
            
        if len(feats)==0: return torch.zeros((0,15),dtype=torch.float32)
        return torch.stack(feats,dim=0)
        
    def _get_adjacency_info(self, mol):
        edges=[]
        for bond in mol.GetBonds():
            i=bond.GetBeginAtomIdx()
            j=bond.GetEndAtomIdx()
            edges.append([i,j])
            edges.append([j,i])
        if len(edges)==0:
            return torch.empty((2,0),dtype=torch.long)
        return torch.tensor(edges,dtype=torch.long).t().contiguous()
        
    def _get_labels(self, mol):
        
        try:
            som_str=mol.GetProp("Merged_SOMs")
            try:
                atom_indices=ast.literal_eval(som_str)
            except:
                match=re.search(r'\[([\d,\s]+)\]',som_str)
                if match:
                    atom_indices=[int(x.strip()) for x in match.group(1).split(',')]
                else:
                    atom_indices=[int(x.strip()) for x in som_str.split(',') if x.strip().isdigit()]
            if not isinstance(atom_indices,list):
                atom_indices=[atom_indices]
        except(KeyError,ValueError,SyntaxError,TypeError):
            atom_indices=[]
        
        y=np.zeros(len(mol.GetAtoms()),dtype=int)
        for idx in atom_indices:
            if 0<=idx<len(y):
                y[idx]=1
            else:
                mol_name=mol.GetProp('_Name') if mol.HasProp('_Name') else 'Unnamed'
                print(f"Warning: Invalid atom index {idx} in molecule {mol_name} (num_atoms: {len(y)})")
        return torch.tensor(y, dtype=torch.float32)
        
    def len(self):
        return len(self._data_list)
        
    def get(self, idx):
        return self._data_list[idx]

# ==============================================================================
# Part 4: Main Function
# ==============================================================================
if __name__ == '__main__':
    ROOT_DIR = '' 
    SDF_FILE = os.path.join(ROOT_DIR, '')
    

    os.makedirs(ROOT_DIR, exist_ok=True)
    

    print("Start creating a dataset...")
    if not os.path.exists(SDF_FILE):
        print(f"Error: Original SDF file not found. Please check the path: {SDF_FILE}")
    else:
        dataset = CYP_GEMSite_Dataset(root=ROOT_DIR, filename=SDF_FILE)
        
        print("\nDataset loading/processing successfully!")
        print(f"Number of graphs (molecules) in the dataset: {len(dataset)}")
        if len(dataset) > 0:
            print("\nExample data for the first graph:")
            first_graph = dataset[0]
            print(first_graph)
            print(f"  - Shape of node feature (X): {first_graph.x.shape}")
            print(f"  - The shape of the edge index (edge_index): {first_graph.edge_index.shape}")
            print(f"  - The shape of the edge feature (edge_attr): {first_graph.edge_attr.shape}")
            print(f"  - The shape of the label (y): {first_graph.y.shape}")
            
            if hasattr(first_graph, 'pos'):
                print(f"  - The shape of the 3D pos: {first_graph.pos.shape}")
            else:
                print(f"  - [!!!Error!!!] 3D coordinates (pos) not found!")
                
            if hasattr(first_graph, 'spd'):
                print(f"  -  The shape of the SPD feature (spd): {first_graph.spd.shape}")
                if first_graph.spd.shape[0] == first_graph.edge_index.shape[1]:
                    print(f"  - [Success] SPD count and edge count match: {first_graph.spd.shape[0]}")
                else:
                    print(f"  - [!!!Error!!!] SPD count ({first_graph.spd.shape[0]}) and edge count ({first_graph.edge_index.shape[1]}) don't match!")
            else:
                print(f"  - [!!!Error!!!] No SPD feature (spd) found!")
            
            if first_graph.x.shape[0] == first_graph.y.shape[0]:
                print(f"  - [Success] The number of nodes (X) and the number of tags (y) match: {first_graph.x.shape[0]}")
            else:
                print(f"  - [!!!Error!!!] The number of nodes (X) {first_graph.x.shape[0]} and the number of tags (y) {first_graph.y.shape[0]} don't match!")