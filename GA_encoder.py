import numpy as np
import torch
from torch_geometric.data import Data

#processed_data = torch.load('processed_data', map_location='cpu', weights_only=False)

def encode_atom(position, atom_features):

    mv = np.zeros(16)

    mv[12] = position[0]
    mv[13] = position[1]
    mv[14] = position[2]

    mv[11] = 1

    return mv


def encode_edge(displacements):

    mv = np.zeros(16)

    mv[12] = displacements[0]
    mv[13] = displacements[1]
    mv[14] = displacements[2]

    mv[11] = 0

    return mv


def encode_lattice(volume):

    mv = np.zeros(16)

    mv[15] = volume

    return mv


def encode_point(position_2d):
    """
    Embed a single 2D AirfRANS coordinate as a PGA trivector — same
    representation convention as encode_atom (position in mv[12:15],
    homogeneous marker in mv[11]), with z fixed to 0 since the airfoil
    domain is 2D.
    """
    mv = np.zeros(16)

    mv[12] = position_2d[0]
    mv[13] = position_2d[1]
    mv[14] = 0.0

    mv[11] = 1

    return mv


def encode_points_batch(positions_2d: np.ndarray) -> np.ndarray:
    """
    Vectorized version of encode_point for an (N, 2) coordinate array —
    avoids a per-point Python loop at the point counts AirfRANS uses
    (thousands of nodes per sample).
    """
    N = positions_2d.shape[0]
    mv = np.zeros((N, 16), dtype=np.float32)
    mv[:, 12] = positions_2d[:, 0]
    mv[:, 13] = positions_2d[:, 1]
    mv[:, 14] = 0.0
    mv[:, 11] = 1.0
    return mv


def encode_points_batch_torch(positions_2d: torch.Tensor) -> torch.Tensor:
    """
    Torch-native version of encode_points_batch — safe to call inside a
    model's forward pass on GPU tensors without a CPU/numpy round-trip.
    """
    N = positions_2d.shape[0]
    mv = torch.zeros(N, 16, dtype=positions_2d.dtype, device=positions_2d.device)
    mv[:, 12] = positions_2d[:, 0]
    mv[:, 13] = positions_2d[:, 1]
    mv[:, 11] = 1.0
    return mv



def prepare_graph_data(processed_entry: dict) -> dict:
    """
    Takes a processed entry and returns everything needed
    for a torch_geometric Data object.
    """
    N = len(processed_entry['positions'])
    E = len(processed_entry['src'])

    # --- node GA multivectors ---
    node_mvs = np.zeros((N, 16))
    for i, pos in enumerate(processed_entry['positions']):
        node_mvs[i] = encode_atom(pos, processed_entry['atom_features'][i])

    # --- node scalar features (stored separately) ---
    node_scalars = processed_entry['atom_features']  # (N, 7)

    # --- edge index ---
    edge_index = np.array([
        processed_entry['src'],
        processed_entry['dst']
    ])  # (2, E)

    # --- edge GA multivectors ---
    edge_mvs = np.zeros((E, 16))
    for k, disp in enumerate(processed_entry['displacements']):
        edge_mvs[k] = encode_edge(disp)

    # --- edge scalar features ---
    distances = np.linalg.norm(
        processed_entry['displacements'], axis=1
    )  # (E,)

    # --- lattice (graph-level) ---
    lattice_mv = encode_lattice(
        processed_entry['lattice_volume']
    )  # (16,)

    # angles normalized to [0,1] — angles are in degrees, max 180
    lattice_angles = processed_entry['lattice_angles']  # hint: where are angles stored?
    lattice_matrix = processed_entry['lattice_matrix'].flatten()  # (9,)

    # --- target ---
    y = processed_entry['targets']  # log10(IC)

    return {
        'node_mvs': node_mvs,  # (N, 16)
        'node_scalars': node_scalars,  # (N, 7)
        'edge_index': edge_index,  # (2, E)
        'edge_mvs': edge_mvs,  # (E, 16)
        'distances': distances,  # (E,)
        'lattice_mv': lattice_mv,  # (16,)
        'lattice_angles': lattice_angles,  # (3,)
        'lattice_matrix': lattice_matrix,  # (9,)
        'y': y,  # scalar
    }


#processed_graph_data = []

#for entry in processed_data:

    #processed_graph_data.append(prepare_graph_data(entry))





def dict_to_pyg_data(d):
    """
    Converts a single feature dictionary into a PyG Data object.
    """
    return Data(
        # Standard Node Features
        x= torch.tensor(d['node_scalars'], dtype=torch.float),  # [N, 7]
        node_mvs=torch.tensor(d['node_mvs'], dtype=torch.float),  # [N, 16]

        # Standard Edge Features
        edge_index=torch.tensor(d['edge_index'], dtype=torch.long),  # [2, E]
        edge_mvs=torch.tensor(d['edge_mvs'], dtype=torch.float),  # [E, 16]
        edge_attr=torch.tensor(d['distances'], dtype=torch.float).view(-1, 1),  # [E, 1]

        # Global Crystal Features (Lattice)
        # We use .view(1, -1) so they batch as [Batch_Size, Feature_Dim]
        lattice_mv=torch.tensor(d['lattice_mv'], dtype=torch.float).view(1, -1),
        lattice_angles=torch.tensor(d['lattice_angles'], dtype=torch.float).view(1, -1),
        lattice_matrix=torch.tensor(d['lattice_matrix'], dtype=torch.float).view(1, -1),

        # Target
        y=torch.tensor([d['y']], dtype=torch.float)  # [1]
    )


# Assume 'crystal_dicts' is your list of dictionaries
#data = [dict_to_pyg_data(d) for d in processed_graph_data]


#torch.save(data, 'data')

