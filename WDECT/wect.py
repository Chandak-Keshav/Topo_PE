import torch
from dataclasses import dataclass
from torch import nn
import geotorch


def segment_add_coo(data: torch.Tensor, index: torch.LongTensor) -> torch.Tensor:
    """
    Replacement for torch_scatter's segment_add_coo.
    """
    num_segments = index.max().item() + 1
    output_shape = (num_segments,) + data.shape[1:]
    output = torch.zeros(output_shape, dtype=data.dtype, device=data.device)
    output.index_add_(0, index, data)
    return output

@dataclass(frozen=True)
class ECTConfig:
    bump_steps: int = 32
    radius: float = 1.1
    ect_type: str = "points"
    normalized: bool = False
    fixed: bool = True

@dataclass()
class Batch:
    x: torch.FloatTensor
    batch: torch.LongTensor
    edge_index: torch.LongTensor | None = None
    face: torch.LongTensor | None = None
    node_weights: torch.FloatTensor | None = None
    edge_weights: torch.FloatTensor | None = None
    face_weights: torch.FloatTensor | None = None


def compute_ecc(nh: torch.FloatTensor,
                index: torch.LongTensor,
                lin: torch.FloatTensor,
                scale: float = 100) -> torch.FloatTensor:
    ecc = torch.nn.functional.sigmoid(scale * (lin - nh))
    ecc = ecc.movedim(0, 2).movedim(0, 1)
    return segment_add_coo(ecc, index)


def compute_ect_points(batch: Batch, v: torch.FloatTensor, lin: torch.FloatTensor):
    weights = batch.node_weights if batch.node_weights is not None else torch.ones(batch.x.size(0), device=batch.x.device)
    nh = (batch.x * weights.unsqueeze(1)) @ v
    return compute_ecc(nh, batch.batch, lin)


def compute_ect_edges(batch: Batch, v: torch.FloatTensor, lin: torch.FloatTensor):
    w_node = batch.node_weights if batch.node_weights is not None else torch.ones(batch.x.size(0), device=batch.x.device)
    nh = (batch.x * w_node.unsqueeze(1)) @ v
    eh, _ = nh[batch.edge_index].max(dim=0)
    w_edge = batch.edge_weights if batch.edge_weights is not None else torch.ones(eh.size(0), device=eh.device)
    eh = eh * w_edge.unsqueeze(1)
    batch_idx = batch.batch[batch.edge_index[0]]
    return compute_ecc(nh, batch.batch, lin) - compute_ecc(eh, batch_idx, lin)


def compute_ect_faces(batch: Batch, v: torch.FloatTensor, lin: torch.FloatTensor):
    w_node = batch.node_weights if batch.node_weights is not None else torch.ones(batch.x.size(0), device=batch.x.device)
    nh = (batch.x * w_node.unsqueeze(1)) @ v
    eh, _ = nh[batch.edge_index].max(dim=0)
    fh, _ = nh[batch.face].max(dim=0)
    w_edge = batch.edge_weights if batch.edge_weights is not None else torch.ones(eh.size(0), device=eh.device)
    w_face = batch.face_weights if batch.face_weights is not None else torch.ones(fh.size(0), device=fh.device)
    eh = eh * w_edge.unsqueeze(1)
    fh = fh * w_face.unsqueeze(1)
    idx_e = batch.batch[batch.edge_index[0]]
    idx_f = batch.batch[batch.face[0]]
    return (
        compute_ecc(nh, batch.batch, lin)
        - compute_ecc(eh, idx_e, lin)
        + compute_ecc(fh, idx_f, lin)
    )


def normalize(ect: torch.FloatTensor) -> torch.FloatTensor:
    return ect / torch.amax(ect, dim=(2, 3), keepdim=True)

class ECTLayer(nn.Module):
    def __init__(self, config: ECTConfig, v: torch.FloatTensor | None = None):
        super().__init__()
        self.config = config
        self.lin = nn.Parameter(
            torch.linspace(-config.radius, config.radius, config.bump_steps)
                 .view(-1, 1, 1, 1), requires_grad=False
        )
        if v is not None:
            if v.ndim == 2:
                v = v.unsqueeze(0)
            if config.fixed:
                self.v = nn.Parameter(v.movedim(-1, -2), requires_grad=False)
            else:
                self.v = nn.Parameter(torch.zeros_like(v.movedim(-1, -2)))
                geotorch.constraints.sphere(self, "v", radius=config.radius)
                self.v = v.movedim(-1, -2)
        else:
            raise ValueError("Directions tensor `v` must be provided.")

        if config.ect_type == "points":
            self.compute_ect = compute_ect_points
        elif config.ect_type == "edges":
            self.compute_ect = compute_ect_edges
        else:
            self.compute_ect = compute_ect_faces

    def forward(self, batch: Batch) -> torch.FloatTensor:
        ect = self.compute_ect(batch, self.v.movedim(-1, -2), self.lin)
        return normalize(ect) if self.config.normalized else ect.squeeze()
