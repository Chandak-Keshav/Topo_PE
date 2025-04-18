from typing import Literal

import geotorch
import torch
from torch import nn
# from torch_scatter import segment_add_coo

from tests.ect import ECTConfig, Batch, normalize


def segment_add_coo(data: torch.Tensor, index: torch.LongTensor) -> torch.Tensor:
    """
    Replacement for torch_scatter's segment_add_coo.

    Parameters
    ----------
    data : torch.Tensor
        Input tensor of shape [N, ...], where `N` is the number of elements.
    index : torch.LongTensor
        Indices tensor of shape [N] indicating the group for each data element.

    Returns
    -------
    torch.Tensor
        Aggregated tensor with summed values for each unique index.
    """
    # Determine the number of unique indices/groups
    num_segments = index.max().item() + 1

    # Create an output tensor initialized to zeros
    output_shape = (num_segments,) + data.shape[1:]
    output = torch.zeros(output_shape, dtype=data.dtype, device=data.device)

    # Perform the segmented addition
    output.index_add_(0, index, data)

    return output

def compute_wecc(
    nh: torch.FloatTensor,
    index: torch.LongTensor,
    lin: torch.FloatTensor,
    weight: torch.FloatTensor,
    scale: float = 500,
):
    """Computes the Weighted Euler Characteristic Curve.

    Parameters
    ----------
    nh : torch.FloatTensor
        The node heights, computed as the inner product of the node coordinates
        x and the direction vector v.
    index: torch.LongTensor
        The index that indicates to which pointcloud a node height belongs. For
        the node heights it is the same as the batch index, for the higher order
        simplices it will have to be recomputed.
    lin: torch.FloatTensor
        The discretization of the interval [-1,1] each node height falls in this
        range due to rescaling in normalizing the data.
    weight: torch.FloatTensor
        The weight of the node, edge or face. It is the maximum of the node
        weights for the edges and faces.
    scale: torch.FloatTensor
        A single number that scales the sigmoid function by multiplying the
        sigmoid with the scale. With high (100>) values, the ect will resemble a
        discrete ECT and with lower values it will smooth the ECT.
    """
    ecc = torch.nn.functional.sigmoid(scale * torch.sub(lin, nh)) * weight.view(
        1, -1, 1
    )
    ecc = ecc.movedim(0, 2).movedim(0, 1)
    return segment_add_coo(ecc, index)


def compute_wect(
    batch: Batch,
    v: torch.FloatTensor,
    lin: torch.FloatTensor,
    wect_type: Literal["points"] | Literal["edges"] | Literal["faces"],
):
    """
    Computes the Weighted Euler Characteristic Transform of a batch of point
    clouds.

    Parameters
    ----------
    batch : Batch
        A batch of data containing the node coordinates, batch index,
        edge_index, face, and node weights.
    v: torch.FloatTensor
        The direction vector that contains the directions.
    lin: torch.FloatTensor
        The discretization of the interval [-1,1] each node height falls in this
        range due to rescaling in normalizing the data.
    wect_type: str
        The type of WECT to compute. Can be "points", "edges", or "faces".
    """
    nh = batch.x @ v
    if wect_type in ["edges", "faces"]:
        edge_weights, _ = batch.node_weights[batch.edge_index].max(axis=0)
        eh, _ = nh[batch.edge_index].min(dim=0)
    if wect_type == "faces":
        face_weights, _ = batch.node_weights[batch.face].max(axis=0)
        fh, _ = nh[batch.face].min(dim=0)

    if wect_type == "points":
        return compute_wecc(nh, batch.batch, lin, batch.node_weights)
    if wect_type == "edges":
        # noinspection PyUnboundLocalVariable
        return compute_wecc(
            nh, batch.batch, lin, batch.node_weights
        ) - compute_wecc(
            eh, batch.batch[batch.edge_index[0]], lin, edge_weights
        )
    if wect_type == "faces":
        # noinspection PyUnboundLocalVariable
        return (
            compute_wecc(nh, batch.batch, lin, batch.node_weights)
            - compute_wecc(
                eh, batch.batch[batch.edge_index[0]], lin, edge_weights
            )
            + compute_wecc(fh, batch.batch[batch.face[0]], lin, face_weights)
        )
    raise ValueError(f"Invalid wect_type: {wect_type}")


class WECTLayer(nn.Module):
    """Machine learning layer for computing the WECT (Weighted ECT).

    Parameters
    ----------
    v: torch.FloatTensor
        The direction vector that contains the directions. The shape of the
        tensor v is either [ndims, num_thetas] or [n_channels, ndims,
        num_thetas].
    config : ECTConfig
        The configuration object of the WECT layer.
    """

    def __init__(self, config: ECTConfig, v=None):
        super().__init__()
        self.config = config
        self.lin = nn.Parameter(
            torch.linspace(
                -config.radius, config.radius, config.bump_steps
            ).view(-1, 1, 1, 1),
            requires_grad=False,
        )

        # If provided with one set of directions.
        # For backwards compatibility.
        if v.ndim == 2:
            v.unsqueeze(0)

        # The set of directions is added
        if config.fixed:
            self.v = nn.Parameter(v.movedim(-1, -2), requires_grad=False)
        else:
            self.v = nn.Parameter(torch.zeros_like(v.movedim(-1, -2)))
            geotorch.constraints.sphere(self, "v", radius=config.radius)
            # Since geotorch randomizes the vector during initialization, we
            # assign the values after registering it with spherical constraints.
            # See Geotorch documentation for examples.
            self.v = v.movedim(-1, -2)

    def forward(self, batch: Batch):
        """Forward method for the WECT Layer.


        Parameters
        ----------
        batch : Batch
            A batch of data containing the node coordinates, edges, faces, batch
            index, and node_weights. It should follow the pytorch geometric
            conventions.

        Returns
        ----------
        wect: torch.FloatTensor
            Returns the WECT of each data object in the batch. If the layer is
            initialized with v of the shape [ndims,num_thetas], the returned
            WECT has shape [batch,num_thetas,bump_steps]. In case the layer is
            initialized with v of the form [n_channels, ndims, num_thetas] the
            returned WECT has the shape [batch,n_channels,num_thetas,bump_steps]
        """
        # Movedim for geotorch
        wect = compute_wect(
            batch, self.v.movedim(-1, -2), self.lin, self.config.ect_type
        )
        if self.config.normalized:
            return normalize(wect)
        return wect.squeeze()
