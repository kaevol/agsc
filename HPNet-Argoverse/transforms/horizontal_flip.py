import math
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform

from utils import wrap_angle


class HorizontalFlip(BaseTransform):
    def __init__(self,
                 flip_p=0.5):
        super(HorizontalFlip, self).__init__()
        self.flip_p = flip_p

    def flip_position_and_heading(self, position, heading):
        position[..., 0] = -position[..., 0]
        angle = wrap_angle(math.pi - heading)
        return position, angle

    def __call__(self, data: HeteroData) -> HeteroData:
        if torch.rand(1).item() < self.flip_p:
            # Flip agent data
            data['agent']['position'], data['agent']['heading'] = self.flip_position_and_heading(
                data['agent']['position'], data['agent']['heading'])
            # Flip velocity x-component
            data['agent']['velocity'][..., 0] = -data['agent']['velocity'][..., 0]

            # Flip corner data
            if 'corner' in data:
                # Flip corner positions
                data['corner']['position'][..., 0] = -data['corner']['position'][..., 0]
                # Flip corner offsets x-component
                data['corner']['offsets'][..., 0] = -data['corner']['offsets'][..., 0]

                # When flipping horizontally, we need to swap left and right corners
                # Original order: [front_left, front_right, rear_right, rear_left]
                # After flip: [front_right, front_left, rear_left, rear_right]
                num_agents = data['agent']['num_nodes']
                num_steps = data['corner']['position'].size(1)

                # Reshape to separate corners
                corner_pos = data['corner']['position'].reshape(num_agents, 4, num_steps, 2)
                corner_off = data['corner']['offsets'].reshape(num_agents, 4, num_steps, 2)

                # Swap corners
                corner_pos_new = corner_pos.clone()
                corner_off_new = corner_off.clone()
                corner_pos_new[:, 0] = corner_pos[:, 1]  # front_left <- front_right
                corner_pos_new[:, 1] = corner_pos[:, 0]  # front_right <- front_left
                corner_pos_new[:, 2] = corner_pos[:, 3]  # rear_right <- rear_left
                corner_pos_new[:, 3] = corner_pos[:, 2]  # rear_left <- rear_right

                corner_off_new[:, 0] = corner_off[:, 1]
                corner_off_new[:, 1] = corner_off[:, 0]
                corner_off_new[:, 2] = corner_off[:, 3]
                corner_off_new[:, 3] = corner_off[:, 2]

                # Reshape back
                data['corner']['position'] = corner_pos_new.reshape(-1, num_steps, 2)
                data['corner']['offsets'] = corner_off_new.reshape(-1, num_steps, 2)

            # Flip lane data
            data['lane']['position'], data['lane']['heading'] = self.flip_position_and_heading(data['lane']['position'],
                                                                                               data['lane']['heading'])

            # Flip polyline data
            data['polyline']['position'], data['polyline']['heading'] = self.flip_position_and_heading(
                data['polyline']['position'], data['polyline']['heading'])
            # Swap left and right polyline sides
            data['polyline']['side'] = 2 - data['polyline']['side']

            # Swap left and right neighbor edges
            data['lane', 'lane']['left_neighbor_edge_index'], data['lane', 'lane']['right_neighbor_edge_index'] = \
                data['lane', 'lane']['right_neighbor_edge_index'], data['lane', 'lane']['left_neighbor_edge_index']
        return data