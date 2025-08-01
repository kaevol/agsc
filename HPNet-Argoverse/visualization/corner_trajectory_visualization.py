import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import os
from torch_geometric.data import Batch
from torch_geometric.utils import unbatch

from utils import compute_corner_points

num_historical_steps = 10


def corner_trajectory_visualization(data: Batch, traj_output: torch.Tensor, corner_output: torch.Tensor,
                                    is_test: bool = False, max_modes_to_show: int = 3) -> None:
    """
    Visualize trajectory predictions including corner points.

    Args:
        data: Batch data containing agent information
        traj_output: Center trajectory predictions [batch, history, modes, future, 2]
        corner_output: Corner offset predictions [batch, history, modes, future, 4, 2]
        is_test: Whether this is test mode (no ground truth)
        max_modes_to_show: Maximum number of prediction modes to visualize
    """
    batch_size = len(data.get('scenario_id', [0]))

    agent_batch = data['agent']['batch']
    agent_position = data['agent']['position'].detach()
    agent_heading = data['agent']['heading'].detach()
    agent_length = data['agent']['length'].detach()
    agent_width = data['agent']['width'].detach()
    agent_position_list = unbatch(agent_position, agent_batch)
    agent_heading_list = unbatch(agent_heading, agent_batch)
    agent_length_list = unbatch(agent_length, agent_batch)
    agent_width_list = unbatch(agent_width, agent_batch)

    num_modes = traj_output.size(2)
    traj_output = traj_output.detach()
    corner_output = corner_output.detach()
    traj_output_list = unbatch(traj_output[:, -1], agent_batch)
    corner_output_list = unbatch(corner_output[:, -1], agent_batch)
    agent_index = data['agent']['agent_index']

    for i in range(batch_size):
        fig, ax = plt.subplots(figsize=(12, 10))

        # Get ego agent data
        ego_idx = agent_index[i]
        agent_position_i = agent_position_list[i][ego_idx].squeeze(0)
        agent_heading_i = agent_heading_list[i][ego_idx].squeeze(0)
        agent_length_i = agent_length_list[i][ego_idx].item()
        agent_width_i = agent_width_list[i][ego_idx].item()

        agent_historical_position = agent_position_i[:num_historical_steps].cpu().numpy()
        agent_historical_heading = agent_heading_i[:num_historical_steps].cpu().numpy()
        agent_future_position = agent_position_i[num_historical_steps:].cpu().numpy()
        agent_prediction_position = traj_output_list[i][ego_idx].squeeze(0).cpu().numpy()
        agent_corner_offsets = corner_output_list[i][ego_idx].squeeze(0).cpu().numpy()

        # Determine plot bounds
        if not is_test:
            x_min = min(np.min(agent_historical_position[:, 0]), np.min(agent_future_position[:, 0]),
                        np.min(agent_prediction_position[:, :, 0]))
            x_max = max(np.max(agent_historical_position[:, 0]), np.max(agent_future_position[:, 0]),
                        np.max(agent_prediction_position[:, :, 0]))
            y_min = min(np.min(agent_historical_position[:, 1]), np.min(agent_future_position[:, 1]),
                        np.min(agent_prediction_position[:, :, 1]))
            y_max = max(np.max(agent_historical_position[:, 1]), np.max(agent_future_position[:, 1]),
                        np.max(agent_prediction_position[:, :, 1]))
        else:
            x_min = min(np.min(agent_historical_position[:, 0]), np.min(agent_prediction_position[:, :, 0]))
            x_max = max(np.max(agent_historical_position[:, 0]), np.max(agent_prediction_position[:, :, 0]))
            y_min = min(np.min(agent_historical_position[:, 1]), np.min(agent_prediction_position[:, :, 1]))
            y_max = max(np.max(agent_historical_position[:, 1]), np.max(agent_prediction_position[:, :, 1]))

        # Add margin
        margin = 10
        ax.set_xlim(x_min - margin, x_max + margin)
        ax.set_ylim(y_min - margin, y_max + margin)

        # Plot historical trajectory with vehicle boxes
        ax.plot(agent_historical_position[:, 0], agent_historical_position[:, 1],
                'g-', linewidth=2, label='Historical Trajectory', zorder=3)

        # Draw vehicle box at current position
        current_corners = compute_corner_points(
            torch.tensor(agent_historical_position[-1]),
            torch.tensor(agent_historical_heading[-1]),
            torch.tensor(agent_length_i),
            torch.tensor(agent_width_i)
        ).numpy()

        vehicle_box = patches.Polygon(current_corners, closed=True,
                                      facecolor='green', edgecolor='darkgreen',
                                      alpha=0.7, linewidth=2, zorder=4)
        ax.add_patch(vehicle_box)

        # Plot ground truth future if not test
        if not is_test:
            ax.plot(agent_future_position[:, 0], agent_future_position[:, 1],
                    'r-', linewidth=2, label='Ground Truth', zorder=3)

            # Draw vehicle box at final ground truth position
            # Note: We don't have future heading in the data, so we estimate it
            if len(agent_future_position) > 1:
                final_heading = np.arctan2(
                    agent_future_position[-1, 1] - agent_future_position[-2, 1],
                    agent_future_position[-1, 0] - agent_future_position[-2, 0]
                )
            else:
                final_heading = agent_historical_heading[-1]

            gt_corners = compute_corner_points(
                torch.tensor(agent_future_position[-1]),
                torch.tensor(final_heading),
                torch.tensor(agent_length_i),
                torch.tensor(agent_width_i)
            ).numpy()

            gt_box = patches.Polygon(gt_corners, closed=True,
                                     facecolor='red', edgecolor='darkred',
                                     alpha=0.5, linewidth=2, zorder=4)
            ax.add_patch(gt_box)

        # Plot predictions (limit number of modes shown)
        modes_to_show = min(num_modes, max_modes_to_show)
        colors = plt.cm.Blues(np.linspace(0.5, 0.9, modes_to_show))

        for j in range(modes_to_show):
            # Plot center trajectory
            ax.plot(agent_prediction_position[j, :, 0], agent_prediction_position[j, :, 1],
                    '-', color=colors[j], linewidth=2, alpha=0.8,
                    label=f'Prediction Mode {j + 1}', zorder=2)

            # Draw vehicle box with predicted corners at final position
            final_center = agent_prediction_position[j, -1]
            final_corner_offsets = agent_corner_offsets[j, -1]  # [4, 2]
            final_corners = final_center + final_corner_offsets

            pred_box = patches.Polygon(final_corners, closed=True,
                                       facecolor=colors[j], edgecolor=colors[j],
                                       alpha=0.5, linewidth=2, zorder=2)
            ax.add_patch(pred_box)

            # Optionally draw corner trajectories
            for corner_idx in range(4):
                corner_traj = agent_prediction_position[j] + agent_corner_offsets[j, :, corner_idx]
                ax.plot(corner_traj[:, 0], corner_traj[:, 1],
                        ':', color=colors[j], linewidth=1, alpha=0.5)

        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title(f'Trajectory Prediction with Corner Points - Scenario {data.get("scenario_id", [i])[i]}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        # Save figure
        os.makedirs('visualization/corner_trajectory', exist_ok=True)
        if is_test:
            os.makedirs('test_output/visualization', exist_ok=True)
            plt.savefig(f'test_output/visualization/corners_{data.get("scenario_id", [i])[i]}.png',
                        dpi=150, bbox_inches='tight')
        else:
            plt.savefig(f'visualization/corner_trajectory/corners_{data.get("scenario_id", [i])[i]}.png',
                        dpi=150, bbox_inches='tight')
        plt.close()


def visualize_corner_details(data: Batch, traj_output: torch.Tensor, corner_output: torch.Tensor,
                             timestep: int = -1, mode: int = 0) -> None:
    """
    Detailed visualization of corner predictions at a specific timestep.
    """
    agent_position = data['agent']['position'][data['agent']['agent_index'][0]]
    agent_heading = data['agent']['heading'][data['agent']['agent_index'][0]]
    agent_length = data['agent']['length'][data['agent']['agent_index'][0]]
    agent_width = data['agent']['width'][data['agent']['agent_index'][0]]

    # Get predictions
    pred_center = traj_output[data['agent']['agent_index'][0], -1, mode, timestep]
    pred_corner_offsets = corner_output[data['agent']['agent_index'][0], -1, mode, timestep]

    # Compute predicted corners
    pred_corners = pred_center + pred_corner_offsets

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot center point
    ax.plot(pred_center[0], pred_center[1], 'ko', markersize=10, label='Predicted Center')

    # Plot corners
    corner_labels = ['Front-Left', 'Front-Right', 'Rear-Right', 'Rear-Left']
    corner_colors = ['red', 'blue', 'green', 'orange']

    for i, (corner, label, color) in enumerate(zip(pred_corners, corner_labels, corner_colors)):
        ax.plot(corner[0], corner[1], 'o', color=color, markersize=8, label=label)
        # Draw offset vector
        ax.arrow(pred_center[0], pred_center[1],
                 pred_corner_offsets[i, 0], pred_corner_offsets[i, 1],
                 head_width=0.2, head_length=0.1, fc=color, ec=color, alpha=0.5)

    # Draw vehicle box
    vehicle_box = patches.Polygon(pred_corners, closed=True,
                                  facecolor='lightgray', edgecolor='black',
                                  alpha=0.3, linewidth=2)
    ax.add_patch(vehicle_box)

    # Add dimensions
    ax.text(pred_center[0], pred_center[1] + agent_width.item() / 2 + 1,
            f'Width: {agent_width.item():.2f}m', ha='center')
    ax.text(pred_center[0] + agent_length.item() / 2 + 1, pred_center[1],
            f'Length: {agent_length.item():.2f}m', ha='left', rotation=90)

    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_title(f'Corner Point Prediction Details - Timestep {timestep}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.show()