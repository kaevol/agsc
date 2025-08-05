import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
import math

from losses import Huber2DLoss
from losses import CELoss
from metrics import BrierMinFDE
from metrics import MR
from metrics import MinADE
from metrics import MinFDE
from modules import Backbone
from modules import MapEncoder

from utils import generate_target
from utils import generate_predict_mask
from utils import compute_corner_points
from utils import compute_corner_offsets


class HPNet(pl.LightningModule):

    def __init__(self,
                 hidden_dim: int,
                 num_historical_steps: int,
                 num_future_steps: int,
                 pos_duration: int,
                 pred_duration: int,
                 a2a_radius: float,
                 l2a_radius: float,
                 num_visible_steps: int,
                 num_modes: int,
                 num_attn_layers: int,
                 num_hops: int,
                 num_heads: int,
                 dropout: float,
                 lr: float,
                 weight_decay: float,
                 warmup_epochs: int,
                 T_max: int,
                 corner_loss_weight_max: float = 0.1,
                 corner_loss_type: str = 'FDE',  # 'FDE' or 'ADE'
                 corner_loss_warmup_epochs: int = 10,
                 **kwargs) -> None:
        super(HPNet, self).__init__()
        self.save_hyperparameters()
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.pos_duration = pos_duration
        self.pred_duration = pred_duration
        self.a2a_radius = a2a_radius
        self.l2a_radius = l2a_radius
        self.num_visible_steps = num_visible_steps
        self.num_modes = num_modes
        self.num_attn_layers = num_attn_layers
        self.num_hops = num_hops
        self.num_heads = num_heads
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.T_max = T_max
        self.corner_loss_weight_max = corner_loss_weight_max
        self.corner_loss_type = corner_loss_type
        self.corner_loss_warmup_epochs = corner_loss_warmup_epochs

        self.Backbone = Backbone(
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            num_future_steps=num_future_steps,
            pos_duration=pos_duration,
            pred_duration=pred_duration,
            a2a_radius=a2a_radius,
            l2a_radius=l2a_radius,
            num_attn_layers=num_attn_layers,
            num_modes=num_modes,
            num_heads=num_heads,
            dropout=dropout
        )
        self.MapEncoder = MapEncoder(
            hidden_dim=hidden_dim,
            num_hops=num_hops,
            num_heads=num_heads,
            dropout=dropout
        )

        self.reg_loss = Huber2DLoss()
        self.prob_loss = CELoss()

        self.brier_minFDE = BrierMinFDE()
        self.minADE = MinADE()
        self.minFDE = MinFDE()
        self.MR = MR()

        self.test_traj_output = dict()
        self.test_corner_output = dict()
        self.test_prob_output = dict()

    def forward(self,
                data: Batch):
        lane_embs = self.MapEncoder(data=data)
        pred = self.Backbone(data=data, l_embs=lane_embs)
        return pred

    def get_corner_loss_weight(self):
        """Calculate current corner loss weight based on training progress"""
        if self.current_epoch < self.corner_loss_warmup_epochs:
            # Linear warmup
            return self.corner_loss_weight_max * (self.current_epoch / self.corner_loss_warmup_epochs)
        else:
            return self.corner_loss_weight_max

    def compute_corner_loss(self, pred_corners, target_corners, target_mask):
        """
        Compute loss for corner point predictions.
        
        Args:
            pred_corners: [Na, F, 4, 2] - predicted corner offsets
            target_corners: [Na, F, 4, 2] - target corner offsets
            target_mask: [Na, F] - mask indicating valid timesteps
        """
        if self.corner_loss_type == 'FDE':
            # Only compute loss for the last frame
            last_frame_mask = target_mask[:, -1]  # [Na]
            num_valid = last_frame_mask.sum()
            
            if num_valid > 0:
                # Select only valid agents
                pred_last = pred_corners[last_frame_mask][:, -1]  # [Na_valid, 4, 2]
                target_last = target_corners[last_frame_mask][:, -1]  # [Na_valid, 4, 2]
                # Reshape for loss computation
                pred_flat = pred_last.reshape(-1, 2)  # [Na_valid * 4, 2]
                target_flat = target_last.reshape(-1, 2)  # [Na_valid * 4, 2]
                corner_loss = self.reg_loss(pred_flat, target_flat)
            else:
                # Return a loss that maintains gradient flow
                corner_loss = pred_corners.sum() * 0.0
        else:  # ADE
            # Compute average loss over all frames
            num_valid = target_mask.sum()
            
            if num_valid > 0:
                # Create mask for all corners and frames
                # We need to select valid (agent, frame) pairs
                valid_indices = target_mask.nonzero(as_tuple=True)
                
                # Extract valid predictions and targets
                pred_valid = pred_corners[valid_indices[0], valid_indices[1]]  # [N_valid, 4, 2]
                target_valid = target_corners[valid_indices[0], valid_indices[1]]  # [N_valid, 4, 2]
                
                # Reshape for loss computation
                pred_flat = pred_valid.reshape(-1, 2)  # [N_valid * 4, 2]
                target_flat = target_valid.reshape(-1, 2)  # [N_valid * 4, 2]
                corner_loss = self.reg_loss(pred_flat, target_flat)
            else:
                # Return a loss that maintains gradient flow
                corner_loss = pred_corners.sum() * 0.0
        
        return corner_loss

    def training_step(self, data, batch_idx):
        traj_propose, traj_output, corner_propose, corner_output, prob_output = self(
            data)  # [(N1,...,Nb),H,K,F,2],[(N1,...,Nb),H,K,F,2],[(N1,...,Nb),H,K,F,4,2],[(N1,...,Nb),H,K,F,4,2],[(N1,...,Nb),H,K]
        
        target_traj, target_mask = generate_target(position=data['agent']['position'],
                                                   mask=data['agent']['visible_mask'],
                                                   num_historical_steps=self.num_historical_steps,
                                                   num_future_steps=self.num_future_steps)  # [(N1,...Nb),H,F,2],[(N1,...Nb),H,F]

        # Generate target corner offsets
        num_agents = data['agent']['position'].size(0)
        target_corners = []
        for h in range(self.num_historical_steps):
            corners_h = []
            for f in range(self.num_future_steps):
                step_idx = h + f + 1
                corners = compute_corner_points(
                    data['agent']['position'][:, step_idx],
                    data['agent']['heading'][:, step_idx],
                    data['agent']['length'],
                    data['agent']['width']
                )
                corner_offsets = compute_corner_offsets(
                    data['agent']['position'][:, step_idx],
                    corners
                )
                corners_h.append(corner_offsets)
            corners_h = torch.stack(corners_h, dim=1)  # [N, F, 4, 2]
            target_corners.append(corners_h)
        target_corners = torch.stack(target_corners, dim=1)  # [N, H, F, 4, 2]

        errors = (torch.norm(traj_propose[..., :2] - target_traj.unsqueeze(2), p=2, dim=-1) * target_mask.unsqueeze(
            2)).sum(dim=-1)  # [(N1,...Nb),H,K]
        best_mode_index = errors.argmin(dim=-1)  # [(N1,...Nb),H]
        traj_best_propose = traj_propose[
            torch.arange(traj_propose.size(0))[:, None], torch.arange(traj_propose.size(1))[None,
                                                         :], best_mode_index]  # [(N1,...Nb),H,F,2]
        traj_best_output = traj_output[
            torch.arange(traj_output.size(0))[:, None], torch.arange(traj_output.size(1))[None,
                                                        :], best_mode_index]  # [(N1,...Nb),H,F,2]
        corner_best_propose = corner_propose[
            torch.arange(corner_propose.size(0))[:, None], torch.arange(corner_propose.size(1))[None,
                                                           :], best_mode_index]  # [(N1,...Nb),H,F,4,2]
        corner_best_output = corner_output[
            torch.arange(corner_output.size(0))[:, None], torch.arange(corner_output.size(1))[None,
                                                          :], best_mode_index]  # [(N1,...Nb),H,F,4,2]

        predict_mask = generate_predict_mask(data['agent']['visible_mask'][:, :self.num_historical_steps],
                                             self.num_visible_steps)  # [(N1,...Nb),H]
        targ_mask = target_mask[predict_mask]  # [Na,F]
        traj_pro = traj_best_propose[predict_mask]  # [Na,F,2]
        traj_ref = traj_best_output[predict_mask]  # [Na,F,2]
        corner_pro = corner_best_propose[predict_mask]  # [Na,F,4,2]
        corner_ref = corner_best_output[predict_mask]  # [Na,F,4,2]
        prob = prob_output[predict_mask]  # [Na,K]
        targ = target_traj[predict_mask]  # [Na,F,2]
        targ_corners = target_corners[predict_mask]  # [Na,F,4,2]
        label = best_mode_index[predict_mask]  # [Na]

        # Center point losses
        reg_loss_propose = self.reg_loss(traj_pro[targ_mask], targ[targ_mask])
        reg_loss_refine = self.reg_loss(traj_ref[targ_mask], targ[targ_mask])
        
        # Corner point losses
        corner_loss_weight = self.get_corner_loss_weight()
        corner_loss_propose = self.compute_corner_loss(corner_pro, targ_corners, targ_mask)
        corner_loss_refine = self.compute_corner_loss(corner_ref, targ_corners, targ_mask)
        
        # Probability loss
        prob_loss = self.prob_loss(prob, label)
        
        # Total loss
        loss = reg_loss_propose + reg_loss_refine + prob_loss + \
               corner_loss_weight * (corner_loss_propose + corner_loss_refine)
        
        # Logging
        self.log('train_reg_loss_propose', reg_loss_propose, prog_bar=True, on_step=True, on_epoch=True, batch_size=1,
                 sync_dist=True)
        self.log('train_reg_loss_refine', reg_loss_refine, prog_bar=True, on_step=True, on_epoch=True, batch_size=1,
                 sync_dist=True)
        self.log('train_corner_loss_propose', corner_loss_propose, prog_bar=True, on_step=True, on_epoch=True, batch_size=1,
                 sync_dist=True)
        self.log('train_corner_loss_refine', corner_loss_refine, prog_bar=True, on_step=True, on_epoch=True, batch_size=1,
                 sync_dist=True)
        self.log('train_corner_loss_weight', corner_loss_weight, prog_bar=True, on_step=True, on_epoch=True, batch_size=1,
                 sync_dist=True)
        self.log('train_prob_loss', prob_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)

        return loss

    def validation_step(self, data, batch_idx):
        traj_propose, traj_output, corner_propose, corner_output, prob_output = self(
            data)  # [(N1,...,Nb),H,K,F,2],[(N1,...,Nb),H,K,F,2],[(N1,...,Nb),H,K,F,4,2],[(N1,...,Nb),H,K,F,4,2],[(N1,...,Nb),H,K]
        
        target_traj, target_mask = generate_target(position=data['agent']['position'],
                                                   mask=data['agent']['visible_mask'],
                                                   num_historical_steps=self.num_historical_steps,
                                                   num_future_steps=self.num_future_steps)  # [(N1,...Nb),H,F,2],[(N1,...Nb),H,F]

        # Generate target corner offsets
        num_agents = data['agent']['position'].size(0)
        target_corners = []
        for h in range(self.num_historical_steps):
            corners_h = []
            for f in range(self.num_future_steps):
                step_idx = h + f + 1
                corners = compute_corner_points(
                    data['agent']['position'][:, step_idx],
                    data['agent']['heading'][:, step_idx],
                    data['agent']['length'],
                    data['agent']['width']
                )
                corner_offsets = compute_corner_offsets(
                    data['agent']['position'][:, step_idx],
                    corners
                )
                corners_h.append(corner_offsets)
            corners_h = torch.stack(corners_h, dim=1)  # [N, F, 4, 2]
            target_corners.append(corners_h)
        target_corners = torch.stack(target_corners, dim=1)  # [N, H, F, 4, 2]

        errors = (torch.norm(traj_propose[..., :2] - target_traj.unsqueeze(2), p=2, dim=-1) * target_mask.unsqueeze(
            2)).sum(dim=-1)  # [(N1,...Nb),H,K]
        best_mode_index = errors.argmin(dim=-1)  # [(N1,...Nb),H]
        traj_best_propose = traj_propose[
            torch.arange(traj_propose.size(0))[:, None], torch.arange(traj_propose.size(1))[None,
                                                         :], best_mode_index]  # [(N1,...Nb),H,F,2]
        traj_best_output = traj_output[
            torch.arange(traj_output.size(0))[:, None], torch.arange(traj_output.size(1))[None,
                                                        :], best_mode_index]  # [(N1,...Nb),H,F,2]
        corner_best_propose = corner_propose[
            torch.arange(corner_propose.size(0))[:, None], torch.arange(corner_propose.size(1))[None,
                                                           :], best_mode_index]  # [(N1,...Nb),H,F,4,2]
        corner_best_output = corner_output[
            torch.arange(corner_output.size(0))[:, None], torch.arange(corner_output.size(1))[None,
                                                          :], best_mode_index]  # [(N1,...Nb),H,F,4,2]

        predict_mask = generate_predict_mask(data['agent']['visible_mask'][:, :self.num_historical_steps],
                                             self.num_visible_steps)  # [(N1,...Nb),H]
        targ_mask = target_mask[predict_mask]  # [Na,F]
        traj_pro = traj_best_propose[predict_mask]  # [Na,F,2]
        traj_ref = traj_best_output[predict_mask]  # [Na,F,2]
        corner_pro = corner_best_propose[predict_mask]  # [Na,F,4,2]
        corner_ref = corner_best_output[predict_mask]  # [Na,F,4,2]
        prob = prob_output[predict_mask]  # [Na,K]
        targ = target_traj[predict_mask]  # [Na,F,2]
        targ_corners = target_corners[predict_mask]  # [Na,F,4,2]
        label = best_mode_index[predict_mask]  # [Na]

        # Center point losses
        reg_loss_propose = self.reg_loss(traj_pro[targ_mask], targ[targ_mask])
        reg_loss_refine = self.reg_loss(traj_ref[targ_mask], targ[targ_mask])
        
        # Corner point losses (use max weight for validation)
        corner_loss_propose = self.compute_corner_loss(corner_pro, targ_corners, targ_mask)
        corner_loss_refine = self.compute_corner_loss(corner_ref, targ_corners, targ_mask)
        
        # Probability loss
        prob_loss = self.prob_loss(prob, label)
        
        # Total loss (use max corner weight for validation)
        loss = reg_loss_propose + reg_loss_refine + prob_loss + \
               self.corner_loss_weight_max * (corner_loss_propose + corner_loss_refine)
        
        # Logging
        self.log('val_reg_loss_propose', reg_loss_propose, prog_bar=True, on_step=False, on_epoch=True, batch_size=1,
                 sync_dist=True)
        self.log('val_reg_loss_refine', reg_loss_refine, prog_bar=True, on_step=False, on_epoch=True, batch_size=1,
                 sync_dist=True)
        self.log('val_corner_loss_propose', corner_loss_propose, prog_bar=True, on_step=False, on_epoch=True, batch_size=1,
                 sync_dist=True)
        self.log('val_corner_loss_refine', corner_loss_refine, prog_bar=True, on_step=False, on_epoch=True, batch_size=1,
                 sync_dist=True)
        self.log('val_prob_loss', prob_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)

        # Compute metrics (unchanged - only for center points)
        agent_index = data['agent']['agent_index'] + data['agent']['ptr'][:-1]
        num_agents = agent_index.size(0)
        agent_traj = traj_output[agent_index, -1]  # [N,K,F,2]
        agent_prob = prob_output[agent_index, -1]  # [N,K]
        agent_targ = target_traj[agent_index, -1]  # [N,F,2]
        fde = torch.norm(agent_traj[:, :, -1, :2] - agent_targ[:, -1, :2].unsqueeze(1), p=2, dim=-1)  # [N,K]
        best_mode_index = fde.argmin(dim=-1)  # [N]
        agent_traj_best = agent_traj[torch.arange(num_agents), best_mode_index]  # [N,F,2]
        self.brier_minFDE.update(agent_traj_best[..., :2], agent_targ[..., :2],
                                 agent_prob[torch.arange(num_agents), best_mode_index])
        self.minADE.update(agent_traj_best[..., :2], agent_targ[..., :2])
        self.minFDE.update(agent_traj_best[..., :2], agent_targ[..., :2])
        self.MR.update(agent_traj_best[..., :2], agent_targ[..., :2])
        self.log('val_minADE', self.minADE, prog_bar=True, on_step=False, on_epoch=True, batch_size=num_agents,
                 sync_dist=True)
        self.log('val_minFDE', self.minFDE, prog_bar=True, on_step=False, on_epoch=True, batch_size=num_agents,
                 sync_dist=True)
        self.log('val_MR', self.MR, prog_bar=True, on_step=False, on_epoch=True, batch_size=num_agents, sync_dist=True)
        self.log('val_brier_minFDE', self.brier_minFDE, prog_bar=True, on_step=False, on_epoch=True,
                 batch_size=num_agents, sync_dist=True)

    def test_step(self, data, batch_idx):
        traj_propose, traj_output, corner_propose, corner_output, prob_output = self(
            data)  # [(N1,...,Nb),H,K,F,2],[(N1,...,Nb),H,K,F,2],[(N1,...,Nb),H,K,F,4,2],[(N1,...,Nb),H,K,F,4,2],[(N1,...,Nb),H,K]

        prob_output = prob_output ** 2
        prob_output = prob_output / prob_output.sum(dim=-1, keepdim=True)

        agent_index = data['agent']['agent_index'] + data['agent']['ptr'][:-1]
        num_agents = agent_index.size(0)
        agent_traj = traj_output[agent_index, -1]  # [N,K,F,2]
        agent_corners = corner_output[agent_index, -1]  # [N,K,F,4,2]
        agent_prob = prob_output[agent_index, -1]  # [N,K]

        for i in range(num_agents):
            id = int(data['scenario_id'][i])
            traj = agent_traj[i].cpu().numpy()
            corners = agent_corners[i].cpu().numpy()
            prob = agent_prob[i].tolist()

            self.test_traj_output[id] = traj
            self.test_corner_output[id] = corners
            self.test_prob_output[id] = prob

    def on_test_end(self):
        # Save test results in a format suitable for evaluation
        import json
        output_path = './test_output'
        import os
        os.makedirs(output_path, exist_ok=True)

        with open(os.path.join(output_path, 'predictions.json'), 'w') as f:
            json.dump({
                'trajectories': self.test_traj_output,
                'corner_offsets': self.test_corner_output,
                'probabilities': self.test_prob_output
            }, f)

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM,
                                    nn.LSTMCell, nn.GRU, nn.GRUCell)
        blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name
                if 'bias' in param_name:
                    no_decay.add(full_param_name)
                elif 'weight' in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ('weight' in param_name or 'bias' in param_name):
                    no_decay.add(full_param_name)
        param_dict = {param_name: param for param_name, param in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {"params": [param_dict[param_name] for param_name in sorted(list(decay))],
             "weight_decay": self.weight_decay},
            {"params": [param_dict[param_name] for param_name in sorted(list(no_decay))],
             "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=self.lr, weight_decay=self.weight_decay)

        warmup_epochs = self.warmup_epochs
        T_max = self.T_max

        def warmup_cosine_annealing_schedule(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            return 0.5 * (1.0 + math.cos(math.pi * (epoch - warmup_epochs + 1) / (T_max - warmup_epochs + 1)))

        scheduler = {
            'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine_annealing_schedule),
            'interval': 'epoch',
            'frequency': 1
        }
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('HPNet')
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--num_historical_steps', type=int, default=10)  # Changed from 20 to 10
        parser.add_argument('--num_future_steps', type=int, default=30)
        parser.add_argument('--pos_duration', type=int, default=10)
        parser.add_argument('--pred_duration', type=int, default=10)
        parser.add_argument('--a2a_radius', type=float, default=50)
        parser.add_argument('--l2a_radius', type=float, default=50)
        parser.add_argument('--num_visible_steps', type=int, default=2)
        parser.add_argument('--num_modes', type=int, default=6)
        parser.add_argument('--num_attn_layers', type=int, default=2)
        parser.add_argument('--num_hops', type=int, default=4)
        parser.add_argument('--num_heads', type=int, default=8)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--lr', type=float, default=3e-4)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--warmup_epochs', type=int, default=4)
        parser.add_argument('--T_max', type=int, default=64)
        parser.add_argument('--corner_loss_weight_max', type=float, default=0.1)
        parser.add_argument('--corner_loss_type', type=str, default='FDE', choices=['FDE', 'ADE'])
        parser.add_argument('--corner_loss_warmup_epochs', type=int, default=10)
        return parent_parser