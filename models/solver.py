import torch
import torch.nn as nn
import torch.nn.functional as F
from models.GMDtools import GMD


class Solver:
    def __init__(self, model, optimizer, mode='base', lambda_spec=1.0, lambda_orth=0.1, accum_steps=1):
        """
        mode:
            'base': Joint training (L_shared + L_spec)
            'DGL' : Detached Gradient Learning (Fusion detached)
            'GMD' : Gradient Modulation (PCGrad style between Shared vs Spec)
            'GGR' : Gradient-Guided Routing (The Proposed Method - Orthogonalization)

        lambda_orth: Weight for Feature Orthogonality Loss
        accum: Gradient accumulation steps
        """
        self.model = model
        self.mode = mode
        self.lambda_spec = lambda_spec
        self.lambda_orth = lambda_orth
        self.accum_steps = accum_steps
        self.criterion = nn.CrossEntropyLoss()

        self.step_count = 0

        if self.mode == 'GMD':
            self.optimizer = GMD(optimizer, reduction='mean')
        else:
            self.optimizer = optimizer

        print(f"Solver initialized in mode: [{self.mode.upper()}] | Accumulation: {self.accum_steps}")

    def train_step(self, batch, device):
        """
        Execute one training iteration.
        """
        # 1. Prepare Data
        x_rgb, x_ir, x_depth, x_pose = batch['rgb'].to(device), batch['ir'].to(device), batch['depth'].to(device), batch['pose'].to(device)
        targets = batch['label'].to(device)

        self.step_count += 1
        is_update_step = (self.step_count % self.accum_steps == 0)

        # Mode: GGR
        if self.mode == 'GGR':
            # 1. Forward
            logits_shared, logits_spec = self.model(x_rgb, x_ir, x_depth, x_pose, gradient_control='GGR')
            # 2. Compute Losses
            l_shared = self.criterion(logits_shared, targets)
            l_specs = {k: self.criterion(v, targets) for k, v in logits_spec.items()}

            loss_dict = {'shared': l_shared.item()}
            loss_dict.update({k: v.item() for k, v in l_specs.items()})

            l_shared_scaled = l_shared / self.accum_steps
            l_specs_scaled = {k: v / self.accum_steps for k, v in l_specs.items()}

            total_spec_loss = sum(l_specs_scaled.values()) * self.lambda_spec
            total_spec_loss.backward(retain_graph=True)

            self._apply_asymmetric_ggr(l_shared_scaled)

            l_orth = torch.tensor(0.0, device=device)
            _model = self.model.module if hasattr(self.model, 'module') else self.model

            if hasattr(_model, 'z_shared') and hasattr(_model, 'features'):
                z_s = _model.z_shared
                for z_p in _model.features.values():
                    l_orth += F.cosine_similarity(z_s, z_p, dim=1).abs().mean()

            loss_dict['orth'] = l_orth.item()
            (l_orth * self.lambda_orth / self.accum_steps).backward()

            if is_update_step:
                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss_val = l_shared + sum(l_specs.values()) * self.lambda_spec
            return total_loss_val.item(), loss_dict

        else:
            ctrl = 'DGL' if self.mode == 'DGL' else ('GMD' if self.mode == 'GMD' else 'base')
            logits_shared, logits_spec = self.model(x_rgb, x_ir, x_depth, x_pose, gradient_control=ctrl)

            l_shared = self.criterion(logits_shared, targets)
            l_specs_dict = {k: self.criterion(v, targets) for k, v in logits_spec.items()}
            l_spec_sum = sum(l_specs_dict.values())

            loss_dict = {'shared': l_shared.item()}
            loss_dict.update({k: v.item() for k, v in l_specs_dict.items()})

            if self.mode == 'GMD':
                objectives = [l_shared / self.accum_steps, l_spec_sum * self.lambda_spec / self.accum_steps]
                self.optimizer.pc_backward(objectives)
                total_loss_val = l_shared + l_spec_sum * self.lambda_spec
            else:
                total_loss = l_shared + self.lambda_spec * l_spec_sum
                (total_loss / self.accum_steps).backward()
                total_loss_val = total_loss

            if is_update_step:
                self.optimizer.step()
                self.optimizer.zero_grad()

            return total_loss_val.item(), loss_dict

    def _apply_asymmetric_ggr(self, l_shared):
        """
            Asymmetric Gradient Guided Routing:
            1. 获取 Shared Loss 对 z 的原始梯度。
            2. 从 z.grad 中读取已存在的 Specific 梯度 (Anchor)。
            3. 根据非对称权重 (Asymmetric Weights) 进行投影修剪。
            4. 手动回传修剪后的 Shared 梯度。
        """
        features = self.model.features
        anchor_weights = {
            'pose': 1.0,
            'rgb': 0.2,
            'ir': 0.2,
            'depth': 0.2
        }

        for modality, z in features.items():
            # Step 1: 计算 Shared 任务对 z 的梯度
            g_shared = torch.autograd.grad(l_shared, z, retain_graph=True, allow_unused=True)[0]
            if g_shared is None: continue

            # Step 2: 获取 Anchor 梯度 (来自 Stage A)
            # z.grad 目前包含的是 \nabla_{spec}
            if z.grad is None:
                g_spec = torch.zeros_like(g_shared)
            else:
                g_spec = z.grad.clone()  # Clone 出来作为几何参考，不修改原梯度

            # Step 3: 非对称正交投影
            weight = anchor_weights.get(modality, 0.2)

            if weight > 0:
                g_s_flat = g_shared.view(g_shared.size(0), -1)
                g_p_flat = g_spec.view(g_spec.size(0), -1)

                dot = (g_s_flat * g_p_flat).sum(dim=1, keepdim=True)
                norm_sq = (g_p_flat * g_p_flat).sum(dim=1, keepdim=True) + 1e-8

                # 投影分量 (冗余部分)
                proj = (dot / norm_sq).view(-1, 1) * g_spec

                # 软修剪: 只切除 weight * proj
                g_shared_modified = g_shared - weight * proj
            else:
                g_shared_modified = g_shared

            # Step 4: 回传修剪后的梯度
            # 注意：这会将 g_shared_modified 累加到 z.grad 上
            # 最终 z.grad = \nabla_{spec} + \nabla_{shared}'
            z.backward(g_shared_modified, retain_graph=True)