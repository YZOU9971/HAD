import torch
import torch.nn as nn
import torch.nn.functional as F
from models.GMDtools import GMD


class Solver:
    def __init__(self, model, optimizer, mode='base', lambda_spec=1.0, lambda_orth=0.1, accum_steps=1):
        self.model = model
        self.mode = str(mode).upper()  # normalize
        self.lambda_spec = float(lambda_spec)
        self.lambda_orth = float(lambda_orth)
        self.accum_steps = int(accum_steps)
        self.criterion = nn.CrossEntropyLoss()
        self.step_count = 0

        if self.mode == 'GMD':
            self.optimizer = GMD(optimizer, reduction='mean')
        else:
            self.optimizer = optimizer

        print(f"Solver initialized in mode: [{self.mode}] | Accumulation: {self.accum_steps}")

    def _unwrap_model(self):
        return self.model.module if hasattr(self.model, 'module') else self.model

    def _build_inputs_from_batch(self, batch, device):
        """
        Compatible with UnifyModel forward kwargs: x_rgb/x_ir/x_depth/x_pose.
        Uses model.modalities to decide which keys are needed.
        """
        _m = self._unwrap_model()
        modalities = getattr(_m, 'modalities', ('rgb', 'ir', 'depth', 'pose'))

        key_map = {'rgb': 'x_rgb', 'ir': 'x_ir', 'depth': 'x_depth', 'pose': 'x_pose'}
        inputs = {}

        for mod in modalities:
            if mod not in batch:
                raise KeyError(f"Batch missing modality '{mod}'. Available keys: {list(batch.keys())}")
            inputs[key_map[mod]] = batch[mod].to(device, non_blocking=True)

        return inputs, modalities

    def train_step(self, batch, device):
        inputs, modalities = self._build_inputs_from_batch(batch, device)
        targets = batch['label'].to(device, non_blocking=True)

        self.step_count += 1
        is_update_step = (self.step_count % self.accum_steps == 0)

        # Mode: GGR
        if self.mode == 'GGR':
            logits_shared, logits_spec = self.model(**inputs, gradient_control='GGR')

            l_shared = self.criterion(logits_shared, targets)
            l_specs = {k: self.criterion(v, targets) for k, v in logits_spec.items()}

            loss_dict = {'shared': l_shared.item()}
            loss_dict.update({k: v.item() for k, v in l_specs.items()})

            # scale for gradient accumulation
            l_shared_scaled = l_shared / self.accum_steps
            l_specs_scaled = {k: v / self.accum_steps for k, v in l_specs.items()}

            # (1) Backprop SPEC first (anchor gradients stored in z.grad)
            total_spec_loss = sum(l_specs_scaled.values()) * self.lambda_spec
            total_spec_loss.backward(retain_graph=True)

            # (2) Apply asymmetric routing to shared gradients (feature-level)
            self._apply_asymmetric_ggr(l_shared_scaled)

            # (3) Correlation penalty between z_shared and modality features (named orth in your code)
            _model = self._unwrap_model()
            l_orth = torch.tensor(0.0, device=device)

            if hasattr(_model, 'z_shared') and hasattr(_model, 'features'):
                z_s = _model.z_shared
                for z_p in _model.features.values():
                    l_orth = l_orth + F.cosine_similarity(z_s, z_p, dim=1).abs().mean()

            loss_dict['orth'] = l_orth.item()
            (l_orth * self.lambda_orth / self.accum_steps).backward()

            if is_update_step:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            total_loss_val = l_shared + sum(l_specs.values()) * self.lambda_spec
            return total_loss_val.item(), loss_dict

        # base / DGL / GMD
        ctrl = 'base'
        if self.mode == 'DGL':
            ctrl = 'DGL'
        elif self.mode == 'GMD':
            ctrl = 'GMD'

        logits_shared, logits_spec = self.model(**inputs, gradient_control=ctrl.lower() if ctrl == 'base' else ctrl)

        l_shared = self.criterion(logits_shared, targets)
        l_specs_dict = {k: self.criterion(v, targets) for k, v in logits_spec.items()}
        l_spec_sum = sum(l_specs_dict.values())

        loss_dict = {'shared': l_shared.item()}
        loss_dict.update({k: v.item() for k, v in l_specs_dict.items()})

        if self.mode == 'GMD':
            objectives = [
                l_shared / self.accum_steps,
                (l_spec_sum * self.lambda_spec) / self.accum_steps
            ]
            self.optimizer.pc_backward(objectives)
            total_loss_val = l_shared + l_spec_sum * self.lambda_spec
        else:
            total_loss = l_shared + self.lambda_spec * l_spec_sum
            (total_loss / self.accum_steps).backward()
            total_loss_val = total_loss

        if is_update_step:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        return total_loss_val.item(), loss_dict

    def _apply_asymmetric_ggr(self, l_shared_scaled):
        """
        1) Read existing z.grad as anchor gradients (from spec backward).
        2) Compute g_shared = d(l_shared)/d(z)
        3) For each modality: g_shared' = g_shared - w_m * proj_{g_spec}(g_shared)
        4) Backprop modified g_shared' into graph once (fast)
        """
        _model = self._unwrap_model()

        features = getattr(_model, 'features', None)
        if not isinstance(features, dict) or len(features) == 0:
            return

        anchor_weights = {
            'pose': 1.0,
            'rgb': 0.2,
            'ir': 0.2,
            'depth': 0.2
        }

        z_list = []
        grad_list = []

        for modality, z in features.items():
            g_shared = torch.autograd.grad(l_shared_scaled, z, retain_graph=True, allow_unused=True)[0]
            if g_shared is None:
                continue

            g_spec = z.grad.detach().clone() if z.grad is not None else torch.zeros_like(g_shared)

            w = float(anchor_weights.get(modality, 0.2))
            if w > 0.0:
                # projection of g_shared onto g_spec
                g_s_flat = g_shared.view(g_shared.size(0), -1)
                g_p_flat = g_spec.view(g_spec.size(0), -1)

                dot = (g_s_flat * g_p_flat).sum(dim=1, keepdim=True)
                norm_sq = (g_p_flat * g_p_flat).sum(dim=1, keepdim=True) + 1e-8
                proj = (dot / norm_sq) * g_p_flat  # (B, Dflat)

                proj = proj.view_as(g_shared)
                g_shared_mod = g_shared - w * proj
            else:
                g_shared_mod = g_shared

            z_list.append(z)
            grad_list.append(g_shared_mod)

        if len(z_list) > 0:
            # one backward call (equivalent to per-z backward accumulation)
            torch.autograd.backward(z_list, grad_list, retain_graph=True)

    def flush_step(self):
        """
        Call this at the end of an epoch if you want to avoid dropping the last partial accumulation.
        """
        if self.step_count % self.accum_steps != 0:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)