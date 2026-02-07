import torch
import torch.nn as nn
import torch.nn.functional as F
from models.GMDtools import GMD


class Solver:
    def __init__(
            self,
            model,
            optimizer,
            mode='base',
            lambda_spec=1.0,
            lambda_orth=0.1,
            accum_steps=1,
            warmup_epochs=1,
            steps_per_epoch=None,
    ):
        self.model = model
        self.mode = str(mode).upper()  # normalize
        self.lambda_spec = float(lambda_spec)
        self.lambda_orth = float(lambda_orth)
        self.accum_steps = int(accum_steps)
        self.criterion = nn.CrossEntropyLoss()
        self.step_count = 0
        self.warmup_epochs = int(warmup_epochs)
        self.steps_per_epoch = steps_per_epoch
        self.warmup_steps = self._compute_warmup_steps()

        self.anchor_decay = 0.9
        self.anchor_ema = {}
        self.grad_norm_ema = {}
        self.anchor_weights = {
            'pose': 1.0,
            'rgb': 0.2,
            'ir': 0.2,
            'depth': 0.2
        }

        if self.mode == 'GMD':
            self.optimizer = GMD(optimizer, reduction='mean')
        else:
            self.optimizer = optimizer

        print(f"Solver initialized in mode: [{self.mode}] | Accumulation: {self.accum_steps}")

    def _unwrap_model(self):
        return self.model.module if hasattr(self.model, 'module') else self.model

    def _compute_warmup_steps(self):
        if self.steps_per_epoch is None or self.warmup_steps <= 0:
            return 0
        return int(self.steps_per_epoch * self.warmup_epochs)

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
            self._update_anchor_ema()
            anchor_weights = self._get_anchor_weights()
            self._apply_asymmetric_ggr(l_shared_scaled, anchor_weights)

            # (3) Correlation penalty between z_shared and modality features (named orth in your code)
            _model = self._unwrap_model()
            l_orth = torch.tensor(0.0, device=device)

            if hasattr(_model, 'z_shared') and hasattr(_model, 'features'):
                z_s = _model.z_shared
                z_s = F.normalize(z_s, dim=1)
                for z_p in _model.features.values():
                    l_orth = l_orth + (z_s * z_p).sum(dim=1).pow(2).mean()

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

    def _update_anchor_ema(self):
        _model = self._unwrap_model()
        features = getattr(_model, 'features', None)
        if not isinstance(features, dict) or len(features) == 0:
            return

        for modality, z in features.items():
            if z.grad is None:
                continue

            grad = z.grad.detach()
            grad_flat = grad.flatten(1)
            grad_dir_flat = F.normalize(grad_flat, dim=1)
            grad_dir = grad_dir_flat.view_as(grad)
            if modality not in self.anchor_ema:
                self.anchor_ema[modality] = grad_dir.clone()
            else:
                self.anchor_ema[modality] = self.anchor_decay * self.anchor_ema[modality] + (
                            1 - self.anchor_decay) * grad_dir

            grad_norm = grad_flat.norm(p=2, dim=1).mean().item()
            if modality not in self.grad_norm_ema:
                self.grad_norm_ema[modality] = grad_norm
            else:
                self.grad_norm_ema[modality] = self.anchor_decay * self.grad_norm_ema[modality] + (
                            1 - self.anchor_decay) * grad_norm

    def _get_anchor_weights(self):
        if self.step_count < self.warmup_steps or len(self.grad_norm_ema) == 0:
            return self.anchor_weights

        mean_norm = sum(self.grad_norm_ema.values()) / max(1, len(self.grad_norm_ema))
        if mean_norm <= 0:
            return self.anchor_weights

        weights = {}
        for modality, base_w in self.anchor_weights.items():
            rel = self.grad_norm_ema.get(modality, mean_norm) / mean_norm
            weights[modality] = float(base_w * rel)
        return weights

    def _apply_asymmetric_ggr(self, l_shared_scaled, anchor_weights):
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

        z_list = []
        grad_list = []

        for modality, z in features.items():
            g_shared = torch.autograd.grad(l_shared_scaled, z, retain_graph=True, allow_unused=True)[0]
            if g_shared is None:
                continue

            if modality in self.anchor_ema:
                g_spec = self.anchor_ema[modality]
            elif z.grad is not None:
                g_spec = F.normalize(z.grad.detach().flatten(1), dim=1).view_as(g_shared)
            else:
                g_spec = torch.zeros_like(g_shared)

            w = float(anchor_weights.get(modality, 0.2))
            if w > 0.0:
                # projection of g_shared onto g_spec
                g_s_flat = g_shared.view(g_shared.size(0), -1)
                g_p_flat = g_spec.view(g_spec.size(0), -1)

                g_p_flat = F.normalize(g_p_flat, dim=1)
                dot = (g_s_flat * g_p_flat).sum(dim=1, keepdim=True)
                proj = dot * g_p_flat  # (B, Dflat)

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
