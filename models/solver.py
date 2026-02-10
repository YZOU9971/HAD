import torch
import torch.nn as nn
import torch.nn.functional as F
from models.GMDtools import GMD


class Solver:
    """
    Modes:
      - BASE:  joint training, total_loss = L_shared + λ_spec * Σ L_aux^m (+ optional L_orth)
      - DGL :  same as BASE, but model detaches modality tokens before fusion (handled in UnifyModel)
      - GMD :  PCGrad-style gradient modulation across objectives (handled by GMD wrapper)
      - GGR :  our routing:
              (1) backprop spec -> get per-modality anchor grads on z_m
              (2) compute shared grads wrt z_m, route them using anchors, backprop routed grads into modality side
              (3) compute shared grads wrt shared params normally (fusion_enc + head_shared)
              (4) optional orth loss (decoupled)
    """

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
        anchor_decay=0.9,
        base_alpha=None,          # modality prior weights
    ):
        self.micro_step = 0
        self.model = model
        self.mode = str(mode).upper()
        self.lambda_spec = float(lambda_spec)
        self.lambda_orth = float(lambda_orth)
        self.accum_steps = int(accum_steps)

        self.criterion = nn.CrossEntropyLoss()
        self.micro_step = 0  # dataloader iterations (for accumulation)

        # ---- warmup steps in "micro steps" (not update steps) ----
        self.warmup_epochs = int(warmup_epochs)
        self.steps_per_epoch = steps_per_epoch
        self.warmup_steps = 0 if (steps_per_epoch is None or self.warmup_epochs <= 0) else int(steps_per_epoch * self.warmup_epochs)

        # ---- anchor EMA (modality-level direction: (D,)) ----
        self.anchor_decay = float(anchor_decay)
        self.anchor_dir = {}      # modality -> tensor(D,)
        self.grad_norm_ema = {}   # modality -> float

        # ---- reliability priors α_m (you can later make them adaptive) ----
        if base_alpha is None:
            base_alpha = {'pose': 1.0, 'rgb': 0.2, 'ir': 0.2, 'depth': 0.2}
        self.base_alpha = {k: float(v) for k, v in base_alpha.items()}

        # ---- optimizer wrapper for GMD ----
        if self.mode == 'GMD':
            self.optimizer = GMD(optimizer, reduction='mean')
        else:
            self.optimizer = optimizer

        print(f"[Solver] mode={self.mode} | accum_steps={self.accum_steps} | warmup_steps={self.warmup_steps}")

    # -------------------------
    # Helpers
    # -------------------------
    def _unwrap_model(self):
        return self.model.module if hasattr(self.model, 'module') else self.model

    def _build_inputs(self, batch, device):
        _m = self._unwrap_model()
        modalities = getattr(_m, 'modalities', ('rgb', 'ir', 'depth', 'pose'))

        key_map = {'rgb': 'x_rgb', 'ir': 'x_ir', 'depth': 'x_depth', 'pose': 'x_pose'}
        inputs = {}
        for mod in modalities:
            if mod not in batch:
                raise KeyError(f"Batch missing modality '{mod}'. keys={list(batch.keys())}")
            inputs[key_map[mod]] = batch[mod].to(device, non_blocking=True)
        return inputs, modalities

    def _shared_params(self):
        """Shared branch params = fusion encoder + shared head."""
        _m = self._unwrap_model()
        params = []
        if hasattr(_m, 'fusion_enc'):
            params += [p for p in _m.fusion_enc.parameters() if p.requires_grad]
        if hasattr(_m, 'head_shared'):
            params += [p for p in _m.head_shared.parameters() if p.requires_grad]
        return params

    @staticmethod
    def _accum_param_grads(params, grads):
        """Manually accumulate grads into .grad (for gradient accumulation compatibility)."""
        for p, g in zip(params, grads):
            if g is None:
                continue
            if p.grad is None:
                p.grad = g.detach().clone()
            else:
                p.grad.add_(g.detach())

    def _update_anchor_from_zgrads(self):
        """
        Build modality-level anchor direction from z.grad (shape: BxD).
        Anchor is normalized, then batch-averaged -> (D,), then EMA-smoothed.
        Also keep EMA of grad norm (for optional adaptive alpha).
        """
        _m = self._unwrap_model()
        feats = getattr(_m, 'features', None)
        if not isinstance(feats, dict) or len(feats) == 0:
            return

        for mod, z in feats.items():
            if z.grad is None:
                continue
            g = z.grad.detach()                # (B, D)
            g = F.normalize(g, dim=1)          # per-sample dir
            g_dir = F.normalize(g.mean(dim=0), dim=0)  # modality-level dir: (D,)

            if mod not in self.anchor_dir:
                self.anchor_dir[mod] = g_dir.clone()
            else:
                self.anchor_dir[mod].mul_(self.anchor_decay).add_(g_dir, alpha=(1 - self.anchor_decay))

            # norm ema (optional)
            g_norm = z.grad.detach().flatten(1).norm(p=2, dim=1).mean().item()
            if mod not in self.grad_norm_ema:
                self.grad_norm_ema[mod] = g_norm
            else:
                self.grad_norm_ema[mod] = self.anchor_decay * self.grad_norm_ema[mod] + (1 - self.anchor_decay) * g_norm

    def _alpha(self, modalities):
        """
        Reliability coefficient α_m.
        - warmup: use base priors
        - later: you can make it adaptive; here we keep it simple + stable
        """
        # 你如果想做 adaptive：在这里用 grad_norm_ema / mean 做缩放即可
        return {m: float(self.base_alpha.get(m, 0.2)) for m in modalities}

    def _route_grad(self, g_shared, mod, alpha):
        """
        Project g_shared (B,D) onto modality anchor direction (D,), subtract scaled projection.
        """
        if alpha <= 0:
            return g_shared

        a = self.anchor_dir.get(mod, None)
        if a is None:
            return g_shared  # no anchor yet

        a = F.normalize(a, dim=0)                         # (D,)
        aB = a.view(1, -1).expand_as(g_shared)            # (B,D)
        # projection: (g·a) a
        dot = (g_shared * aB).sum(dim=1, keepdim=True)    # (B,1)
        proj = dot * aB                                   # (B,D)
        return g_shared - alpha * proj

    # -------------------------
    # Main step
    # -------------------------
    def train_step(self, batch, device):
        inputs, modalities = self._build_inputs(batch, device)
        targets = batch['label'].to(device, non_blocking=True)

        self.micro_step += 1
        do_update = (self.micro_step % self.accum_steps == 0)

        # --------------
        # BASE / DGL / GMD
        # --------------
        if self.mode in {'BASE', 'DGL', 'GMD'}:
            ctrl = 'base' if self.mode == 'BASE' else self.mode  # 'DGL'/'GMD' go to model
            logits_shared, logits_spec = self.model(**inputs, gradient_control=ctrl)

            l_shared = self.criterion(logits_shared, targets)
            l_specs = {m: self.criterion(logits_spec[m], targets) for m in logits_spec.keys()}
            l_spec_sum = sum(l_specs.values())

            total = l_shared + self.lambda_spec * l_spec_sum
            loss_dict = {'shared': l_shared.item(), **{m: l_specs[m].item() for m in l_specs}}

            if self.mode == 'GMD':
                objectives = [l_shared / self.accum_steps, (self.lambda_spec * l_spec_sum) / self.accum_steps]
                self.optimizer.pc_backward(objectives)
            else:
                (total / self.accum_steps).backward()

            if do_update:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            return total.item(), loss_dict

        # --------------
        # GGR
        # --------------
        if self.mode == 'GGR':
            logits_shared, logits_spec = self.model(**inputs, gradient_control='GGR')

            l_shared = self.criterion(logits_shared, targets)
            l_specs = {m: self.criterion(logits_spec[m], targets) for m in logits_spec.keys()}

            loss_dict = {'shared': l_shared.item(), **{m: l_specs[m].item() for m in l_specs}}

            # scale for accumulation
            l_shared_s = l_shared / self.accum_steps
            l_specs_s = {m: (l_specs[m] / self.accum_steps) for m in l_specs}

            _m = self._unwrap_model()
            feats = getattr(_m, 'features', None)
            if not isinstance(feats, dict) or len(feats) == 0:
                raise RuntimeError("UnifyModel.features is empty; cannot run GGR.")

            # ---- (1) SPEC backward -> anchors on z_m ----
            spec_total = self.lambda_spec * sum(l_specs_s.values())
            spec_total.backward(retain_graph=True)
            self._update_anchor_from_zgrads()

            # ---- (2) Shared grads to shared params (fusion_enc + head_shared) ----
            # This does NOT send any grad into modality tokens; it's param-only.
            shared_params = self._shared_params()
            g_shared_params = torch.autograd.grad(
                l_shared_s, shared_params, retain_graph=True, allow_unused=True
            )
            self._accum_param_grads(shared_params, g_shared_params)

            # ---- (3) Shared grads wrt modality tokens z_m -> route -> backward into modality side ----
            alpha = self._alpha(modalities)
            z_list, gz_list = [], []
            for mod in modalities:
                z = feats.get(mod, None)
                if z is None:
                    continue
                g = torch.autograd.grad(l_shared_s, z, retain_graph=True, allow_unused=True)[0]
                if g is None:
                    continue

                # warmup: no routing, just pass through
                if self.micro_step <= self.warmup_steps:
                    g_r = g
                else:
                    g_r = self._route_grad(g, mod, alpha.get(mod, 0.2))

                z_list.append(z)
                gz_list.append(g_r)

            if len(z_list) > 0:
                torch.autograd.backward(z_list, gz_list, retain_graph=True)

            # ---- (4) Orth loss (decoupled) ----
            l_orth = torch.tensor(0.0, device=device)
            if hasattr(_m, 'z_shared') and hasattr(_m, 'features'):
                z_s = F.normalize(_m.z_shared, dim=1)
                for z in _m.features.values():
                    z_n = F.normalize(z, dim=1)
                    l_orth = l_orth + (z_s * z_n).sum(dim=1).pow(2).mean()

            loss_dict['orth'] = l_orth.item()
            (self.lambda_orth * l_orth / self.accum_steps).backward()

            if do_update:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            total = l_shared + self.lambda_spec * sum(l_specs.values()) + self.lambda_orth * l_orth
            return total.item(), loss_dict

        raise ValueError(f"Unknown mode={self.mode}")

    def flush_step(self):
        """Force optimizer step if the epoch ends in the middle of accumulation."""
        if self.micro_step % self.accum_steps != 0:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
