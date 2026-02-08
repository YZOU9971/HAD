import torch
import torch.nn as nn

from models.CTR_GCN.CTR_GCN import CTR_GCN_Model
from models.omnivore import omnivore_swinB


class UnifyModel(nn.Module):
    """
    Compatible with arbitrary modality subsets.

    - modalities: list of strings in {'rgb','ir','depth','pose'}
    - forward expects inputs for all modalities, but will only use those in `modalities`.
      You can pass None for unused modalities.
    """
    ALL_MODALITIES = ['rgb', 'ir', 'depth', 'pose']

    def __init__(self,
                 num_classes=120,
                 embed_dim=512,
                 fusion_depth=2,
                 fusion_heads=8,
                 drop_rate=0.1,
                 modalities=('rgb', 'ir', 'depth', 'pose'),):
        super().__init__()

        self.modalities = tuple(modalities)
        assert len(self.modalities) >= 1, "Modalities must be non-empty."
        for m in self.modalities:
            assert m in self.ALL_MODALITIES, f"Unknown modality: {m}."

        print(f"Building Backbones... modalities={self.modalities}")

        # Encoders
        CTR_GCN_params = {
            'num_class': num_classes,
            'num_point': 25,
            'num_person': 2,
            'graph': 'models.CTR_GCN.NTURGBD.Graph',
            'in_channels': 3,
            'adaptive': True,
            'backbone_only': True
        }

        self.swin_out_dim = 1024
        self.gcn_out_dim = 256

        # (RGB/IR/Depth are frozen)
        self.enc_rgb = omnivore_swinB(pretrained=True, load_heads=False)
        self.enc_ir = omnivore_swinB(pretrained=True, load_heads=False)
        self.enc_depth = omnivore_swinB(pretrained=True, load_heads=False)

        # Pose encoder is trainable (strong anchor)
        self.enc_pose = CTR_GCN_Model(**CTR_GCN_params)

        # Projectors (unified via ModuleDict)
        self.projector = nn.ModuleDict({
            'rgb': nn.Sequential(nn.Linear(self.swin_out_dim, embed_dim), nn.LayerNorm(embed_dim), nn.GELU()),
            'ir': nn.Sequential(nn.Linear(self.swin_out_dim, embed_dim), nn.LayerNorm(embed_dim), nn.GELU()),
            'depth': nn.Sequential(nn.Linear(self.swin_out_dim, embed_dim), nn.LayerNorm(embed_dim), nn.GELU()),
            'pose': nn.Sequential(nn.Linear(self.gcn_out_dim, embed_dim), nn.BatchNorm1d(embed_dim), nn.GELU()),
        })

        # Specific heads
        self.head_spec = nn.ModuleDict({
            'rgb': nn.Linear(embed_dim, num_classes),
            'ir': nn.Linear(embed_dim, num_classes),
            'depth': nn.Linear(embed_dim, num_classes),
            'pose': nn.Linear(embed_dim, num_classes),
        })

        # Fusion (shared)
        self.fusion_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Modality embeddings: one per modality token + (optional) CLS embedding
        # Using a dict-like embedding table to avoid length mismatch.
        self.modality_id = {m: i for i, m in enumerate(self.ALL_MODALITIES)}
        self.modal_embed_table = nn.Embedding(len(self.ALL_MODALITIES) + 1, embed_dim)  # +1 for CLS id
        self.CLS_ID = len(self.ALL_MODALITIES)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=fusion_heads,
            dim_feedforward=embed_dim * 4,
            dropout=drop_rate,
            activation='gelu',
            batch_first=True
        )
        self.fusion_enc = nn.TransformerEncoder(enc_layer, num_layers=fusion_depth)
        self.head_shared = nn.Linear(embed_dim, num_classes)

        # Runtime caches (used by Solver)
        self.features = {}  # modality -> z_m (projected)
        self.z_shared = None  # z_s (CLS output)

        # Optional diagnostics (large tensors)
        self._fusion_in = None  # (B, 1+M, D)
        self._token_ids = None  # (B, 1+M)

        self._init_weights()
        self._freeze_backbones()

    def _init_weights(self):
        print("Initializing weights...")
        nn.init.trunc_normal_(self.fusion_token, std=0.02)
        nn.init.trunc_normal_(self.modal_embed_table.weight, std=0.02)

        # init projectors + heads + fusion + shared head
        modules_to_init = list(self.projector.values()) + list(self.head_spec.values()) + [self.fusion_enc, self.head_shared]

        for module in modules_to_init:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)

    def _freeze_backbones(self):
        print('Freezing Omnivore backbones (RGB/IR/Depth)...')
        for module in [self.enc_rgb, self.enc_ir, self.enc_depth]:
            for p in module.parameters():
                p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        # keep frozen backbones in eval mode (BN/dropout stability)
        self.enc_rgb.eval()
        self.enc_ir.eval()
        self.enc_depth.eval()

    # Encoding helpers
    def _encode_one(self, modality, x):
        """Encode one modality -> feature vector."""
        if modality == 'rgb':
            with torch.no_grad():
                return self.enc_rgb(x)
        if modality == 'ir':
            with torch.no_grad():
                return self.enc_ir(x)
        if modality == 'depth':
            with torch.no_grad():
                return self.enc_depth(x)
        if modality == 'pose':
            return self.enc_pose(x)
        raise ValueError(f"Unknown modality: {modality}")

    def get_fusion_inputs(self):
        """
        Returns the last forward fusion input tokens:
        - fusion_in: (B, 1+M, D)
        - token_ids: (B, 1+M)
        Useful for token-level diagnostics.
        """
        return self._fusion_in, self._token_ids

    def forward(self, x_rgb=None, x_ir=None, x_depth=None, x_pose=None, gradient_control='base'):
        """
        gradient_control:
          - 'base': normal joint training
          - 'DGL' : detach modality tokens before fusion (stop shared->modality)
          - 'GMD' : forward same as base; solver uses PCGrad
          - 'GGR' : forward same as base; solver uses routing/orth/warmup, etc.
        """
        # gather inputs
        x_map = {'rgb': x_rgb, 'ir': x_ir, 'depth': x_depth, 'pose': x_pose}
        # sanity: required modalities must be provided
        for m in self.modalities:
            if x_map[m] is None:
                raise ValueError(f"Input for modality '{m}' is None but required by model.modalities={self.modalities}")

        # Batch size from first used modality
        B = x_map[self.modalities[0]].shape[0]
        device = x_map[self.modalities[0]].device

        # 1) Encode + Project only used modalities
        z = {}
        for m in self.modalities:
            f_m = self._encode_one(m, x_map[m])
            z[m] = self.projector[m](f_m)

        # store for Solver
        self.features = z

        # retain grad only for GGR (Solver reads z.grad as anchor)
        if gradient_control == 'GGR':
            for v in z.values():
                v.retain_grad()

        # 2) Specific heads (only for used modalities)
        logits_spec = {m: self.head_spec[m](z[m]) for m in self.modalities}

        # 3) Fusion inputs (order = self.modalities)
        if gradient_control == 'DGL':
            z_in = [z[m].detach() for m in self.modalities]
        else:
            z_in = [z[m] for m in self.modalities]

        feats_stack = torch.stack(z_in, dim=1)                                      # (B, M, D)
        cls_tokens = self.fusion_token.expand(B, -1, -1)                            # (B,1,D)
        fusion_in = torch.cat((cls_tokens, feats_stack), dim=1)             # (B,1+M,D)

        # 4) Add token-wise modality embedding (CLS + each modality token)
        ids = [self.CLS_ID] + [self.modality_id[m] for m in self.modalities]        # length=1+M
        token_ids = torch.tensor(ids, device=device).view(1, -1).expand(B, -1)      # (B,1+M)
        fusion_in = fusion_in + self.modal_embed_table(token_ids)                   # (B,1+M,D)

        if gradient_control == 'GGR':
            fusion_in.retain_grad()

        self._fusion_in = fusion_in
        self._token_ids = token_ids

        # 5) Shared fusion + head
        fusion_out = self.fusion_enc(fusion_in)
        self.z_shared = fusion_out[:, 0, :]
        logits_shared = self.head_shared(self.z_shared)

        return logits_shared, logits_spec