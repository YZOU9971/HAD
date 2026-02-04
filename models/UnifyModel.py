import torch
from data.dataset import get_dataset
from torch.utils.data import DataLoader
import torch.nn as nn

from models.CTR_GCN.CTR_GCN import CTR_GCN_Model
from models.omnivore import omnivore_swinB


class UnifyModel(nn.Module):
    def __init__(self,
                 num_classes=60,
                 embed_dim=512,
                 fusion_depth=2,
                 fusion_heads=8,
                 drop_rate=0.1):
        super().__init__()

        # ==============================
        # 1. Unimodal Backbone Encoders
        # ==============================
        print("Building Backbones...")
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

        # Omnivore backbones (will be frozen)
        self.enc_rgb = omnivore_swinB(pretrained=True, load_heads=False)
        self.enc_ir = omnivore_swinB(pretrained=True, load_heads=False)
        self.enc_depth = omnivore_swinB(pretrained=True, load_heads=False)

        # Pose backbone (kept trainable as Strong Anchor)
        self.enc_pose = CTR_GCN_Model(**CTR_GCN_params)

        # ==============================
        # 2. Projection Layers
        # ==============================
        self.proj_rgb = nn.Sequential(nn.Linear(self.swin_out_dim, embed_dim), nn.LayerNorm(embed_dim), nn.GELU())
        self.proj_ir = nn.Sequential(nn.Linear(self.swin_out_dim, embed_dim), nn.LayerNorm(embed_dim), nn.GELU())
        self.proj_depth = nn.Sequential(nn.Linear(self.swin_out_dim, embed_dim), nn.LayerNorm(embed_dim), nn.GELU())
        self.proj_pose = nn.Sequential(
            nn.Linear(self.gcn_out_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.GELU()
        )

        # ==============================
        # 3. Specific Heads (Auxiliary Supervision)
        # ==============================
        # These provide the source of \nabla_{spec}
        self.head_spec_rgb = nn.Linear(embed_dim, num_classes)
        self.head_spec_ir = nn.Linear(embed_dim, num_classes)
        self.head_spec_depth = nn.Linear(embed_dim, num_classes)
        self.head_spec_pose = nn.Linear(embed_dim, num_classes)

        # ==============================
        # 4. Fusion Module (Shared)
        # ==============================
        self.fusion_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.modal_embed = nn.Parameter(torch.randn(1, 5, embed_dim) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=fusion_heads, dim_feedforward=embed_dim * 4,
            dropout=drop_rate, activation='gelu', batch_first=True
        )
        self.fusion_enc = nn.TransformerEncoder(enc_layer, num_layers=fusion_depth)
        self.head_shared = nn.Linear(embed_dim, num_classes)

        self.features = {}
        # Used to store z_shared for L_orth calculation
        self.z_shared = None

        self._init_weights()
        self._freeze_backbones()

    def _init_weights(self):
        print("Initializing weights...")
        for m in [self.fusion_token, self.modal_embed]:
            nn.init.trunc_normal_(m, std=0.02)

        modules_to_init = [
            self.proj_rgb, self.proj_ir, self.proj_depth, self.proj_pose,
            self.head_spec_rgb, self.head_spec_ir, self.head_spec_depth, self.head_spec_pose,
            self.fusion_enc, self.head_shared
        ]
        for module in modules_to_init:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None: nn.init.constant_(m.bias, 0)
                elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)

    def _freeze_backbones(self):
        print('Freezing Omnivore backbones (RGB/IR/Depth)...')
        # Only freeze visual backbones, Pose is kept trainable
        for module in [self.enc_rgb, self.enc_ir, self.enc_depth]:
            for param in module.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        # Keep frozen backbones in eval mode (especially for BN)
        self.enc_rgb.eval()
        self.enc_ir.eval()
        self.enc_depth.eval()
        # Pose backbone follows the global train/eval mode

    def forward(self, x_rgb, x_ir, x_depth, x_pose, gradient_control='base'):
        B = x_rgb.shape[0]

        # Phase 1: Encoding
        with torch.no_grad():
            f_rgb = self.enc_rgb(x_rgb)
            f_ir = self.enc_ir(x_ir)
            f_depth = self.enc_depth(x_depth)
        f_pose = self.enc_pose(x_pose)

        # Phase 2: Projection
        z_rgb = self.proj_rgb(f_rgb)
        z_ir = self.proj_ir(f_ir)
        z_depth = self.proj_depth(f_depth)
        z_pose = self.proj_pose(f_pose)

        self.features = {'rgb': z_rgb, 'ir': z_ir, 'depth': z_depth, 'pose': z_pose}

        if gradient_control in ['GMD', 'GGR']:
            for v in self.features.values():
                v.retain_grad()

        # Phase 3: Specific Heads (The Source of Anchor Gradients)
        logits_spec = {
            'rgb': self.head_spec_rgb(z_rgb),
            'ir': self.head_spec_ir(z_ir),
            'depth': self.head_spec_depth(z_depth),
            'pose': self.head_spec_pose(z_pose)
        }

        # Phase 4: Fusion Input
        if gradient_control == 'DGL':
            z_in = [v.detach() for v in [z_rgb, z_ir, z_depth, z_pose]]
        else:
            z_in = [z_rgb, z_ir, z_depth, z_pose]

        # Phase 5: Shared Fusion
        feats_stack = torch.stack(z_in, dim=1)
        cls_tokens = self.fusion_token.expand(B, -1, -1)
        fusion_in = torch.cat((cls_tokens, feats_stack), dim=1)
        fusion_in = fusion_in + self.modal_embed

        fusion_out = self.fusion_enc(fusion_in)
        z_shared = fusion_out[:, 0, :]

        # 🟢 CRITICAL: Store z_shared for L_orth in Solver
        self.z_shared = z_shared

        logits_shared = self.head_shared(z_shared)

        return logits_shared, logits_spec