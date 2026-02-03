import torch
from data.dataset import get_dataset
from torch.utils.data import DataLoader
import torch.nn as nn

from models.CTR_GCN.CTR_GCN import CTR_GCN_Model
from models.swin_transformer import SwinTransformer3D
from models.omnivore import omnivore_swinB


def load_data(
        BATCH_SIZE=4,
        dataset_name='NTU120',
        split='train',
        num_frames=64,
        image_size=(224, 224),
        sample_mod='linspace'
):
    args = {
        'benchmark': 'xsub',
        'modalities': ['rgb', 'pose', 'depth', 'ir'],
        # 'modalities': ['pose'],
        'use_val': False  # 暂时关闭 val 切分，简化调试
    }
    data_set = get_dataset(
        dataset_name,
        split,
        args,
        num_frames=num_frames,
        image_size=image_size,
        sample_mod=sample_mod
    )
    data_loader = DataLoader(data_set, batch_size=BATCH_SIZE)
    return data_set, data_loader


def linear_projection(input_tensor, output_dim):
    if input_tensor.dim() == 2:
        input_dim = input_tensor.size(1)
    elif input_tensor.dim() == 1:
        input_dim = input_tensor.size(0)
        input_tensor = input_tensor.unsqueeze(0)
    linear = nn.Linear(input_dim, output_dim).cuda()
    return linear(input_tensor)


def test_CTR_GCN_backbone():
    B, C, T, V, M = 4, 3, 64, 25, 2
    dummy_input = torch.randn(B, C, T, V, M).to('cuda')
    CTR_GCN_params = {
        'num_class': 120,
        'num_point': V,  # Vertices
        'num_person': M,  # Max_people
        'graph': 'models.CTR_GCN.NTURGBD.Graph',  # 指向 Graph 类所在模块
        'in_channels': C,  # (x, y, z)
        'adaptive': True,
        'backbone_only': True  # True as backbone, False as single classifier
    }
    model = CTR_GCN_Model(**CTR_GCN_params).to('cuda')
    print('CTR_GCN Model Created')
    model.eval()
    with torch.no_grad():
        output_feature = model(dummy_input)
        print(f"Input Tensor Shape: {dummy_input.shape}")
        print(f"Output Tensor Shape: {output_feature.shape}")


def test_SwinT_backbone():
    # 1. Input: (Batch_size, Channels, Time/Frames, Height, Width)
    # Swin-B: 224x224
    B, C, T, H, W = 4, 3, 64, 224, 224
    dummy_input = torch.randn(B, C, T, H, W)


    # 2. 实例化模型 (这里配置一个 Tiny 或 Small 版本用于测试，Swin-B 改 embed_dim=128, depths=[2, 2, 18, 2])
    # 注意：patch_size=(2,4,4) 意味着时间维度下采样 2 倍，空间下采样 4 倍
    model = SwinTransformer3D(
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        patch_size=(2, 4, 4),
        window_size=(8, 7, 7),
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
    )
    print('Swin Transformer Model Created')
    model = model.to('cuda')
    dummy_input = dummy_input.to('cuda')

    with torch.no_grad():
        output = model(dummy_input)

    # 4. 检查输出
    # Swin Transformer 默认输出是 [B, C, T', H', W'] 或者经过 Global Avg Pool 后的 [B, C]
    # 取决于代码实现是否包含最后的 head。
    # 如果 swin_transformer.py 是纯 backbone，它可能输出 feature map。
    print(f"Input tensor Shape: {dummy_input.shape}")
    print(f"Output Tensor Shape: {output.shape}")



if __name__ == '__main__':
    """
    CTR_GCN: input_dim = [B, C, T, V, M], output_dim = [B, 256]
    SwinTransformer3D: input_dim = [B, C, T, H, W], output_dim = [B, 1024]
    RGB/IR: channel = 3
    Depth: channel = 1
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 4
    C = 3
    T = 64
    V = 25
    M = 2
    H = 112
    W = 112
    num_frames = 16
    image_size = (H, W)
    # linsapce or repeat
    sample_mod = 'linspace'



    CTR_GCN_params = {
        'num_class': 120,
        'num_point': V,  # Vertices
        'num_person': M,  # Max_people
        'graph': 'models.CTR_GCN.NTURGBD.Graph',  # 指向 Graph 类所在模块
        'in_channels': C,  # (x, y, z)
        'adaptive': True,
        'backbone_only': True  # True as backbone, False as single classifier
    }
    SwinT_params = {
        'embed_dim': 128,
        'depths': [2, 2, 18, 2],
        'num_heads': [4, 8, 16, 32],
        'patch_size': (2, 4, 4),
        'window_size': (8, 7, 7),
        'mlp_ratio': 4.,
        'qkv_bias': True,
        'drop_rate': 0.,
        'attn_drop_rate': 0.,
        'drop_path_rate': 0.1
    }

    CTR_GCN_model = CTR_GCN_Model(**CTR_GCN_params).to(device).eval()

    """
    SwinT_model_rgb = SwinTransformer3D(**SwinT_params).to(device)
    SwinT_model_depth = SwinTransformer3D(**SwinT_params).to(device)
    SwinT_model_ir = SwinTransformer3D(**SwinT_params).to(device)
    """

    # SwinTransformer3D(**SwinT_params).to(device)
    [rgb_enc, ir_enc, depth_enc, pose_enc] = [
        omnivore_swinB(pretrained=True, load_heads=False).to(device),
        omnivore_swinB(pretrained=True, load_heads=False).to(device),
        omnivore_swinB(pretrained=True, load_heads=False).to(device),
        CTR_GCN_Model(**CTR_GCN_params).to(device).eval()
    ]

    # load data_set(single) / data_loader(in batch)
    _, data_loader = load_data(
        BATCH_SIZE=BATCH_SIZE,
        dataset_name='NTU120',
        split='train',
        num_frames=num_frames,
        image_size=image_size,
        sample_mod=sample_mod
    )
    """
    pose_params = [BATCH_SIZE, C, T, V, M]
    frames_params = [BATCH_SIZE, C, T, H, W]
    dummy_pose_input = torch.randn(pose_params).to(device)
    dummy_frames_input = torch.randn(frames_params).to(device)
    
    test_CTR_GCN_backbone()
    test_SwinT_backbone()
    """
    for batch in data_loader:
        name = batch['sample_name']
        y = batch['label'].to(device)
        pose = batch['pose'].to(device)
        rgb = batch['rgb'].to(device)
        depth = batch['depth'].to(device).repeat(1, 3, 1, 1, 1)
        ir = batch['ir'].to(device)
        print(y.shape, pose.shape, rgb.shape, depth.shape, ir.shape)

        with torch.no_grad():
            pose_output = pose_enc(pose)
            rgb_output = rgb_enc(rgb)
            depth_output = depth_enc(depth)
            ir_output = ir_enc(ir)
            print(pose_output.shape, rgb_output.shape, depth_output.shape, ir_output.shape)
            pose_projected = linear_projection(pose_output, 512)
            rgb_projected = linear_projection(rgb_output, 512)
            depth_projected = linear_projection(depth_output, 512)
            ir_projected = linear_projection(ir_output, 512)
            print(pose_projected.shape, rgb_projected.shape, depth_projected.shape, ir_projected.shape)
        break