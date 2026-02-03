import time
import torch
import argparse
import os

from torch.utils.data import DataLoader
from data.dataset import get_dataset
from models.UnifyModel import UnifyModel

# ==========================================
# 配置区域 (保持与 train.py 一致)
# ==========================================
default_args = {
    'benchmark': 'xsub',  # 对应 train.py 中的设置
    'modalities': ['rgb', 'pose', 'depth', 'ir'],  # 需要与训练时保持一致
    'num_frames': 32,
    'use_val': False  # 测试脚本通常不需要再次切分 val
}
BATCH_SIZE = 4  # 推理时不需要梯度累积，Batch Size 可以根据显存适当调大
CHECKPOINT_PATH = 'work_dir/ggr_experiment/best_model.pth'  # 🟢 请修改为你实际的模型路径


def get_test_dataloader(args, batch_size):
    print(f"Loading Test Dataset ({args['benchmark']})...")
    # 注意：这里 split 必须填 'test'
    test_set = get_dataset('NTU120', 'test', args)

    return DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )


def accuracy(output, target, topk=(1,)):
    """计算 Top-k 准确率"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def main():
    # 1. 解析参数 (可选，为了方便命令行修改模型路径)
    parser = argparse.ArgumentParser(description='Test Script')
    parser.add_argument('--checkpoint', type=str, default=CHECKPOINT_PATH, help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size for testing')
    cmd_args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Start Testing | Device: {device} | Checkpoint: {cmd_args.checkpoint}")

    # 2. 加载数据集
    test_loader = get_test_dataloader(default_args, cmd_args.batch_size)
    print(f"Data Ready. Test Batches: {len(test_loader)}")

    # 3. 构建模型
    # 注意：num_classes 必须与训练时一致 (NTU120=120, NTU60=60)
    model = UnifyModel(num_classes=120).to(device)

    # 4. 加载权重
    if os.path.isfile(cmd_args.checkpoint):
        print(f"Loading checkpoint from {cmd_args.checkpoint} ...")
        checkpoint = torch.load(cmd_args.checkpoint, map_location=device)

        # 兼容直接保存 state_dict 或保存了完整 info 的情况
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # 处理可能的 DataParallel 'module.' 前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v

        msg = model.load_state_dict(new_state_dict, strict=True)
        print(f"Checkpoint loaded. {msg}")
    else:
        print(f"Error: No checkpoint found at {cmd_args.checkpoint}")
        return

    # 5. 开始测试
    model.eval()

    top1_acc_avg = 0.0
    top5_acc_avg = 0.0
    total_batches = len(test_loader)
    start_time = time.time()

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            # 数据搬运
            x_rgb = batch['rgb'].to(device)
            x_ir = batch['ir'].to(device)
            x_depth = batch['depth'].to(device)
            x_pose = batch['pose'].to(device)
            targets = batch['label'].to(device)

            # 推理 (gradient_control='base' 即可，不需要 GGR 路由)
            logits_shared, _ = model(x_rgb, x_ir, x_depth, x_pose, gradient_control='base')

            # 计算 Batch 精度
            acc1, acc5 = accuracy(logits_shared, targets, topk=(1, 5))
            top1_acc_avg += acc1.item()
            top5_acc_avg += acc5.item()

            if i % 10 == 0:
                print(f"Iter {i}/{total_batches} | Batch Top-1: {acc1.item():.2f}%")

    # 6. 最终结果
    top1_acc_avg /= total_batches
    top5_acc_avg /= total_batches
    total_time = time.time() - start_time

    print("\n" + "=" * 40)
    print(f"✅ Test Finished in {total_time:.1f}s")
    print(f"🏆 Top-1 Accuracy: {top1_acc_avg:.2f}%")
    print(f"🥈 Top-5 Accuracy: {top5_acc_avg:.2f}%")
    print("=" * 40)


if __name__ == '__main__':
    main()