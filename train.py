import time
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from data.dataset import get_dataset
from models.UnifyModel import UnifyModel
from models.solver import Solver

args = {
    'benchmark': 'xsub',
    'modalities': ['rgb', 'pose', 'depth', 'ir'],
    # 'modalities': ['pose'],
    'num_frames': 32,
    'use_val': True
}
BATCH_SIZE = 4
ACCUM_STEPS = 8
# 等效batch_size = 4 * 8 = 32
EPOCH = 3
BASE_LR = 1e-3
WEIGHT_DECAY = 1e-4
MODE = 'base'
LAMBDA_SPEC = 1.0
LAMBDA_ORTH = 0.1
WORK_DIR = 'checkpoints'


def get_dataloader(args, BATCH_SIZE):
    train_set = get_dataset('NTU120', 'train', args)
    # train_set = get_dataset('PKUMMD', 'train', args)
    # train_set = get_dataset('NUCLA', 'train', args)
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        drop_last=True
    )
    val_loader = None
    if args['use_val']:
        # 通常 NTU 的验证集 split 叫 'test' 或 'val'，根据你的 dataset.py 实现调整
        # 这里假设是 'test' (标准 Cross-Subject/View 测试集)
        val_set = get_dataset('NTU120', 'test', args)
        val_loader = DataLoader(
            val_set,
            batch_size=BATCH_SIZE,  # 验证集不需要梯度累积，BS=4 即可
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    return train_loader, val_loader


def accuracy(output, target, topk=(1,)):
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Start Training | Device: {device} | Mode: {MODE}")
    print(f"Physical Batch: {BATCH_SIZE} | Accumulation: {ACCUM_STEPS} | Effective Batch: {BATCH_SIZE * ACCUM_STEPS}")

    model = UnifyModel(num_classes=120).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    model_params_count = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_count = sum(p.numel() for p in trainable_params) / 1e6
    print(f"Model Total Params: {model_params_count:.2f} M")
    print(f"Trainable Params:   {trainable_count:.2f} M (Visual Backbones Frozen)")

    optimizer = torch.optim.Adam(trainable_params, lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCH, eta_min=1e-6)

    solver = Solver(
        model,
        optimizer,
        mode=MODE,
        lambda_spec=LAMBDA_SPEC,
        lambda_orth=LAMBDA_ORTH,
        accum_steps=ACCUM_STEPS
    )

    train_loader, val_loader = get_dataloader(args, BATCH_SIZE)
    print(f"Data Ready. Train Batches: {len(train_loader)}")
    if val_loader:
        print(f"            Val Batches:   {len(val_loader)}")

    global_step = 0
    best_acc = 0.0

    for epoch in range(1, EPOCH + 1):
        model.train()
        start = time.time()
        total_loss = 0

        print(f"\n=== Epoch {epoch}/{EPOCH} ===")

        for i, batch in enumerate(train_loader):
            loss_val, loss_dict = solver.train_step(batch, device)

            total_loss += loss_val
            global_step += 1

            if i % 10 == 0:
                log_str = f"Iter {global_step} | Total: {loss_val:.4f}"

                if 'shared' in loss_dict:
                    log_str += f" | Shared: {loss_dict['shared']:.4f}"

                if 'orth' in loss_dict and MODE == 'GGR':
                    log_str += f" | Orth: {loss_dict['orth']:.4f}"

                for k in ['rgb', 'pose', 'depth', 'ir']:
                    if k in loss_dict:
                        log_str += f" | {k}: {loss_dict[k]:.4f}"

                print(log_str)

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        epoch_time = (time.time() - start) / 60
        print(f"Epoch {epoch} Finished. Avg Loss: {avg_loss:.4f}. Time: {epoch_time:.1f} min")

        if args['use_val'] and val_loader is not None:
            print("Running Validation...")
            model.eval()

            top1_acc_avg = 0.0
            top5_acc_avg = 0.0
            total_batches = len(val_loader)

            with torch.no_grad():
                for batch in val_loader:
                    x_rgb = batch['rgb'].to(device)
                    x_ir = batch['ir'].to(device)
                    x_depth = batch['depth'].to(device)
                    x_pose = batch['pose'].to(device)
                    targets = batch['label'].to(device)

                    # 验证时只看 Shared Head (gradient_control='base' 即可)
                    logits_shared, _ = model(x_rgb, x_ir, x_depth, x_pose, gradient_control='base')

                    acc1, acc5 = accuracy(logits_shared, targets, topk=(1, 5))
                    top1_acc_avg += acc1.item()
                    top5_acc_avg += acc5.item()

            top1_acc_avg /= total_batches
            top5_acc_avg /= total_batches

            print(f"Val Result: Top-1: {top1_acc_avg:.2f}% | Top-5: {top5_acc_avg:.2f}%")

            # --- Save Best Model ---
            if top1_acc_avg > best_acc:
                best_acc = top1_acc_avg
                save_path = os.path.join(WORK_DIR, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc,
                }, save_path)
                print(f"🔥 New Best Accuracy! Model saved to {save_path}")

        # 定期保存 checkpoint (防止意外中断)
        if epoch % 2 == 0:
            torch.save(model.state_dict(), os.path.join(WORK_DIR, f'epoch_{epoch}.pth'))

if __name__ == '__main__':
    main()

