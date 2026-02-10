import time
import os
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from data.dataset import get_dataset
from models.UnifyModel import UnifyModel
from models.solver import Solver


# -------------------------
# Config
# -------------------------
args = {
    'benchmark': 'xsub',
    'modalities': ['rgb', 'pose', 'depth', 'ir'],
    'num_frames': 32,
    'use_val': True
}

BATCH_SIZE = 4
ACCUM_STEPS = 8
EPOCH = 3
BASE_LR = 1e-3
WEIGHT_DECAY = 1e-4

MODE = 'GGR'  # 'base' | 'DGL' | 'GMD' | 'GGR'
LAMBDA_SPEC = 1.0
LAMBDA_ORTH = 0.1
WARMUP_EPOCHS = 0

WORK_DIR = 'checkpoints'
SEED = 42
LOG_INTERVAL = 10


# -------------------------
# Utils
# -------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-gpu safe
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------------
# Data
# -------------------------
def get_dataloader(args, batch_size):
    train_set = get_dataset('NTU120', 'train', args)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        drop_last=True
    )

    val_loader = None
    if args.get('use_val', False):
        val_set = get_dataset('NTU120', 'test', args)
        val_loader = DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    return train_loader, val_loader


# -------------------------
# Metrics
# -------------------------
@torch.no_grad()
def accuracy(output, target, topk=(1,)):
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


def build_inputs_from_batch(batch, device, modalities):
    inputs = {}
    key_map = {'rgb': 'x_rgb', 'ir': 'x_ir', 'depth': 'x_depth', 'pose': 'x_pose'}
    for m in modalities:
        if m not in batch:
            raise KeyError(f"Batch missing modality '{m}'. Available keys: {list(batch.keys())}")
        inputs[key_map[m]] = batch[m].to(device, non_blocking=True)
    return inputs


# -------------------------
# Main
# -------------------------
def main():
    seed_everything(SEED)
    os.makedirs(WORK_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    modalities = args['modalities']

    print(f"Start Training | Device: {device} | Mode: {MODE}")
    print(f"Modalities: {modalities}")
    print(f"Physical Batch: {BATCH_SIZE} | Accumulation: {ACCUM_STEPS} | Effective Batch: {BATCH_SIZE * ACCUM_STEPS}")

    # 1) Model
    model = UnifyModel(num_classes=120, modalities=tuple(modalities)).to(device)

    # 2) Optimizer / Scheduler
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    model_params_count = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_count = sum(p.numel() for p in trainable_params) / 1e6
    print(f"Model Total Params: {model_params_count:.2f} M")
    print(f"Trainable Params:   {trainable_count:.2f} M")

    optimizer = torch.optim.Adam(trainable_params, lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCH, eta_min=1e-6)

    # 3) Data
    train_loader, val_loader = get_dataloader(args, BATCH_SIZE)
    print(f"Data Ready. Train Batches: {len(train_loader)}")
    if val_loader:
        print(f"            Val Batches:   {len(val_loader)}")

    # 4) Solver
    solver = Solver(
        model,
        optimizer,
        mode=MODE,
        lambda_spec=LAMBDA_SPEC,
        lambda_orth=LAMBDA_ORTH,
        accum_steps=ACCUM_STEPS,
        warmup_epochs=WARMUP_EPOCHS,
        steps_per_epoch=len(train_loader)
    )

    best_acc = 0.0

    for epoch in range(1, EPOCH + 1):
        model.train()
        start = time.time()
        total_loss = 0.0

        print(f"\n=== Epoch {epoch}/{EPOCH} ===")

        for i, batch in enumerate(train_loader):
            loss_val, loss_dict = solver.train_step(batch, device)
            total_loss += float(loss_val)

            # derived counters (no extra state)
            micro_step = solver.micro_step  # must exist in solver
            update_step = micro_step // ACCUM_STEPS

            if (i % LOG_INTERVAL) == 0:
                log_str = f"Micro {micro_step} | Update {update_step} | Total: {loss_val:.4f}"
                if 'shared' in loss_dict:
                    log_str += f" | Shared: {loss_dict['shared']:.4f}"
                if MODE == 'GGR' and 'orth' in loss_dict:
                    log_str += f" | Orth: {loss_dict['orth']:.4f}"
                for m in modalities:
                    if m in loss_dict:
                        log_str += f" | {m}: {loss_dict[m]:.4f}"
                print(log_str)

        # flush last partial accumulation (if any)
        solver.flush_step()

        scheduler.step()
        avg_loss = total_loss / max(1, len(train_loader))
        epoch_time = (time.time() - start) / 60.0
        print(f"Epoch {epoch} Finished. Avg Loss: {avg_loss:.4f}. Time: {epoch_time:.1f} min")

        # -------------------------
        # Validation
        # -------------------------
        if args.get('use_val', False) and val_loader is not None:
            print("Running Validation...")
            model.eval()

            top1_acc_avg = 0.0
            top5_acc_avg = 0.0
            total_batches = len(val_loader)

            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validating", leave=False):
                    targets = batch['label'].to(device, non_blocking=True)
                    inputs = build_inputs_from_batch(batch, device, modalities)

                    logits_shared, _ = model(**inputs, gradient_control='base')
                    acc1, acc5 = accuracy(logits_shared, targets, topk=(1, 5))
                    top1_acc_avg += acc1.item()
                    top5_acc_avg += acc5.item()

            top1_acc_avg /= max(1, total_batches)
            top5_acc_avg /= max(1, total_batches)
            print(f"Val Result: Top-1: {top1_acc_avg:.2f}% | Top-5: {top5_acc_avg:.2f}%")

            if top1_acc_avg > best_acc:
                best_acc = top1_acc_avg
                save_path = os.path.join(WORK_DIR, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc,
                    'mode': MODE,
                    'modalities': modalities,
                    'args': args,
                }, save_path)
                print(f"🔥 New Best Accuracy! Model saved to {save_path}")

        if epoch % 2 == 0:
            ckpt_path = os.path.join(WORK_DIR, f'epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'mode': MODE,
                'modalities': modalities,
                'args': args,
            }, ckpt_path)

    print(f"Training done. Best Top-1: {best_acc:.2f}% | Mode={MODE} | Modalities={modalities}")


if __name__ == '__main__':
    main()
