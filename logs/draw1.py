import matplotlib.pyplot as plt
import mplcursors


def smooth(scalars, weight=0.8):
    if not scalars:
        return []
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


# ==========================================
# 参数设置
# ==========================================
file_path = 'attempt6_DGL_3epoch.txt'  # 🟢 修改为你的 log 文件
smooth_weight = 0.9

# 🟢 [新增]在此处设置左轴固定范围 (最小值, 最大值)
# 如果不想固定，设置为 None
y_limit_main = (0, 6)
# y_limit_main = None  # 解除固定，自动缩放


# ==========================================
# 读取与解析
# ==========================================
loss_data = {}

epoch_offset = 0
last_local_iter = -1
global_iters = []
epoch_boundaries = set()
# ... (中间解析代码保持不变，为了节省篇幅略过，请保持你原有的逻辑) ...
# 为了确保代码完整运行，这里复制回你的解析逻辑：
with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        if not line: continue
        if '=== Epoch' in line:
            if global_iters: epoch_boundaries.add(global_iters[-1])
            continue
        if not (line.startswith('Iter') or line.startswith('Micro')): continue
        parts = line.split('|')
        try:
            iter_str = parts[0].strip()
            current_local_iter = int(iter_str.replace(':', '').split()[1])
        except (IndexError, ValueError):
            continue
        if last_local_iter != -1 and current_local_iter < last_local_iter:
            epoch_offset += last_local_iter
            epoch_boundaries.add(epoch_offset)
        last_local_iter = current_local_iter
        global_iter = epoch_offset + current_local_iter
        global_iters.append(global_iter)
        current_losses = {}
        for part in parts[1:]:
            if ':' not in part: continue
            try:
                k, v = part.split(':', 1)
                if k.strip() == 'Total': continue
                current_losses[k.strip()] = float(v.strip())
            except ValueError:
                pass
        for k, v in current_losses.items():
            if k not in loss_data: loss_data[k] = [None] * (len(global_iters) - 1)
            loss_data[k].append(v)
        for k in list(loss_data.keys()):
            if k not in current_losses: loss_data[k].append(None)
sorted_boundaries = sorted(list(epoch_boundaries))  # set转list再sort

# ==========================================
# 绘图
# ==========================================
fig, ax_left = plt.subplots(figsize=(15, 6))
ax_right = ax_left.twinx()

has_data = False
handles_left, labels_left = [], []
handles_right, labels_right = [], []

# -------------------------------
# 左轴：所有非 orth
# -------------------------------
for name, values in loss_data.items():
    if name.lower() == 'orth':
        continue

    clean_x, clean_y = [], []
    for x, y in zip(global_iters, values):
        if y is not None:
            clean_x.append(x)
            clean_y.append(y)

    if clean_x:
        line, = ax_left.plot(
            clean_x,
            smooth(clean_y, smooth_weight),
            linewidth=2,
            alpha=0.85,
            label=name
        )
        handles_left.append(line)
        labels_left.append(name)
        has_data = True

# -------------------------------
# 右轴：orth（灰色，不占颜色轮次）
# -------------------------------
orth_key = None
for k in loss_data.keys():
    if k.lower() == 'orth':
        orth_key = k
        break

if orth_key is not None:
    values = loss_data[orth_key]
    clean_x, clean_y = [], []
    for x, y in zip(global_iters, values):
        if y is not None:
            clean_x.append(x)
            clean_y.append(y)

    if clean_x:
        line, = ax_right.plot(
            clean_x,
            smooth(clean_y, smooth_weight),
            color='gray',
            linestyle='--',
            linewidth=2,
            alpha=0.7,
            label=orth_key
        )
        handles_right.append(line)
        labels_right.append(orth_key)
        has_data = True

if not has_data:
    print("❌ 错误：没有解析到有效数据。")
else:
    # Epoch 竖线
    for boundary in sorted_boundaries:
        ax_left.axvline(x=boundary, color='#666666', linestyle='--', linewidth=1.5, alpha=0.5)

    ax_left.set_xlabel('Global Iterations', fontsize=12)
    ax_left.set_ylabel('Loss (main)', fontsize=12)

    # 🟢 [修改] 应用左轴固定范围
    if y_limit_main is not None:
        ax_left.set_ylim(y_limit_main)

    ax_right.set_ylabel('Orth Loss', fontsize=12, color='gray')
    ax_left.grid(True, linestyle='--', alpha=0.3)

    # 合并 legend
    lines = handles_left + handles_right
    lbls = labels_left + labels_right
    if lines:
        ax_left.legend(lines, lbls, fontsize=10, loc='upper right')

    plt.title(f'Training Loss Trends (smooth={smooth_weight}) - {file_path}', fontsize=16)
    plt.tight_layout()
    mplcursors.cursor(hover=True)
    plt.show()

    print(f"✅ 绘图成功！检测到 {len(sorted_boundaries)} 次 Epoch 切换。")