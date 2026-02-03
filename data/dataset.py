import torch
import numpy as np
import cv2
import os
import glob
import pickle
import re

from torch.utils.data import Dataset
from pathlib import Path

try:
    from decord import VideoReader, cpu, gpu
    ctx = gpu(0) if torch.cuda.is_available() else cpu(0)
    HAS_DECORD = True
except ImportError:
    HAS_DECORD = False
    print("[警告] 未检测到 decord 库，将使用 OpenCV 读取视频，速度可能较慢。建议 pip install decord")

root_path = Path(__file__).parent.parent

# Cross Subject (X-Sub)
NTU60_TRAIN_CS = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
NTU120_TRAIN_CS = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38,
                   45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78, 80, 81, 82,
                   83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103]


class BaseMultiModalDataset(Dataset):
    """
    Basic class for multi-modal dataset with general loading logic
    """
    def __init__(self, data_root, modalities, num_frames=64, image_size=(224, 224), sample_mod='linspace'):
        self.data_root = data_root
        self.modalities = modalities    # [rgb, pose, ir, depth]
        self.num_frames = num_frames
        self.image_size = image_size
        self.samples = []
        self.skeleton_data = None
        self.sample_mod = sample_mod

    def __len__(self):
        return len(self.samples)

    def _get_indices(self, total_frames):
        if total_frames <= 0:
            return np.zeros(self.num_frames, dtype=int)
        if self.sample_mod == 'linspace':
            return np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        elif self.sample_mod == 'repeat':
            if total_frames >= self.num_frames:
                return np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            else:
                base = np.arange(total_frames)
                pad = np.full(self.num_frames - total_frames, total_frames - 1)
                idx = np.concatenate([base, pad])
            return idx
        else:
            raise NotImplementedError

    def _process_video_decord(self, video_path):
        if not os.path.exists(video_path):
            return torch.zeros(3, self.num_frames, *self.image_size)
        try:
            vr = VideoReader(video_path, ctx=cpu(0), width=self.image_size[1], height=self.image_size[0])
            indices = self._get_indices(len(vr))
            frames = vr.get_batch(indices).asnumpy()
            # (T, H, W, C) -> (C, T, H, W)
            tensor = torch.tensor(frames, dtype=torch.float32).permute(3, 0, 1, 2)
            return tensor / 255.0
        except: return torch.zeros(3, self.num_frames, *self.image_size)

    def _process_video_cv2(self, video_path):
        frames = []
        if not os.path.exists(video_path):
            return torch.zeros(3, self.num_frames, *self.image_size)
        cap = cv2.VideoCapture(video_path)
        indices = self._get_indices(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, self.image_size) # BGR -> RGB
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else: frames.append(np.zeros((*self.image_size, 3), dtype=np.uint8))
        cap.release()
        tensor = torch.tensor(np.array(frames), dtype=torch.float32).permute(3, 0, 1, 2)
        return tensor / 255.0

    def _process_image_folder(self, folder_path, suffix, is_color=True):
        if not os.path.exists(folder_path):
            channels = 3 if is_color else 1
            return torch.zeros(channels, self.num_frames, *self.image_size)

        files = glob.glob(os.path.join(folder_path, f'*{suffix}'))
        pattern = re.compile(r'frame_(\d+)_', re.IGNORECASE)

        def sort_key(fname):
            match = pattern.search(os.path.basename(fname))
            return int(match.group(1)) if match else 0

        files.sort(key=sort_key)

        indices = self._get_indices(len(files))
        frames = []

        for i in indices:
            if i < len(files):
                if is_color:
                    img = cv2.imread(files[i])  # BGR
                    if img is not None:
                        img = cv2.resize(img, self.image_size)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    else: img = np.zeros((*self.image_size, 3), dtype=np.uint8)
                else:
                    img = cv2.imread(files[i], cv2.IMREAD_GRAYSCALE)
                    if img is not None: img = cv2.resize(img, self.image_size)
                    else: img = np.zeros(self.image_size, dtype=np.uint8)
                frames.append(img)
            else:
                # Padding
                if is_color: frames.append(np.zeros((*self.image_size, 3), dtype=np.uint8))
                else: frames.append(np.zeros(self.image_size, dtype=np.uint8))

        tensor = torch.tensor(np.array(frames), dtype=torch.float32)
        if is_color: tensor = tensor.permute(3, 0, 1, 2)            # (T, H, W, C) -> (C, T, H, W)
        else: tensor = tensor.unsqueeze(0)                          # (T, H, W) -> (1, T, H, W)
        return tensor / 255.0

    def _process_depth_folder(self, folder_path):
        images = sorted(glob.glob(os.path.join(folder_path, '*.png')))
        indices = self._get_indices(len(images))
        frames = []
        for i in indices:
            if i < len(images):
                img = cv2.imread(images[i], cv2.IMREAD_GRAYSCALE)
                if img is not None: img = cv2.resize(img, self.image_size)
                else: img = np.zeros(self.image_size)
                frames.append(img)
            else: frames.append(np.zeros(self.image_size))
        # (T, H, W) -> (1, T, H, W)
        tensor = torch.tensor(np.array(frames), dtype=torch.float32)
        return tensor.unsqueeze(0).repeat(3, 1, 1, 1) / 255.0

    def _process_skeleton(self, raw_data):
        """
        Raw: (C, T_raw, V, M) -> Target: (C, T_fixed, V, M)
        """
        C, T_raw, V, M = raw_data.shape
        if T_raw > 0:
            indices = self._get_indices(T_raw)
            data = raw_data[:, indices, :, :]
        else: data = np.zeros((C, self.num_frames, V, M), dtype=np.float32)
        return torch.tensor(data, dtype=torch.float32)

    def __getitem__(self, idx):
        global_idx, meta = self.samples[idx]
        sample_name = meta['name']
        result = {
            'label': int(meta['label']),
            'sample_name': sample_name,
            'missing_skeleton': meta.get('missing_skeleton', False)
        }

        # Pose
        if 'pose' in self.modalities:
            raw_pose_object = self.skeleton_data[global_idx]
            raw_pose = np.array(raw_pose_object, dtype=np.float32)
            result['pose'] = self._process_skeleton(raw_pose)
        # RGB
        if 'rgb' in self.modalities:
            path = os.path.join(self.data_root, 'rgb', f"{sample_name}_rgb.avi")
            result['rgb'] = self._process_video_decord(path) if HAS_DECORD else self._process_video_cv2(path)
        # IR
        if 'ir' in self.modalities:
            path = os.path.join(self.data_root, 'ir', f"{sample_name}_ir.avi")
            result['ir'] = self._process_video_decord(path) if HAS_DECORD else self._process_video_cv2(path)
        # Depth
        if 'depth' in self.modalities:
            path = os.path.join(self.data_root, 'depth_masked', sample_name)
            result['depth'] = self._process_depth_folder(path)

        return result


class NTURGBDDataset(BaseMultiModalDataset):
    def __init__(self,
                 meta_path,
                 skeleton_path,
                 data_root,
                 dataset_version='NTU120',
                 benchmark='xsub',
                 split='train',
                 modalities=['rgb', 'pose', 'ir', 'depth'],
                 num_frames=64,
                 image_size=(224, 224),
                 sample_mod='linspace'):

        super().__init__(data_root, modalities, num_frames, image_size, sample_mod)

        print(f"Loading NTU-RGB+D metadata from {meta_path} ...")
        with open(meta_path, 'rb') as f:
            self.all_meta = pickle.load(f)
        self.skeleton_data = np.load(skeleton_path, allow_pickle=True)

        self.dataset_version = dataset_version
        self.benchmark = benchmark
        self.split = split
        self._filter_samples()
        print(f"NTU-RGB+D {120 if dataset_version == 'NTU120' else 60} | {benchmark} | {split}: {len(self.samples)} samples")

    def _filter_samples(self):
        train_subjects = NTU120_TRAIN_CS if self.dataset_version == 'NTU120' else NTU60_TRAIN_CS

        for idx, meta in enumerate(self.all_meta):
            setup_id = meta['setup_id']
            cam_id = meta['cam_id']
            subject_id = meta['subject_id']
            action_id = meta['label'] + 1

            if self.dataset_version == 'NTU60':
                if setup_id >= 18 or action_id > 60:
                    continue

            is_train = False
            if self.benchmark == 'xsub':
                is_train = (subject_id in train_subjects)
            elif self.benchmark == 'xview':
                is_train = (cam_id != 1)
            elif self.benchmark == 'xsetup':
                is_train = (setup_id % 2 == 0)

            target_is_train = (self.split in ['train', 'val'])

            if (target_is_train and is_train) or (self.split == 'test' and not is_train):
                self.samples.append((idx, meta))


class NUCLADataset(BaseMultiModalDataset):
    def __init__(self,
                 meta_path,
                 skeleton_path,
                 data_root,
                 split='train',
                 modalities=['rgb', 'pose', 'depth'],
                 num_frames=64,
                 image_size=(224, 224),
                 sample_mod='linspace'):

        safe_modalities = [m for m in modalities if m != 'ir']
        super().__init__(data_root, safe_modalities, num_frames, image_size, sample_mod)

        print(f"Loading N-UCLA metadata from {meta_path} ...")
        with open(meta_path, 'rb') as f:
            self.all_meta = pickle.load(f)

        print(f"Loading N-UCLA skeletons from {skeleton_path} ...")
        self.skeleton_data = np.load(skeleton_path, allow_pickle=True)

        self.split = split
        self._filter_samples()
        print(f"N-UCLA | {split} | {len(self.samples)} samples")

    def _filter_samples(self):
        """
        Train: View 1, View 2
        Test:  View 3
        """
        for idx, meta in enumerate(self.all_meta):
            view_id = meta.get('view_id', 0)

            is_train = (view_id == 1 or view_id == 2)

            target_is_train = (self.split in ['train', 'val'])

            if (target_is_train and is_train) or (self.split == 'test' and not is_train):
                if meta.get('quality') != 'missing':
                    self.samples.append((idx, meta))

    def __getitem__(self, idx):
        original_idx = self.samples[idx][0]
        meta = self.samples[idx][1]

        sample_name = meta['sample_name']
        file_name = meta['file_name']
        view_id = meta['view_id']

        result = {
            'label': int(meta['label']),
            'sample_name': sample_name,
            'missing_skeleton': meta.get('missing_skeleton', False)
        }

        if 'pose' in self.modalities:
            raw_pose_object = self.skeleton_data[original_idx]
            result['pose'] = self._process_skeleton(raw_pose_object)

        clip_path = os.path.join(self.data_root, f"view_{view_id}", file_name)

        if 'rgb' in self.modalities:
            result['rgb'] = self._process_image_folder(clip_path, '_rgb.jpg', is_color=True)

        if 'depth' in self.modalities:
            result['depth'] = self._process_image_folder(clip_path, '_depth.png', is_color=False)

        return result


def get_dataset(dataset_name, split, args, num_frames=64, image_size=(224, 224),sample_mod='linspace'):
    """
    Args:
         dataset_name: 'NTU60', 'NTU120', 'NUCLA'
         split: 'train', 'val', 'test'
         args: Dict, including 'modalities', 'benchmark', 'use_val', 'num_frames'...
    """
    NTU_ROOT = os.path.join(root_path, r"datasets\NTURGBD")
    NTU_META = os.path.join(NTU_ROOT, 'meta.pkl')
    NUCLA_ROOT = os.path.join(root_path, r"datasets\NUCLA")
    NUCLA_META = os.path.join(NUCLA_ROOT, 'meta.pkl')

    modalities = args.get('modalities', ['rgb', 'pose', 'ir', 'depth'])
    num_frames = args.get('num_frames', num_frames)

    dataset = None

    if 'NTU' in dataset_name:
        benchmark = args.get('benchmark', 'xsub')
        dataset = NTURGBDDataset(
            meta_path=NTU_META,
            data_root=NTU_ROOT,
            skeleton_path=os.path.join(NTU_ROOT, 'skeletons.npy'),
            dataset_version=dataset_name,
            benchmark=benchmark,
            split=split,
            modalities=modalities,
            num_frames=num_frames,
            image_size=image_size,
            sample_mod=sample_mod,
        )

    elif 'NUCLA' in dataset_name:
        dataset = NUCLADataset(
            meta_path=NUCLA_META,
            skeleton_path=os.path.join(NUCLA_ROOT, 'skeletons.npy'),
            data_root=NUCLA_ROOT,
            split=split,
            modalities=modalities,
            num_frames=num_frames,
            image_size=image_size,
            sample_mod=sample_mod,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if split in ['train', 'val'] and args.get('use_val', False) and dataset is not None:
        total_samples = len(dataset.samples)
        rng = np.random.RandomState(42)
        indices = rng.permutation(total_samples)

        split_point = int(total_samples * 0.1)

        if split == 'val':
            selected_indices = indices[:split_point]
        else:
            selected_indices = indices[split_point:]

        dataset.samples = [dataset.samples[i] for i in selected_indices]
        print(f"-> Split '{split}' resized to {len(dataset)} (Val Ratio: 0.1)")

    return dataset
