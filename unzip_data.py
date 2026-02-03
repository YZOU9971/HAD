import os
import zipfile
from tqdm import tqdm
import re
import shutil

# ================= 配置路径 =================
SOURCE_ROOT = r"C:\Users\YZ\Downloads\Compressed"
TARGET_ROOT = r"datasets\NTURGBD"

SAMPLE_PATTERN = re.compile(r'S\d{3}C\d{3}P\d{3}R\d{3}A\d{3}')
# ===========================================


def extract_simple(zip_path, dst_root, mode):
    with zipfile.ZipFile(zip_path, 'r') as zf:
        all_files = zf.infolist()
        for file_info in tqdm(all_files, desc=os.path.basename(zip_path), leave=False):
            if file_info.is_dir():
                continue
            raw_path = file_info.filename               # nturgbd_rgb_s001/S001C003P008R002A060_rgb.avi
            filename = os.path.basename(raw_path)       # S001C003P008R002A060_rgb.avi
            if filename.startswith('.') or 'MACOSX' in raw_path:
                continue

            final_path = None
            if mode == 'flatten':
                if filename.endswith(('.avi', 'skeleton')):
                    final_path = os.path.join(dst_root, filename)
            elif mode == 'folder':
                match = SAMPLE_PATTERN.search(raw_path)
                if match:
                    instance_name = match.group(0)
                    final_path = os.path.join(dst_root, instance_name, filename)

            if final_path:
                os.makedirs(os.path.dirname(final_path), exist_ok=True)
                with zf.open(file_info) as src, open(final_path, 'wb') as dst:
                    shutil.copyfileobj(src, dst)


def main():
    tasks = [
        # ('skeletons',       'nturgbd_skeletons_s{:03d}.zip',    'flatten'),
        ('rgb',             'nturgbd_rgb_s{:03d}.zip',          'flatten'),
        ('ir',              'nturgbd_ir_s{:03d}.zip',           'flatten'),
        ('depth_masked',    'nturgbd_depth_masked_s{:03d}.zip', 'folder')
    ]

    for sub_folder, name_pattern, mode in tasks:
        dst_dir = os.path.join(TARGET_ROOT, sub_folder)
        os.makedirs(dst_dir, exist_ok=True)
        print(f"\n >>> Processing modality: {sub_folder} | mode: {mode}.")
        for i in range(1, 33):
            zip_name = name_pattern.format(i)
            zip_path = os.path.join(SOURCE_ROOT, sub_folder, zip_name)
            if not os.path.exists(zip_path):
                zip_path = os.path.join(SOURCE_ROOT, zip_name)

            if os.path.exists(zip_path):
                print(f"\n[{i}/32] {zip_name}...")
                try: extract_simple(zip_path, dst_dir, mode)
                except Exception as e: print(f"    [ERROR] {e}")
            else: pass


if __name__ == "__main__":
    main()