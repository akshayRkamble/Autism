import os
import shutil

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src = os.path.dirname(__file__)

for fname in ['image_rf.pkl', 'image_scaler.pkl']:
    src_path = os.path.join(src, fname)
    dst_path = os.path.join(root, fname)
    if os.path.exists(src_path):
        shutil.move(src_path, dst_path)
        print(f"Moved {fname} to root directory.")
    else:
        print(f"{fname} not found in src directory.")
