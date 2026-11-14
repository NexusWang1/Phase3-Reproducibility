"""Incrementally install Phase3-only files; frozen Phase2 files are untouched."""
import argparse,shutil
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--ultralytics-root',type=Path,required=True);a=p.parse_args();src=Path(__file__).parent
 for rel in ['nn/modules/depth_service.py','nn/modules/ddm_cues.py','nn/modules/ddm_branch.py','nn/modules/ddm_fusion_hook.py','models/yolo/detect/field3d.py']:
  dst=a.ultralytics_root/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src/'ultralytics'/rel,dst);print(f'[install] {rel}')
if __name__=='__main__':main()
