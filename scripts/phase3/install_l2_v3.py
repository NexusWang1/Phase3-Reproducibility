"""Install only the repaired ddm_cues.py overlay; preserve all other modules."""
import argparse,hashlib,shutil
from pathlib import Path

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--ultralytics-root',type=Path,required=True);a=p.parse_args()
 package=Path(__file__).resolve().parent
 source=package/'ultralytics/nn/modules/ddm_cues.py'
 target=(a.ultralytics_root/'nn/modules/ddm_cues.py').resolve()
 expected=(a.ultralytics_root.resolve()/'nn/modules').resolve()
 if target.parent!=expected or not target.is_file():raise RuntimeError(f'Invalid installed DDM cue target: {target}')
 if sha(source)==sha(target):print('[L2-V3] already installed; no write');return
 backup=target.with_name('ddm_cues.py.pre_l2_v3_20260930')
 if backup.exists():raise FileExistsError(f'Backup already exists; refusing overwrite: {backup}')
 shutil.copy2(target,backup);shutil.copy2(source,target)
 if sha(source)!=sha(target):raise RuntimeError('Installed source hash mismatch')
 print(f'[L2-V3] backup={backup} installed={target} sha256={sha(target)}')
if __name__=='__main__':main()
