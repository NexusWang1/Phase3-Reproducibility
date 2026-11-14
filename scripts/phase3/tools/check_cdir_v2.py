"""Pure-numpy c_dir v2 synthetic regression and monotonicity checks."""
import importlib.util
import os
from pathlib import Path
import numpy as np


def main():
    os.environ['PHY_CDIR_FS']='.5';os.environ['PHY_CDIR_MINH']='10'
    file=Path(__file__).resolve().parents[1]/'ultralytics/models/yolo/detect/field3d.py'
    spec=importlib.util.spec_from_file_location('phase3_field3d_local',file)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    c=mod.deflection_confidence
    h=[[1.,0.]]*10;f=np.full((10,10),.8,np.float32)
    full=c(f,[1.,0.],h,kept_classes={0})
    no_box=c(f,[1.,0.],h,kept_classes={2})
    reverse=c(f,[1.,0.],[[-1.,0.]]*10,kept_classes={1})
    cold=c(f,[1.,0.],[[1.,0.]],kept_classes={0})
    weak=c(np.full((10,10),.1,np.float32),[1.,0.],h,kept_classes={0})
    blank=c(np.zeros((10,10),np.float32),[1.,0.],h,kept_classes={0})
    assert np.isclose(full,1.) and np.isclose(no_box,.3) and np.isclose(reverse,.3)
    assert np.isclose(cold,.3) and 0<weak<full and blank==0.
    assert full>no_box and full>reverse and full>cold
    print('[DDM-V2-CDIR] PASS',{'full':full,'no_box':no_box,'reverse':reverse,'cold':cold,'weak':weak,'blank':blank})


if __name__=='__main__':main()
