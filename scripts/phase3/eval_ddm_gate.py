"""DDM A/B/C gate runner with preregistered automatic acceptance."""
import argparse,json,subprocess,sys
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--eval-command',required=True,help='command template; use {mode} {w1} {w2} {w3} {vth} {vc0} {out}');p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise FileExistsError(a.out)
 a.out.mkdir(parents=True);groups=[('A',0,.5,.3,.2,0,.25),('B',1,.5,.3,.2,.005,.25),('C',1,.5,.3,.2,.01,.25)];rows=[]
 for name,mode,w1,w2,w3,vth,vc0 in groups:
  od=a.out/name;cmd=a.eval_command.format(mode=mode,w1=w1,w2=w2,w3=w3,vth=vth,vc0=vc0,out=od);code=subprocess.call(cmd,shell=True)
  metrics_path=od/'metrics.json';metrics=json.loads(metrics_path.read_text()) if code==0 and metrics_path.is_file() else None
  rows.append({'group':name,'phy_ddm':mode,'weights':[w1,w2,w3],'veto_t':vth,'veto_c':vc0,'returncode':code,'output_dir':str(od),'metrics':metrics})
 report={'groups':rows,'acceptance':'B mAP50 >= A mAP50 - 0.001 and B hard-negative FP <= 0.9*A; 0.001 allows the v1 observed 2e-5 AP integration-tail noise, not a relaxed quality bar.'}
 if not any(x['returncode'] for x in rows):
  A,B=rows[0]['metrics'],rows[1]['metrics']
  if A is None or B is None:raise RuntimeError('evaluation succeeded but metrics.json is missing')
  report['pass']=bool(B['mAP50']>=A['mAP50']-.001 and B['hardneg_fp']<=.9*A['hardneg_fp'])
  report['decision']='PASS' if report['pass'] else 'FAIL'
 (a.out/'report.json').write_text(json.dumps(report,indent=2));print('[DDM-GATE] '+report.get('decision','FAIL'));print(json.dumps(report,indent=2))
 if any(x['returncode'] for x in rows):raise RuntimeError('one or more evaluation groups failed')
 if not report.get('pass',False):raise SystemExit(2)
if __name__=='__main__':main()
