"""Paired CPU audit of a transductive transform on actual archived activations."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='2'
import argparse,json,hashlib,datetime,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from band_normalization import normalize_band
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,balanced_accuracy_score,accuracy_score

MODELS={'Qwen/Qwen3.5-4B':'Qwen_Qwen3.5-4B','deepseek-ai/DeepSeek-R1-0528-Qwen3-8B':'deepseek-ai_DeepSeek-R1-0528-Qwen3-8B','allenai/Olmo-3-7B-Think':'allenai_Olmo-3-7B-Think','Skywork/Skywork-OR1-Math-7B':'Skywork_Skywork-OR1-Math-7B'}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--out',type=Path,default=Path('output'));args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
 lab=pd.read_csv(args.input/'data/labels.csv').dropna(subset=['label']);lab['eval_model']=lab.model.str.split('|').str[0]
 hashes={}; rec=[]
 for target,safe in MODELS.items():
  own=lab[lab.eval_model==target][['problem_id','label','source']].assign(own=True)
  other=lab[(lab.eval_model!=target)&~lab.problem_id.isin(own.problem_id)]
  prox=other.groupby('problem_id',as_index=False).label.mean();prox.label=prox.label.round();prox['own']=False;prox['source']='proxy'
  r=pd.concat([own,prox],ignore_index=True);pids=r.problem_id.to_numpy();y=r.label.to_numpy(dtype=int)
  p=args.input/'features'/(safe+'.npz');hashes[safe]=hashlib.sha256(p.read_bytes()).hexdigest()
  with np.load(p) as z:
   ix={v:i for i,v in enumerate(z['ids'])};idx=[ix[x] for x in pids]
   last=z['last'][idx].astype(np.float32);mean=z['mean'][idx].astype(np.float32)
  L=last.shape[1]-1; band=last[:,int(.4*L):int(.8*L)+1]
  mean75=mean[:,int(.75*L)];mean1=mean[:,-1]
  # Match the audited transform exactly on all rows, then compare train-only statistics.
  all_band=((band-band.mean(axis=0))/(band.std(axis=0)+1e-6)).mean(axis=1)
  for seed in range(3):
   rng=np.random.default_rng(seed);ug=rng.permutation(np.unique(pids));fm={g:i%5 for i,g in enumerate(ug)};fold=np.array([fm[g] for g in pids])
   for f in range(5):
    tr=fold!=f;te=(fold==f)&r.own.to_numpy();ii=np.flatnonzero(te)
    if not len(ii):continue
    assert not set(pids[tr])&set(pids[te])
    train_band=normalize_band(band,tr)
    ps={}
    for mode,b in [('all_data',all_band),('train_only',train_band)]:
     pp=[]
     for X in [b,mean75,mean1]:
      clf=make_pipeline(StandardScaler(),PCA(n_components=min(16,int(tr.sum())-1),svd_solver='randomized',random_state=seed),LogisticRegression(C=.1,class_weight='balanced',max_iter=2000,random_state=seed))
      clf.fit(X[tr],y[tr]);pp.append(clf.predict_proba(X[te])[:,1])
     ps[mode]=np.mean(pp,axis=0)
    for j,i in enumerate(ii):rec.append(dict(seed=seed,model=target,problem_id=pids[i],truth=int(y[i]),all_data=float(ps['all_data'][j]),train_only=float(ps['train_only'][j])))
  print(json.dumps({'model':target,'own_rows':len(own),'status':'done'}),flush=True)
 d=pd.DataFrame(rec);d.to_csv(args.out/'aimo_band_oof.csv',index=False)
 ms=[]
 for seed,g in d.groupby('seed'):
  for mode in ['all_data','train_only']:ms.append(dict(seed=int(seed),mode=mode,auroc=roc_auc_score(g.truth,g[mode]),accuracy=accuracy_score(g.truth,g[mode]>=.5),balanced_accuracy=balanced_accuracy_score(g.truth,g[mode]>=.5)))
 avg=d.groupby(['model','problem_id','truth'],as_index=False)[['all_data','train_only']].mean();groups=avg.problem_id.unique();ix={g:np.flatnonzero(avg.problem_id.to_numpy()==g) for g in groups};rng=np.random.default_rng(20261028);ds=[]
 for _ in range(2000):
  ii=np.concatenate([ix[g] for g in rng.choice(groups,len(groups),replace=True)]);g=avg.iloc[ii]
  if g.truth.nunique()==2:ds.append(roc_auc_score(g.truth,g.train_only)-roc_auc_score(g.truth,g.all_data))
 out={'kind':'new_paired_preprocessing_intervention_on_real_whitebox_activations','source_job':'f2260c0d9ed842d6','source_hashes':hashes,'seeds':[0,1,2],'folds':5,'rows_per_seed':d.groupby('seed').size().to_dict(),'metrics':ms,'mean_metrics':pd.DataFrame(ms).groupby('mode')[['auroc','accuracy','balanced_accuracy']].mean().to_dict(orient='index'),'delta_auroc_on_mean_predictions':roc_auc_score(avg.truth,avg.train_only)-roc_auc_score(avg.truth,avg.all_data),'paired_problem_ci95':np.quantile(ds,[.025,.975]).tolist(),'caveats':['Exploratory; known public labels and selected ensemble.','Randomized PCA fixed in both arms; not an exact reproduction of sklearn auto full-PCA original.','No new GPU inference; measures preprocessing effect, not general causal circuit interpretation.'],'generated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 (args.out/'aimo_band_audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()
