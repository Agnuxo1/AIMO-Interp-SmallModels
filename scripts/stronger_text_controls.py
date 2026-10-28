import argparse,json,hashlib,datetime
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,accuracy_score,balanced_accuracy_score
ROOT=Path(__file__).resolve().parents[1]
SMALL=['Qwen/Qwen3.5-4B','deepseek-ai/DeepSeek-R1-0528-Qwen3-8B','allenai/Olmo-3-7B-Think','Skywork/Skywork-OR1-Math-7B']

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,default=ROOT/'src/data');args=ap.parse_args()
 lab=pd.read_csv(args.source/'labels.csv').dropna(subset=['label']);lab['eval_model']=lab.model.str.split('|').str[0]
 texts={x['id']:x['problem'] for x in [json.loads(s) for s in (args.source/'problems.jsonl').read_text(encoding='utf-8').splitlines()]}
 records=[]
 for seed in (20261028,20261029,20261030):
  # One common problem->fold mapping across models, so pooled cross-model analyses remain grouped.
  rng=np.random.default_rng(seed); uniq=rng.permutation(sorted(texts));fm={g:i%5 for i,g in enumerate(uniq)}
  for target in SMALL:
   own=lab[lab.eval_model==target][['problem_id','label','source']].assign(own=True)
   other=lab[(lab.eval_model!=target)&(~lab.problem_id.isin(own.problem_id))]
   proxy=other.groupby('problem_id',as_index=False).label.mean();proxy.label=proxy.label.round();proxy['own']=False;proxy['source']='proxy'
   r=pd.concat([own,proxy],ignore_index=True); y=r.label.to_numpy(dtype=int)
   tx=[texts[p] for p in r.problem_id]
   X=np.array([[len(t),t.count('$'),t.count('\\'),len(t.split())] for t in tx],float)
   fold=np.array([fm[p] for p in r.problem_id])
   for f in range(5):
    tr=fold!=f;te=(fold==f)&r.own.to_numpy();ii=np.flatnonzero(te)
    if not len(ii):continue
    assert not set(r.problem_id[tr])&set(r.problem_id[te])
    c=make_pipeline(StandardScaler(),LogisticRegression(C=.1,class_weight='balanced',max_iter=2000,random_state=seed))
    c.fit(X[tr],y[tr]); pc=c.predict_proba(X[te])[:,1]
    cv=make_pipeline(TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),max_features=2000,min_df=2),LogisticRegression(C=.1,class_weight='balanced',max_iter=2000,random_state=seed))
    cv.fit([tx[i] for i in np.flatnonzero(tr)],y[tr]); p1=cv.predict_proba([tx[i] for i in ii])[:,1]
    quad=make_pipeline(PolynomialFeatures(2,include_bias=False),StandardScaler(),LogisticRegression(C=.1,class_weight='balanced',max_iter=2000,random_state=seed))
    quad.fit(np.log1p(X[tr]),y[tr]);p2=quad.predict_proba(np.log1p(X[te]))[:,1]
    for j,i in enumerate(ii):records.append(dict(seed=seed,model=target,problem_id=r.problem_id.iloc[i],source=r.source.iloc[i],truth=int(y[i]),control=float(pc[j]),I1_char=float(p1[j]),I2_quadratic=float(p2[j]),I3_hybrid=float((pc[j]+p1[j])/2)))
 pred=pd.DataFrame(records);pred.to_csv(ROOT/'experiments/text_controls_oof.csv',index=False)
 out={'kind':'new_exploratory_stronger_text_controls','n_rows_per_seed':pred.groupby('seed').size().to_dict(),'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.source/'labels.csv',args.source/'problems.jsonl']},'by_seed':[],'mean_metrics':{}}
 names=['control','I1_char','I2_quadratic','I3_hybrid']
 for seed,g in pred.groupby('seed'):
  for name in names:out['by_seed'].append({'seed':int(seed),'method':name,'auroc':roc_auc_score(g.truth,g[name]),'accuracy':accuracy_score(g.truth,g[name]>=.5),'balanced_accuracy':balanced_accuracy_score(g.truth,g[name]>=.5)})
 sums=pd.DataFrame(out['by_seed']).groupby('method')[['auroc','accuracy','balanced_accuracy']].mean()
 out['mean_metrics']=sums.to_dict(orient='index')
 # Average each case's OOF probabilities across seeds before clustered inference; seeds are not subjects.
 avg=pred.groupby(['model','problem_id','source','truth'],as_index=False)[names].mean()
 rng=np.random.default_rng(20261028);groups=avg.problem_id.unique();ix={g:np.flatnonzero(avg.problem_id.to_numpy()==g) for g in groups}
 samples=[np.concatenate([ix[g] for g in rng.choice(groups,len(groups),replace=True)]) for _ in range(2000)]
 out['paired_problem_cluster_contrasts']={}
 for name in names[1:]:
  ds=[]
  for s in samples:
   g=avg.iloc[s]
   if g.truth.nunique()==2:ds.append(roc_auc_score(g.truth,g[name])-roc_auc_score(g.truth,g.control))
  out['paired_problem_cluster_contrasts'][name]={'delta_auroc':roc_auc_score(avg.truth,avg[name])-roc_auc_score(avg.truth,avg.control),'ci95':np.quantile(ds,[.025,.975]).tolist()}
 out['generated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 (ROOT/'experiments/text_controls_results.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
 print(json.dumps(out,indent=2))
if __name__=='__main__':main()
