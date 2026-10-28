# Experiment log — AIMO-Interp Small Models Track

Protocol: hypothesis → design → control → run with fixed seeds → official metric (accuracy) plus balanced accuracy/AUC (test set is near-balanced) → accept/reject. Evaluation population = rows labelled by one of the four Small-Track models; splits grouped by problem (no problem in both train and test); 5 seeds x 5 folds unless stated. Negative results are logged.

## E0 — contract check (2026-10-28)
- Hypothesis: the always-true bundle passes the official ingestion contract for the Small track.
- Local run (official run_local.py, public val-sample, --small): accuracy 0.375, coverage 1.0, invalid 0.
- Codabench submission 1015190 (solution-always-true-small.zip) uploaded 2026-10-28 08:26 CET.
- Result: Codabench 1015190 Finished, Small Model Track, accuracy 0.50, coverage 1.0, invalid 0 (public 14-case set). Contract accepted.

## E1 — layer/pooling grid (2026-10-28, job f2260c0d9ed842d6 features; CPU CV on servidor-ia)
- Hypothesis H1: robustness is linearly decodable from the evaluated model's residual stream before generation, beyond text-surface controls.
- Design: per model, StandardScaler -> PCA(16) -> logistic regression (class_weight=balanced), C in {0.01, 0.1}; features = last prompt token or token mean at relative depth {0, .25, .5, .75, 1} or the standardised 0.4L-0.8L band. "pooled": add problem-level proxy labels from other labelling models (only on problems without an own label). Evaluation only on own-model labels (n=97: 43 olympiad rows from train-main-v2 + 54 MATH rows for DeepSeek). 5 seeds x 5 grouped folds.
- Controls: length features (4 surface statistics), TF-IDF logistic, layer-0 (embeddings), shuffled labels (E3).
- Result (acc / bacc / AUROC, mean over seeds): length 0.703/0.699/0.749; TF-IDF 0.676/0.680/0.731; embeddings mean-pool 0.660/0.662/0.744; last:0.5 pooled 0.736/0.752/0.816; last:band pooled 0.742/0.759/0.807. Own-labels-only probes are weaker (best AUROC 0.786) -> pooling labels across labelling models helps.
- Decision: accept H1 provisionally (mid-depth residual stream > all surface controls on AUROC); confirm with permutation test and paired bootstrap.

## E2 — probe ensembles (same protocol)
- Hypothesis: averaging probes from complementary readouts (last-token band + late mean-pool) improves stability.
- Result: last:band+mean:0.75+mean:1, C=0.1: acc 0.748±0.017, bacc 0.756±0.014, AUROC 0.825±0.009; olympiad subset bacc 0.635±0.020, AUROC 0.762±0.040. Adding length features hurts (AUROC 0.791).
- Decision: accepted as submission candidate S1 (selected among 22 configurations: mild selection bias, disclosed).

## E2b — end-to-end contract test of S1 (job b441079c0e034cd8, RTX 3090, official run_local.py)
- 43 own-labelled train-main-v2 cases, all four models: coverage 1.0, invalid 0, identical output in two runs (in-sample accuracy 0.791, NOT a generalisation estimate). Job wall time 430 s for two full runs including cold model loads (~215 s per run).
- Bundle SHA-256 37af28ba9ee197fd60340fbd60e6959add4155778e097fe91c434d7cc16ded9d.
- Codabench submission 1015306 (S1-probe-ensemble-small.zip), 2026-10-28 09:41 CET.

## E3 — layer sweep (C=0.1, pooled labels, 21 relative depths x 2 poolings, 5 seeds)
- Last token: layer 0 = 0.50 AUROC by construction (identical final template token); 0.663 at 5 % depth; 0.78 at 25-30 %; plateau 0.80-0.81 from 40 % to the last layer (max 0.813 at 50 % and 85-90 %).
- Token mean: 0.740 at layer 0 (bag of embeddings) rising to 0.80 at 65-85 % depth.
- Decision: supports H1's depth prediction (signal is built during prompt processing, not present in raw embeddings at the readout position).

## E4 — paired bootstrap over problems (2000 resamples, seed-averaged OOF probabilities)
- S1 minus control, AUROC: length +0.095 [-0.014, +0.204]; TF-IDF +0.104 [-0.006, +0.214]; embeddings +0.084 [-0.012, +0.187].
- Balanced accuracy: length +0.067 [-0.080, +0.193]; TF-IDF +0.031 [-0.086, +0.150]; embeddings +0.081 [-0.050, +0.207].
- Decision: effect sizes are consistently positive (~+0.09-0.10 AUROC) but 95 % intervals include 0 with only 97 own-labelled rows. Not conclusive at 95 % on own labels alone -> E7 adds power with problem-level labels.
- Codabench 1015306 Finished: public 14-case Small set accuracy 0.571 (8/14), coverage 1.0, invalid 0. With 14 cases one case = 0.071; used only as a validity check (JEV decision).

## Applied preprocessing correction,28Oct2026
CV and ensembles now pass a training-row mask into band normalization. Paired experiment using the shared deployed helper on real archived activations reproduces prior audit: AUROC .832179→.832035, same accuracy .742268. Positive signal retained; this test does not establish future equivalence. The full historical grid remains historical; exporter/inference already used saved train-fitted statistics. Two unit tests enforce isolation from held-out rows and reject missing/empty masks.
Three exploratory stronger text controls and paired intervals: experiments/text_controls_results.json, scripts/stronger_text_controls.py. No extra official submission made by this publication.
