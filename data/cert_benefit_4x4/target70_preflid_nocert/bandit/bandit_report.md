# Online Matching-Bandit Experiment

* setting: online learning, NO train/test split; utilities from the FULL data; symmetric (mirror) preferences -> UNIQUE stable matching H* (men-GS == women-GS == cascade, verified)
* W(H*) = 2.5927; Hungarian = 2.7185
* BT feedback, eta_task=4.9605, eta_model=4.4838
* same estimator (MLE-GS) for both algorithms; only the SAMPLING policy differs (P2ETG uniform round-robin vs PrefLID RRT rounds)

## preflid (20 seeds)

* stopped (certified): 3/20; final exact: 20/20
* final cumulative regret: mean 249.0 (std 354.2); median 136.2
* final instantaneous regret: mean 0.0000; best-seen mean 0.0000
* elapsed: mean 1.6s/seed
