# Online Matching-Bandit Experiment

* setting: online learning, NO train/test split; utilities from the FULL data; symmetric (mirror) preferences -> UNIQUE stable matching H* (men-GS == women-GS == cascade, verified)
* W(H*) = 2.5927; Hungarian = 2.7185
* BT feedback, eta_task=12.8635, eta_model=11.6274
* same estimator (MLE-GS) for both algorithms; only the SAMPLING policy differs (P2ETG uniform round-robin vs PrefLID RRT rounds)

## preflid (20 seeds)

* stopped (certified): 18/20; final exact: 20/20
* final cumulative regret: mean 45.7 (std 40.7); median 32.5
* final instantaneous regret: mean 0.0000; best-seen mean 0.0000
* elapsed: mean 1.1s/seed
