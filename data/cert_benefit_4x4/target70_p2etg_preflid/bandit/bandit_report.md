# Online Matching-Bandit Experiment

* setting: online learning, NO train/test split; utilities from the FULL data; symmetric (mirror) preferences -> UNIQUE stable matching H* (men-GS == women-GS == cascade, verified)
* W(H*) = 2.5927; Hungarian = 2.7185
* BT feedback, eta_task=4.9605, eta_model=4.4838
* same estimator (MLE-GS) for both algorithms; only the SAMPLING policy differs (P2ETG uniform round-robin vs PrefLID RRT rounds)

## p2etg (20 seeds)

* stopped (certified): 1/20; final exact: 20/20
* final cumulative regret: mean 281.2 (std 261.8); median 175.9
* final instantaneous regret: mean 0.0000; best-seen mean 0.0000
* elapsed: mean 1.3s/seed

## preflid (20 seeds)

* stopped (certified): 20/20; final exact: 20/20
* final cumulative regret: mean 249.0 (std 354.2); median 136.2
* final instantaneous regret: mean 0.0000; best-seen mean 0.0000
* elapsed: mean 1.8s/seed
