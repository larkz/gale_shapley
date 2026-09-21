# Online Matching-Bandit Experiment

* setting: online learning, NO train/test split; utilities from the FULL data; symmetric (mirror) preferences -> UNIQUE stable matching H* (men-GS == women-GS == cascade, verified)
* W(H*) = 3.3021; Hungarian = 3.3021
* BT feedback, eta_task=7.9217, eta_model=3.8169
* same estimator (MLE-GS) for both algorithms; only the SAMPLING policy differs (P2ETG uniform round-robin vs PrefLID RRT rounds)

## p2etg (10 seeds)

* stopped (certified): 10/10; final exact: 10/10
* final cumulative regret: mean 324.2 (std 344.0); median 290.5
* final instantaneous regret: mean 0.0000; best-seen mean 0.0000
* elapsed: mean 1.4s/seed

## preflid (10 seeds)

* stopped (certified): 10/10; final exact: 10/10
* final cumulative regret: mean 121.7 (std 54.7); median 133.8
* final instantaneous regret: mean 0.0000; best-seen mean 0.0000
* elapsed: mean 0.8s/seed
