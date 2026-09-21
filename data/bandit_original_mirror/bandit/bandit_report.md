# Online Matching-Bandit Experiment

* setting: online learning, NO train/test split; utilities from the FULL data; symmetric (mirror) preferences -> UNIQUE stable matching H* (men-GS == women-GS == cascade, verified)
* W(H*) = 4.7163; Hungarian = 5.2316
* BT feedback, eta_task=6.5667, eta_model=5.1446
* same estimator (MLE-GS) for both algorithms; only the SAMPLING policy differs (P2ETG uniform round-robin vs PrefLID RRT rounds)

## p2etg (10 seeds)

* stopped (certified): 0/10; final exact: 4/10
* final cumulative regret: mean 1795.5 (std 1517.7); median 1225.8
* final instantaneous regret: mean 0.0021; best-seen mean 0.0000
* elapsed: mean 25.1s/seed

## preflid (10 seeds)

* stopped (certified): 0/10; final exact: 2/10
* final cumulative regret: mean 142.0 (std 103.1); median 113.3
* final instantaneous regret: mean 0.0159; best-seen mean 0.0000
* elapsed: mean 0.4s/seed
