# Online Matching-Bandit Experiment

* setting: online learning, NO train/test split; utilities from the FULL data; symmetric (mirror) preferences -> UNIQUE stable matching H* (men-GS == women-GS == cascade, verified)
* W(H*) = 2.1104; Hungarian = 2.1104
* BT feedback, eta_task=4.2983, eta_model=4.2940
* same estimator (MLE-GS) for both algorithms; only the SAMPLING policy differs (P2ETG uniform round-robin vs PrefLID RRT rounds)

## p2etg (10 seeds)

* stopped (certified): 10/10; final exact: 10/10
* final cumulative regret: mean 41.0 (std 43.2); median 29.0
* final instantaneous regret: mean 0.0000; best-seen mean 0.0000
* elapsed: mean 0.1s/seed

## preflid (10 seeds)

* stopped (certified): 10/10; final exact: 9/10
* final cumulative regret: mean 1447.6 (std 4511.8); median 17.5
* final instantaneous regret: mean 0.0071; best-seen mean 0.0000
* elapsed: mean 0.1s/seed
