# Online Matching-Bandit Experiment

* setting: online learning, NO train/test split; utilities from the FULL data; symmetric (mirror) preferences -> UNIQUE stable matching H* (men-GS == women-GS == cascade, verified)
* W(H*) = 7.1137; Hungarian = 7.1835
* BT feedback, eta_task=8.2399, eta_model=3.7146
* same estimator (MLE-GS) for both algorithms; only the SAMPLING policy differs (P2ETG uniform round-robin vs PrefLID RRT rounds)

## p2etg (10 seeds)

* stopped (certified): 0/10; final exact: 10/10
* final cumulative regret: mean 6926.2 (std 4696.1); median 6383.0
* final instantaneous regret: mean 0.0000; best-seen mean 0.0000
* elapsed: mean 48.8s/seed

## preflid (10 seeds)

* stopped (certified): 0/10; final exact: 9/10
* final cumulative regret: mean 9711.1 (std 8997.2); median 5429.9
* final instantaneous regret: mean 0.0100; best-seen mean 0.0100
* elapsed: mean 10.6s/seed
