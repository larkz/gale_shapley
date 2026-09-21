# Online Matching-Bandit Experiment

* setting: online learning, NO train/test split; utilities from the FULL data; symmetric (mirror) preferences -> UNIQUE stable matching H* (men-GS == women-GS == cascade, verified)
* W(H*) = 5.5028; Hungarian = 5.5727
* BT feedback, eta_task=11.6463, eta_model=4.9291
* same estimator (MLE-GS) for both algorithms; only the SAMPLING policy differs (P2ETG uniform round-robin vs PrefLID RRT rounds)

## p2etg (10 seeds)

* stopped (certified): 0/10; final exact: 10/10
* final cumulative regret: mean 553.0 (std 509.4); median 429.5
* final instantaneous regret: mean 0.0000; best-seen mean 0.0000
* elapsed: mean 25.6s/seed

## preflid (10 seeds)

* stopped (certified): 9/10; final exact: 10/10
* final cumulative regret: mean 2344.3 (std 810.5); median 2041.9
* final instantaneous regret: mean 0.0000; best-seen mean 0.0000
* elapsed: mean 8.8s/seed
