# Online Matching-Bandit Experiment

* setting: online learning, NO train/test split; utilities from the FULL data; symmetric (mirror) preferences -> UNIQUE stable matching H* (men-GS == women-GS == cascade, verified)
* W(H*) = 5.5028; Hungarian = 5.5727
* BT feedback, eta_task=11.6463, eta_model=4.9291
* same estimator (MLE-GS) for both algorithms; only the SAMPLING policy differs (P2ETG uniform round-robin vs PrefLID RRT rounds)

## p2etg (10 seeds)

* stopped (certified): 0/10; final exact: 3/10
* final cumulative regret: mean 4690.5 (std 5546.6); median 2694.7
* final instantaneous regret: mean 0.0249; best-seen mean 0.0000
* elapsed: mean 25.4s/seed

## preflid (10 seeds)

* stopped (certified): 0/10; final exact: 3/10
* final cumulative regret: mean 7255.9 (std 4919.3); median 7365.9
* final instantaneous regret: mean 0.0085; best-seen mean 0.0000
* elapsed: mean 9.5s/seed
