## Benchmarking experiments on test functions


All BO runs on test problems are performed using `multi_round_experiment.py`, followed by a last-round full exploit run on the previous results using `redo_last_round_full_exploit.py`. 


`submit.sh` runs all the experiments.


Once all runs are complete, `compute_metrics.py` computes the metrics from the saved results.