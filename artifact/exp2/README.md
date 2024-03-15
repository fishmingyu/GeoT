## Run exp2

Enter benchmark_cpp directory
```bash
cd benchmark/benchmark_cpp
```
Then run 
```bash
# make sure you rm *csv
python run_ablation.py
python run_eval.py
```

> Note that the ablation result may vary. To keep result in reasonable way, we parser the rule directly to eval benchmark, and treat it as he performance of decision tree.

## Draw figure
```bash
python feature_extract.py
python query_rule.py
python ablation_rule.py
```
