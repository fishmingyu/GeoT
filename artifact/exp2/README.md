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

Then we draw the figure
```bash
python ablation_rule.py
```