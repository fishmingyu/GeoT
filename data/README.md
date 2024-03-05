## Download augmented dataset

```bash
pip install gdown
python download_train_data.py
```

Or you could augment the data locally by runing
```bash
python augment_dataset.py
```

## Download results.py
```bash
python download_results.py
```

## Process the groundtruth
```bash
cd process
python filter_pr.py
python filter_sr.py
```

## Run analysis
```bash
sudo apt-get install msttcorefonts
```