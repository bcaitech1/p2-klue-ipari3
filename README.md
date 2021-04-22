# Pstage_03_KLUE_Relation_extraction

### install
- $`pip install -r requirements.txt`

### connect
* Notebook: http://27.96.130.126:8890?token=BAwBBg0aG0RBSkYhAAhABwlBFgMZ
* Tensorboard:  
    - $`tensorboard --logdir=.`
    - http://localhost:6006/
    - (X) http://27.96.130.126:6008

### training
* python train.py

### inference
* python inference.py --model_dir=[model_path]
* ex) python inference.py --model_dir=./results/checkpoint-500

### evaluation
* python eval_acc.py
