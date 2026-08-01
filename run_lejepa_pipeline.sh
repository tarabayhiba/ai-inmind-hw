# Runs the full LeJEPA pipeline: self-supervised pretraining, supervised fine-tuning + test-set evaluation.
#Stops on 1st failure so a broken pretrain run doesn't silently fine-tune garbage
set -e

python pretrain_lejepa.py
python finetune_lejepa.py
