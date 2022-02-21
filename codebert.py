lang=python #programming language
lr=5e-5
batch_size=16
beam_size=10
source_length=256
target_length=128
data_dir=/home/featurize/data/dataset
output_dir=model1/$lang
train_file=$data_dir/$lang/train.jsonl
dev_file=$data_dir/$lang/valid.jsonl
eval_steps=1000 #400 for ruby, 600 for javascript, 1000 for others
train_steps=50000 #20000 for ruby, 30000 for javascript, 50000 for others
pretrained_model=roberta-base #Roberta: roberta-base
