python3.6 preprocess.py -train_src data/train/src.txt -train_tgt data/train/tgt.txt -valid_src data/validation/src.txt -valid_tgt data/validation/tgt.txt -save_data data/demo
python3.6 train.py -data data/demo -save_model demo-model --learning_rate 0.01 --early_stopping 100 --train_steps 15000
python3.6 translate.py -model demo-model_XYZ.pt -src data/test/src.txt -output data/pred.txt