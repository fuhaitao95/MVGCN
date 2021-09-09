## MVGCN

MVGCN: a novel multi-view graph convolutional network (MVGCN) framework for link prediction in biomedical bipartite networks. 

Developer: Fu Haitao from BBDM lab, College of Informatics, Huazhong Agricultural University, Wuhan 430070, China.

## Tutorial

1. Split data for cross validation and indenpendent test experiment via script *split_data.py*: `python split_data.py fold_number DATANAME seed_indent seed_cross`

2. To perform cross validation for finetuning the hyperparameters by running script *command_optimal* (if you don't want to finetune the hyperparameters, just skip this step): 

   `python command_optimal.py --dataName DATANAME --exp_name mid_dim/num_layer/alp_beta --seed_cross seed_cross --seed_indent seed_indent`

3. To get the experiment results by running script *command_optimal.py*: 

   `python command_optimal.py --dataName DATANAME --exp_name optimal_cross --seed_cross seed_cross --seed_indent seed_indent`

   `python command_optimal.py --dataName DATANAME --exp_name optimal_indent --seed_cross seed_cross --seed_indent seed_indent`

## Requirements

numpy 1.18.0

pandas 1.1.0

scipy 1.4.1

scikit-learn 0.22

tensorflow 1.15.0

pytorch 1.6.0

python 3.7.1

## Contact

Please feel free to contact us if you need any help: [fuhaitao@webmail.hzau.edu.cn](mailto:fuhaitao@webmail.hzau.edu.cn)

