## MVGCN

MVGCN: a novel multi-view graph convolutional network (MVGCN) framework for link prediction in biomedical bipartite networks. 

Developer: Fu Haitao from BBDM lab, College of Informatics, Huazhong Agricultural University, Wuhan 430070, China.

## Tutorial

1. Split data for cross validation and indenpendent test experiment via the script *split_data.py*: `python split_data.py fold_number DATANAME seed_indent seed_cross`

2. To perform cross validation for finding the optimal hyperparameters by running the script *command_optimal.py* (if you don't want to finetune the hyperparameters, just skip this step): 

   `python command_optimal.py --dataName DATANAME --exp_name mid_dim/num_layer/alp_beta --seed_cross seed_cross --seed_indent seed_indent`

3. To get the experiment results by running the script *command_optimal.py*: 

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

Please feel free to contact us if you need any help: [fuhaitao@webmail.hzau.edu.cn](mailto:fuhaitao@webmail.hzau.edu.cn) OR [fuhaitao95@qq.com](mailto:fuhaitao95@qq.com)
__Attention__: Only real name emails will be replied. Please provide as much detail as possible about the problem you are experiencing.
__注意__：只回复实名电子邮件。请尽可能详细地描述您遇到的问题，可以附上截图等。
