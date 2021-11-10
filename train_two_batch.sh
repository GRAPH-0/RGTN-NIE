 python two_branch/two_branch_batch_train.py --dataset IMDB_S_rel_two --data_path datasets/imdb_s_rel.pk \
 --loss-lambda 0.7 --loss-alpha 0.6 --list-num 100 --residual --norm --num-hidden 8 --num-heads 8 --num-out-heads 8 \
 --save-path gtrans-list-716-100_checkpoint.pt --spm --pred-dim 16 --gpu 1