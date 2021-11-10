python two_branch/two_branch_train.py --dataset FB15k_rel_two --data_path datasets/fb15k_rel.pk --loss-lambda 0.8 \
--num-hidden 8 --num-heads 8 --spm --residual --norm --loss-alpha 0.3 --list-num 100 \
--save-path gtran-list-3-100_checkpoint.pt --gpu 0