## Potential Energy based Mixture Model for Noisy Label Learning
#@ Wenbin Yang Ralph.Yang@dell.com
#@ Zijia Wang zijia_wang@dell.com
# Based on https://github.com/HanxunH/SCELoss-Reproduce
# Example for 0.4 Symmetric noise rate with PEMM

```
 python -u train_PEMM.py  --loss         PEMM               \
                      --dataset_type cifar10           \
                        --l2_reg       1e-2              \
                        --seed         123               \
                        --alpha        0.1               \
                        --beta         1.0               \
                        --version      CIFAR10           \
                        --nr           0.4               \
                     --batch_size      512               \
                     --data_nums_workers 1               \
                    --epoch            60                \
                     --checkpoint_path 'mdoel_s.pkl'
```

# Example for 0.4 Asymmetric noise rate with PEMM
```
python -u train_PEMM.py  --loss         PEMM               \
                      --dataset_type cifar10           \
                        --l2_reg       1e-2              \
                        --seed         123               \
                        --alpha        0.1               \
                        --beta         1.0               \
                        --version      CIFAR100    \
                        --nr           0.3               \
                     --batch_size      256               \
                     --data_nums_workers 1               \
                    --epoch            95                \
                     --asym \
                     --checkpoint_path 'model_a.pkl'
```
# For feature visulization, go to jupyter notebook playground "Tester_1.ipynb" for more detials
