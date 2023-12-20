```python
python train.py --sigma 1.0 --dataset mnist --N_m 1 --atk_method onepixel  --poison_r 0.02 --delta 0.1 --epoch_switch 0
python train.py --sigma 1.0 --dataset mnist --N_m 1 --atk_method onepixel  --poison_r 0.02 --delta 0.1 --epoch_switch 0
```

```python
python train.py --sigma 1.0 --dataset cifar --N_m 1 --atk_method onepixel  --poison_r 0.02 --delta 0.1 --epoch_switch 0
python eval.py --sigma 1.0 --dataset cifar --N_m 1 --atk_method onepixel  --poison_r 0.02 --delta 0.1 --epoch_switch 0
```

```
python train.py --sigma 1.0 --dataset imagenet --N_m 1 --atk_method onepixel  --poison_r 0.02 --delta 0.1 --epoch_switch 0
python eval.py --sigma 1.0 --dataset imagenet --N_m 1 --atk_method onepixel  --poison_r 0.02 --delta 0.1 --epoch_switch 0
```

