# CyCADA Enhancement with 3-Level Cycle-Consistency
A Enhancement of [CyCADA](https://arxiv.org/pdf/1711.03213.pdf).
    
# Train image adaptation only (digits)

## 0. VirtualENV 생성
```
$ git clone --recursive https://github.com/DevSusu/cycada_release.git cycada
$ cd cycada
$ python -m venv ENV
$ source ENV/bin/activate
(ENV) $ pip install -r requirements.txt
(ENV) $ cd cyclegan
(ENV) $ pip install -r requirements.txt
```

## 1. mnist -> svhn 이미지 Translation
* dataroot는 임의로 설정했습니다. **변경하신다면 아래 두 command와 train_adda.py 파일을 수정**하시면 됩니다.
* dataroot에 svhn의 test, train.mat 이 있어야 합니다
* train 시 `./cyclegan/checkpoints/loss_log.txt` 에 loss 정보가 계속 append 됩니다
* train 시` --batchSize` , `--nThreads` 는 기기에 따라 변경하면 됩니다.
* perceptual loss 없는 버전은 `--model cycle_gan_semantic` 옵션을 주면 됨. 1~31 사이로 설정
* translate 결과는 `--results_dir` 옵션에 따라 저장됨

```
(ENV) $ cd cyclegan
(ENV) $ python train.py --name cycada_svhn2mnist_noIdentity \
                        --print_freq 1000 --resize_or_crop=None --loadSize=32 --fineSize=32 \
                        --which_model_netD n_layers --n_layers_D 3 --lambda_A 1 --lambda_B 1 \
                        --lambda_identity 0 --no_flip  --which_direction AtoB \
                        --model cycle_gan_semantic_percep --percep=1 --nThreads 8 --batchSize 50 --dataset_mode mnist_svhn --dataroot /x/devsusu/

(ENV) $ python test.py --name cycada_svhn2mnist_noIdentity --no_flip \ 
                       --resize_or_crop=None --loadSize=32 --fineSize=32 \
                       --which_model_netD n_layers --n_layers_D 3 --model cycle_gan_semantic \
                       --batchSize 100 --phase train --how_many 100000 --which_epoch 75 \
                       --which_direction AtoB --dataset_mode mnist_svhn --dataroot /x/devsusu/ --results_dir /x/devsusu/mnist2svhn
```

## 2. 결과 Evaluating
* cycada 디렉토리에서 이루어집니다.

```
(ENV) $ cd ..
(ENV) $ python train_adda.py
```

위 1,2를 `percep` 옵션과 `eval_{percep}.txt` 파일명을 바꿔주면서 계속해서 실행합니다.

