# CyCADA Enhancement with 3-Level Cycle-Consistency
A Enhancement of [CyCADA](https://arxiv.org/pdf/1711.03213.pdf).

## Setup
* Check out the repo (recursively will also checkout the CyCADA fork of the CycleGAN repo).<br>
`git clone --recursive https://github.com/DevSusu/cycada_release.git cycada`
* Install python requirements
    * pip install -r requirements.txt
    
## Train image adaptation only (digits)


```
# 0. VirtualENV 생성
$ python -m venv ENV
$ source ENV/bin/activate
(ENV) $ pip install -r requirements.txt
(ENV) $ cd cyclegan
(ENV) $ pip install -r requirements.txt
```



```
# 1. mnist -> svhn 이미지 Translation
(ENV) $ cd cyclegan

#  training. --batchSize, --nThreads 는 기기에 따라 변경
#  percep = 0 이면 perceptual loss 없음. 1~31 사이로 설정
#  dataroot는 임의로 설정했습니다. 변경하신다면 아래 두 command와 train_adda.py 파일까지 수정하셔야 합니다.
#  ./cyclegan/checkpoints/loss_log.txt 에 loss 정보가 계속 append 됩니다
(ENV) $ python train.py --name cycada_svhn2mnist_noIdentity \
                        --print_freq 1000 --resize_or_crop=None --loadSize=32 --fineSize=32 \
                        --which_model_netD n_layers --n_layers_D 3 \
                        --model cycle_gan_semantic --lambda_A 1 --lambda_B 1 \
                        --lambda_identity 0 --no_flip  --which_direction BtoA \
                        --percep=0 --nThreads 8 --batchSize 50 --dataset_mode svhn_mnist --dataroot /x/devsusu/

#  translation. /x/devsusu/cycada_svhn2mnist_noIdentity/images 에 저장됨(--results_dir 옵션)
(ENV) $ python test.py --name cycada_svhn2mnist_noIdentity --no_flip \ 
                       --resize_or_crop=None --loadSize=32 --fineSize=32 \
                       --which_model_netD n_layers --n_layers_D 3 --model cycle_gan_semantic \
                       --which_direction BtoA --phase train --how_many 100000 --which_epoch 75 \
                       --batchSize 100 --dataset_mode svhn_mnist --dataroot /x/devsusu/ --results_dir /x/devsusu/
```



```
# 2. Evaluating
#  cycada 디렉토리로 이동
(ENV) $ cd ..
(ENV) $ python train_adda.py > eval_0.txt
```

위 1,2를 `percep` 옵션과 `eval_{percep}.txt` 파일명을 바꿔주면서 계속해서 실행합니다.

