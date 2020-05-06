# CyCADA Enhancement with 3-Level Cycle-Consistency
A Enhancement of [CyCADA](https://arxiv.org/pdf/1711.03213.pdf).

## Setup
* Check out the repo (recursively will also checkout the CyCADA fork of the CycleGAN repo).<br>
`git clone --recursive https://github.com/DevSusu/cycada_release.git cycada`
* Install python requirements
    * pip install -r requirements.txt
    
## Train image adaptation only (digits)
```
cd cyclegan
pip install -r requirements.txt
./train_cycada.sh
./test_cycada.sh all
```
* Image adaptation builds on the work on [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). The submodule in this repo is a fork which also includes the semantic consistency loss. 
* Added perceptual loss on cyclegan_semantic model
* Producing SVHN as MNIST 
   * For an example of how to train image adaptation on SVHN->MNIST, see `cyclegan/train_cycada.sh`. From inside the `cyclegan` subfolder run `train_cycada.sh`. 
   * The snapshots will be stored in `cyclegan/cycada_svhn2mnist_noIdentity`. Inside `test_cycada.sh` set the epoch value to the epoch you wish to use and then run the script to generate 50 transformed images (to preview quickly) or run `test_cycada.sh all` to generate the full ~73K SVHN images as MNIST digits. 
   * Results are stored inside `cyclegan/results/cycada_svhn2mnist_noIdentity/train_75/images`. 
   * Note we use a dataset of mnist_svhn and for this experiment run in the reverse direction (BtoA), so the source (SVHN) images translated to look like MNIST digits will be stored as `[label]_[imageId]_fake_B.png`. Hence when images from this directory will be loaded later we will only images which match that naming convention.

## Train feature adaptation following image adaptation
* Use the feature space adapt code with the data and models from image adaptation
* For example: to train for the SVHN to MNIST shift, set `src = 'mnist2svhn'` and `tgt = 'svhn'` inside `scripts/train_adda.py`
* Either download the relevant images above or run image space adaptation code and extract transferred images
