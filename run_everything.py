
import os

dataroot = '/x/devsusu/'
batch_size = 50
num_threads = 8

train_cmd_base = "python cyclegan/train.py --name cycada_svhn2mnist_noIdentity " + \
                        "--print_freq 1000 --resize_or_crop=None --loadSize=32 --fineSize=32 " + \
                        "--which_model_netD n_layers --n_layers_D 3 --lambda_A 1 --lambda_B 1 " + \
                        "--lambda_identity 0 --no_flip  --which_direction AtoB " + \
                        "--model cycle_gan_semantic_percep --percep={} --nThreads {} --batchSize {} --dataset_mode mnist_svhn --dataroot {}"

train_batch_size = 100
test_cmd = "python cyclegan/test.py --name cycada_svhn2mnist_noIdentity " + \
                        "--resize_or_crop=None --loadSize=32 --fineSize=32 " + \
                        "--which_model_netD n_layers --n_layers_D 3 --model cycle_gan_semantic_percep " + \
                        "--no_flip --batchSize {} --dataset_mode mnist_svhn " + \
                        "--which_direction AtoB --phase train --how_many 100000 --which_epoch 80" + \
                        "--dataroot {} --results_dir {}"

test_cmd = test_cmd.format(train_batch_size, dataroot, dataroot)

for i in range(1,32):
    train_cmd = train_cmd_base.format(i, num_threads, batch_size, dataroot)
    os.system(train_cmd)
    os.system(test_cmd)

    os.system('mkdir run')

    os.system('python train_adda.py')
    os.system('mv {}/mnist_svhn run/'.format(dataroot))
    os.system('mv run run_percep_{}'.format(i))
