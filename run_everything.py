
import os

dataroot = '~/x/devsusu/'
batch_size = 50
num_threads = 8

train_cmd_base = "python cyclegan/train.py --name cycada_svhn2mnist_noIdentity " + \
                        "--print_freq 1000 --resize_or_crop=None --loadSize=32 --fineSize=32 " + \
                        "--which_model_netD n_layers --n_layers_D 3 --lambda_A 1 --lambda_B 1 " + \
                        "--lambda_identity 0 --no_flip  --which_direction AtoB " + \
                        "--model cycle_gan_semantic_percep --percep={} --nThreads {} --batchSize {} --dataset_mode mnist_svhn --dataroot {}"

train_batch_size = 200
test_cmd = "python cyclegan/test.py --name cycada_svhn2mnist_noIdentity " + \
                        "--resize_or_crop=None --loadSize=32 --fineSize=32 " + \
                        "--which_model_netD n_layers --n_layers_D 3 --model cycle_gan_semantic_percep " + \
                        "--no_flip --batchSize {} --dataset_mode mnist_svhn " + \
                        "--which_direction AtoB --phase train --how_many 100000 --which_epoch 80 " + \
                        "--dataroot {}"

test_cmd = test_cmd.format(train_batch_size, dataroot, dataroot)
print(test_cmd)

for i in range(1,32):
    os.system('mkdir results')

    train_cmd = train_cmd_base.format(i, num_threads, batch_size, dataroot)
    print(train_cmd)
    os.system(train_cmd)
    os.system(test_cmd)

    os.system('mv results/mnist2svhn {}/'.format(dataroot))

    os.system('mkdir run')

    os.system('python train_adda.py > run/output.txt')
    os.system('mv ./checkpoints/cycada_svhn2mnist_noIdentity run/')
    os.system('mv ./results run/results')
    os.system('mv {}/mnist2svhn run/mnist2svhn'.format(dataroot))
    os.system('mv run run_percep_{}'.format(i))
