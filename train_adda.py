if __name__ == '__main__':

    import os
    from os.path import join

    from cycada.tools.train_task_net import train as train_source
    from cycada.tools.test_task_net import load_and_test_net
    from cycada.tools.train_adda_net import train_adda

    import torch
    print(__file__ , 'cuda available: ', torch.cuda.is_available())
    import numpy as np

    # set random seed to 4325
    # to reproduce the exact numbers
    np.random.seed(4325)

    ###################################
    # Set to your preferred data path #
    ###################################
    datadir = '/x/devsusu/'
    ###################################

    # Choose GPU ID
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    # Problem Params
    src = 'mnist2svhn'
    tgt = 'svhn'
    iteration = 1 #'no_cycle'

    base_src = src.split('2')[0]

    model = 'LeNet'
    num_cls = 10

    # Output directory
    outdir = 'results/{}_to_{}/iter_{}'.format(src, tgt, iteration)
    #outdir = 'results/{}_to_{}'.format(src, tgt)

    # Optimization Params
    betas = (0.9, 0.999) # Adam default
    weight_decay = 0 # Adam default
    batch = 128

    src_lr = 1e-4
    src_num_epoch = 100
    src_datadir = join(datadir, src)
    src_net_file = join(outdir, '{}_net_{}.pth'.format(model, src))
    adda_num_epoch = 200
    adda_lr = 1e-5
    adda_net_file = join(outdir, 'adda_{:s}_net_{:s}_{:s}.pth'
            .format(model, src, tgt))


    #######################
    # 1. Train Source Net #
    #######################
    f = open('run/eval.txt', 'a+')
    if os.path.exists(src_net_file):
        print('Skipping source net training, exists:', src_net_file)
    else:
        train_source(src, src_datadir, model, num_cls, file=f,
                outdir=outdir, num_epoch=src_num_epoch, batch=batch,
                lr=src_lr, betas=betas, weight_decay=weight_decay)


    #####################
    # 2. Train Adda Net #
    #####################

    if os.path.exists(adda_net_file):
        print('Skipping adda training, exists:', adda_net_file)
    else:
        train_adda(src, tgt, model, num_cls, file=f, num_epoch=adda_num_epoch,
                batch=batch, datadir=datadir,
                outdir=outdir, src_weights=src_net_file,
                lr=adda_lr, betas=betas, weight_decay=weight_decay)

    ##############################
    # 3. Evalute source and adda #
    ##############################
    tgt_datadir = join(datadir, tgt)
    print()
    if src == base_src:
        print('----------------')
        print('Test set:', src)
        print('----------------')
        print('Evaluating {} source model: {}'.format(src, src_net_file))
        load_and_test_net(src, src_datadir, src_net_file, model, num_cls, file=f,
                dset='test', base_model=None)


    print('----------------')
    print('Test set:', tgt)
    print('----------------')
    print('Evaluating {} source model: {}'.format(src, src_net_file))
    cm = load_and_test_net(tgt, tgt_datadir, src_net_file, model, num_cls, file=f,
            dset='test', base_model=None)

    print(cm)

    print('Evaluating {}->{} adda model: {}'.format(src, tgt, adda_net_file))
    cm = load_and_test_net(tgt, tgt_datadir, adda_net_file, 'AddaNet', num_cls, file=f,
            dset='test', base_model=model)
    print(cm)

    f.close()

    # cyclegan loss, opt 정보 복사해오기
    os.system('cp cyclegan/checkpoints/*.txt runs/'.format(cm))

    # 결과 파일 이름 변경
    os.system('mv run/eval.txt runs/eval_{:.2f}.txt'.format(cm))
