import numpy as np
import logging
import sys
import time
import argparse
import os

import torch
from torch import nn
import torchvision.transforms as transforms

from utils.data import datasets
from utils.model import models
from utils.evaluate import Evaluator
from utils.loss import myloss
# import visdom


def main(seed=2018, epoches=80):
    parser = argparse.ArgumentParser(description='my_trans')

    # dataset option
    parser.add_argument('--dataset_name', type=str, default='dtd', choices=['dtd'], help='dataset name (default: my)')
    parser.add_argument('--model_name', type=str, default='dtd', choices=['baseline', 'attention', 'tex'], help='model name (default: my)')
    parser.add_argument('--loss_name', type=str, default='weighted_bce', choices=['weighted_bce', 'DF'], help='model name (default: my)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', help='learning rate (default: auto)')
    parser.add_argument('--checkname', type=int, default=0, help='set the checkpoint name')
    parser.add_argument('--train_batch_size', type=int, default=8,
                        metavar='N', help='input batch size for training (default: auto)')
    parser.add_argument('--test_batch_size', type=int, default=8,
                        metavar='N', help='input batch size for testing (default: auto)')
    # parser.add_argument('--test_iter', type=int, default=200,
    #                     metavar='N', help='iteration for test')
    # parser.add_argument('--lr-scheduler', type=str, default='poly',
    #                     choices=['poly', 'step', 'cos'], help='lr scheduler mode: (default: cos)')

    args = parser.parse_args()

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    if args.dataset_name == 'dtd':
        transform_zk = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5355, 0.4852, 0.4441), std=(0.2667, 0.2588, 0.2667))
        ])
        evaluator = Evaluator(num_class=6)
    # elif args.dataset_name == 'os':
    #     transform_zk = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.4625, 0.3921, 0.3216), std=(0.2735, 0.2645, 0.2647))
    #     ])
    #     evaluator = Evaluator(num_class=6)
    # elif args.dataset_name == 'ADE':
    #     transform_zk = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.4966, 0.4630, 0.4220), std=(0.2547, 0.2520, 0.2680))
    #     ])
    #     evaluator = Evaluator(num_class=7)

    mydataset_embedding = datasets[args.dataset_name]
    data_val1 = mydataset_embedding(split='test1', transform=transform_zk, checkpoint=args.checkname)
    loader_val1 = torch.utils.data.DataLoader(data_val1, batch_size=args.test_batch_size, shuffle=False)
    data_train = mydataset_embedding(split='train', transform=transform_zk, checkpoint=args.checkname)
    loader_train = torch.utils.data.DataLoader(data_train, batch_size=args.train_batch_size, shuffle=True)

    # evaluator = Evaluator(num_class=6)

    dir_name = 'log/' + str(args.dataset_name) + '_' + str(args.model_name) + '_' + str(args.loss_name) + '_' + data_val1.test[0] + '_' + str(args.lr)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    now_time = str(time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()))
    logging.basicConfig(level=logging.INFO,
                        filename=dir_name + '/output_' + now_time + '.log',
                        datefmt='%Y/%m/%d %H:%M:%S',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logging.info('dataset_name: %s, model_name: %s, loss_name: %s', args.dataset_name, args.model_name, args.loss_name)
    logging.info('test with: %s', data_val1.test)

    model = models[args.model_name]()

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    if torch.cuda.is_available():
        model = model.cuda()

    model.train()

    criterion = myloss[args.loss_name]()

    optim_para = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.SGD(optim_para, lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    # viz = visdom.Visdom(env='train')
    # # loss_win = viz.line(np.arange(10))
    # x = 0
    # y = 0
    # loss_win = viz.line(X=np.array([x]), Y=np.array([y]), opts=dict(title='Update'))
    # # acc_win = viz.line(X=np.column_stack((np.array(0), np.array(0))),
    # #                    Y=np.column_stack((np.array(0), np.array(0))))

    IoU_final = 0
    epoch_final = 0
    losses = 0
    visual_loss = []
    iteration = 0

    for epoch in range(epoches):
        scheduler.step()
        train_loss = 0
        logging.info('epoch:' + str(epoch))
        start = time.time()
        np.random.seed(epoch)
        for i, data in enumerate(loader_train):
            _, _, inputs, target, patch, _ = data[0], data[1], data[2], data[3], data[4], data[5]

            inputs = inputs.float()
            iteration += 1
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                target = target.cuda(async=True)
                patch = patch.cuda()

            output = model(inputs, patch)

            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            losses += loss.item()

            if iteration % 20 == 0:
                run_time = time.time() - start
                start = time.time()
                losses = losses / 20
                # visual_loss.append(losses)
                logging.info('iter:' + str(iteration) + " time:" + str(run_time) + " train loss = {:02.5f}".format(losses))

                # viz.line(Y=np.array([losses]), X=np.array([iteration]), update='append', win=loss_win)
                losses = 0

        snapshot_path = dir_name + '/snapshot-epoch_{epoches}_texture.pth'.format(epoches=now_time)
        model.eval()

        # pic_dir = dir_name + '/' + str(epoch) + '/'
        # if epoch % 10 == 9:
        #     if not os.path.exists(pic_dir):
        #         os.mkdir(pic_dir)
        #     visual(model, loader_val1, pic_dir)

        evaluator.reset()
        np.random.seed(2019)
        for i, data in enumerate(loader_val1):
            _, _, inputs, target, patch, image_class = data[0], data[1], data[2], data[3], data[4], data[5]
            inputs = inputs.float()
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                target = target.cuda(async=True)
                patch = patch.cuda()

            scores = model(inputs, patch)
            scores[scores >= 0.5] = 1
            scores[scores < 0.5] = 0
            seg = scores[:, 0, :, :].long()
            pred = seg.data.cpu().numpy()
            target = target.cpu().numpy()
            # Add batch sample into evaluator
            evaluator.add_batch(target, pred, image_class)

        mIoU, mIoU_d = evaluator.Mean_Intersection_over_Union()
        FBIoU = evaluator.FBIoU()

        logging.info("{:10s} {:.3f}".format('IoU_mean', mIoU))
        logging.info("{:10s} {}".format('IoU_mean_detail', mIoU_d))
        logging.info("{:10s} {:.3f}".format('FBIoU', FBIoU))
        if mIoU > IoU_final:
            epoch_final = epoch
            IoU_final = mIoU
            torch.save(model.state_dict(), snapshot_path)
        logging.info('best_epoch:' + str(epoch_final))
        logging.info("{:10s} {:.3f}".format('best_IoU', IoU_final))
        model.train()

    logging.info(epoch_final)
    logging.info(IoU_final)


if __name__ == '__main__':
    main()