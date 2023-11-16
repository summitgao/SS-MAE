from utility import accuracy
import math
import time
import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report
from utils.optimizer_step import Optimizer

from utility import output_metric

import argparse

parser = argparse.ArgumentParser("HSI")

# ---- stage
parser.add_argument('--is_train', default=0, type=int)
parser.add_argument('--is_load_pretrain', default=0, type=int)
parser.add_argument('--is_pretrain', default=1, type=int)
parser.add_argument('--is_test', default=0, type=int)
parser.add_argument('--model_file', default='model', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

# ---- network parameter
parser.add_argument('--size_SA', default=49, type=int, help='the size of spatial attention')
parser.add_argument('--channel_num', default=51, type=int, help="the size of spectral attention (berlin 248, augsburg "
                                                                "188, houston 2018 51)")
parser.add_argument('--epoch', default=500, type=int)
parser.add_argument('--pca_num', default=20, type=int)
parser.add_argument('--mask_ratio', default=0.3, type=float)
parser.add_argument('--crop_size', type=int, default=7)

# ----- data
parser.add_argument('--device', default="cuda:0", type=str)
parser.add_argument('--dataset', default='Houston2018', type=str,
                    help='Houston2018 Berlin Augsburg')
parser.add_argument('--num_classes', default=20, type=int, help="berlin 8, augsburg 7, houston 2018 20")
parser.add_argument('--pretrain_num', default=50000, type=int)

# --- vit
parser.add_argument('--patch_size', default=1, type=int)
parser.add_argument('--finetune', default=0, type=int)
parser.add_argument('--mae_pretrain', default=1, type=int)
parser.add_argument('--depth', default=2, type=int)
parser.add_argument('--head', default=4, type=int)
parser.add_argument('--dim', default=256, type=int)

# ---- train
parser.add_argument('--model_name', type=str)
parser.add_argument('--warmup_epochs', default=5, type=int)
parser.add_argument('--test_interval', default=5, type=int)
parser.add_argument('--optimizer_name', default="adamw", type=str)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--cosine', default=0, type=int)
parser.add_argument('--weight_decay', default=5e-2, type=float)
parser.add_argument('--batch_size', default=64, type=int)

args = parser.parse_args()

from get_dat import get_dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

device = torch.device(args.device if torch.cuda.is_available() else "cpu")


from net.VIT.mae import MAEVisionTransformers as MAE
from net.VIT.mae import VisionTransfromers as MAEFinetune
from loss.mae_loss import MSELoss, build_mask_spa, build_mask_chan


pretrain_loader, train_loader, test_loader, trntst_loader, all_loader = get_dataset(args)


def min_max(input):
    return (input - input.min()) / (input.max() - input.min()) * 255


def Pretrain(args,
             scaler,
             model,
             criterion,
             optimizer,
             epoch,
             batch_iter,
             batch_size
             ):
    """Traing with the batch iter for get the metric
    """

    total_loss = 0
    n = 0
    loader_length = len(pretrain_loader)
    print("pretrain_loader-------------", loader_length)
    for batch_idx, (hsi, lidar, _, _, hsi_pca) in enumerate(pretrain_loader):
        n = n + 1
        # TODO: add the layer-wise lr decay
        if args.cosine:
            # cosine learning rate
            lr = cosine_learning_rate(
                args, epoch, batch_iter, optimizer, batch_size
            )
        else:
            # step learning rate
            lr = step_learning_rate(
                args, epoch, batch_iter, optimizer, batch_size
            )

        # forward
        hsi = hsi.to(device)
        hsi = hsi[:, 0, :, :, :]
        lidar = lidar.to(device)

        outputs_spa, mask_index_spa, outputs_chan, mask_index_chan = model(torch.cat((hsi, lidar), 1))
        mask_spa = build_mask_spa(mask_index_spa, args.patch_size, args.crop_size)
        mask_chan = build_mask_chan(mask_index_chan, channel_num=args.channel_num, patch_size=args.patch_size)
        losses = criterion(outputs_spa, torch.cat((hsi, lidar), 1), mask_spa) + criterion(outputs_chan,
                                                                                          torch.cat((hsi, lidar), 1),
                                                                                          mask_chan.unsqueeze(-1))

        optimizer.zero_grad()
        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss = total_loss + losses.data.item()
        batch_iter += 1

        print(
            "Epoch:", epoch,
            " batch_idx:", batch_idx,
            " batch_iter:", batch_iter,
            " losses:", losses.data.item(),
            " lr:", lr
        )
    print(
        "Epoch:", epoch,
        " losses:", total_loss / n,
        " lr:", lr
    )
    return total_loss / n, batch_iter, scaler


def Train(args,
          scaler,
          model,
          criterion,
          optimizer,
          epoch,
          batch_iter,
          batch_size
          ):
    """Traing with the batch iter for get the metric
    """

    acc = 0
    n = 0
    for batch_idx, (hsi, lidar, tr_labels, hsi_pca) in enumerate(train_loader):
        n = n + 1
        hsi = hsi.to(device)
        hsi = hsi[:, 0, :, :, :]
        hsi_pca = hsi_pca.to(device)
        lidar = lidar.to(device)
        tr_labels = tr_labels.to(device)
        # TODO: add the layer-wise lr decay
        if args.cosine:
            # cosine learning rate
            lr = cosine_learning_rate(
                args, epoch, batch_iter, optimizer, batch_size
            )
        else:
            # step learning rate
            lr = step_learning_rate(
                args, epoch, batch_iter, optimizer, batch_size
            )
            # forward

        outputs, _ = model(hsi, lidar, hsi_pca)
        losses = criterion(outputs, tr_labels)

        optimizer.zero_grad()

        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_acc, _ = accuracy(outputs, tr_labels)

        acc = acc + batch_acc[0]

        batch_iter += 1

        print(
            "Epoch:", epoch,
            " batch_idx:", batch_idx,
            " batch_iter:", batch_iter,
            " batch_acc:", batch_acc[0],
            " lr:", lr
        )
    print(
        "Epoch:", epoch,
        " acc:", acc / n,
        " lr:", lr
    )
    return acc / n, batch_iter, scaler


def val(
        args,
        model
):
    """Validation and get the metric
    """
    batch_acc_list = []
    count = 0
    with torch.no_grad():
        for batch_idx, (hsi, lidar, tr_labels, hsi_pca) in enumerate(test_loader):

            hsi = hsi.to(device)
            hsi = hsi[:, 0, :, :, :]
            lidar = lidar.to(device)
            hsi_pca = hsi_pca.to(device)
            tr_labels = tr_labels.to(device)

            outputs, _ = model(hsi, lidar, hsi_pca)

            batch_accuracy, _ = accuracy(outputs, tr_labels)

            batch_acc_list.append(batch_accuracy[0])

            if count == 0:
                y_pred_test = np.argmax(outputs.detach().cpu().numpy(), axis=1)
                gty = tr_labels.detach().cpu().numpy()
                count = 1
            else:
                y_pred_test = np.concatenate((y_pred_test, np.argmax(outputs.detach().cpu().numpy(), axis=1)))  #
                gty = np.concatenate((gty, tr_labels.detach().cpu().numpy()))

    OA2, AA_mean2, Kappa2, AA2 = output_metric(gty, y_pred_test)
    classification = classification_report(gty, y_pred_test, digits=4)
    print(classification)
    print("OA2=", OA2)
    print("AA_mean2=", AA_mean2)
    print("Kappa2=", Kappa2)
    print("AA2=", AA2)
    epoch_acc = np.mean(batch_acc_list)

    print("Epoch_mean_accuracy:" % epoch_acc)

    return epoch_acc


def step_learning_rate(args, epoch, batch_iter, optimizer, train_batch):
    total_epochs = args.epoch
    warm_epochs = args.warmup_epochs
    if epoch <= warm_epochs:
        lr_adj = (batch_iter + 1) / (warm_epochs * train_batch)
        # lr_adj = 1.
    elif epoch < int(0.6 * total_epochs):
        lr_adj = 1.
    elif epoch < int(0.8 * total_epochs):
        lr_adj = 1e-1
    elif epoch < int(1 * total_epochs):
        lr_adj = 1e-2
    else:
        lr_adj = 1e-3

    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr * lr_adj
    return args.lr * lr_adj


def cosine_learning_rate(args, epoch, batch_iter, optimizer, train_batch):
    """Cosine Learning rate
    """
    total_epochs = args.max_epochs
    warm_epochs = args.warmup_epochs
    if epoch <= warm_epochs:
        lr_adj = (batch_iter + 1) / (warm_epochs * train_batch) + 1e-6
    else:
        lr_adj = 1 / 2 * (1 + math.cos(batch_iter * math.pi /
                                       ((total_epochs - warm_epochs) * train_batch)))

    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr * lr_adj
    return args.lr * lr_adj


def translate_state_dict(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if 'module' in key:
            new_state_dict[key[7:]] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(k=maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        crr = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            acc = correct_k.mul_(1 / batch_size).item()
            res.append(acc)  # unit: percentage (%)
            crr.append(correct_k)
        return res, crr


if args.is_pretrain == 1:
    model = MAE(
        channel_number=args.channel_num,
        img_size=args.crop_size,
        patch_size=args.patch_size,
        encoder_dim=args.dim,
        encoder_depth=args.depth,
        encoder_heads=args.head,
        decoder_dim=args.dim,
        decoder_depth=args.depth,
        decoder_heads=args.head,
        mask_ratio=args.mask_ratio,
        args=args
    )
else:
    model = MAEFinetune(
        channel_number=args.channel_num,
        img_size=args.crop_size,
        patch_size=args.patch_size,
        embed_dim=args.dim,
        depth=args.depth,
        num_heads=args.head,
        num_classes=args.num_classes,
        args=args
    )

device = torch.device(args.device if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("using " + args.device + " as device")
else:
    print("using cpu as device")
model.to(device)



if __name__ == '__main__':
    # 创建 trainloader 和 testloader
    total_loss = 0
    max_acc = 0

    model = model.to(device)
    model.cuda(device=device)
    optimizer = Optimizer(args.optimizer_name)(
        param=model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        finetune=args.finetune
    )
    scaler = torch.cuda.amp.GradScaler()
    if args.is_pretrain == 1:
        criterion = MSELoss(device=device)
        print("Pretraining!!--------")
        min_loss = 1e8
        batch_iter = 0
        for epoch in range(args.epoch):
            model.train()
            n = 0
            loss, batch_iter, scaler = Pretrain(args, scaler, model, criterion, optimizer, epoch, batch_iter,
                                                args.batch_size)

            if loss < min_loss:

                state_dict = translate_state_dict(model.state_dict())
                state_dict = {
                    'epoch': epoch,
                    'state_dict': state_dict,
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(
                    state_dict,
                    'model/' + 'pretrain_' + args.dataset + '_num' + str(args.pretrain_num) + '_crop_size' + str(
                        args.crop_size) + '_mask_ratio_' + str(args.mask_ratio) \
                    + '_DDH_' + str(args.depth) + str(args.dim) + str(args.head) + '_epoch_' + str(
                        epoch) + '_loss_' + str(loss) + '.pth'
                )
                min_loss = loss

    if args.is_train == 1:
        criterion = nn.CrossEntropyLoss()
        batch_iter = 0
        for epoch in range(args.epoch):
            model.train()
            n = 0
            loss, batch_iter, scaler = Train(args, scaler, model, criterion, optimizer, epoch, batch_iter,
                                             args.batch_size)
            if epoch % args.test_interval == 0:
                # For some datasets (such as Berlin), the test set is too large and the test speed is slow,
                # so it is recommended to split the validation set with a small sample size
                model.eval()
                acc1 = val(args, model)
                print("epoch:", epoch, "acc:", acc1)
                if acc1 > max_acc:
                    state_dict = translate_state_dict(model.state_dict())
                    state_dict = {
                        'epoch': epoch,
                        'state_dict': state_dict,
                        'optimizer': optimizer.state_dict(),
                    }
                    torch.save(state_dict, 'model/' + 'train_' + args.dataset + '_num' + str(
                        args.pretrain_num) + '_crop_size' + str(args.crop_size) + '_mask_ratio_' + str(args.mask_ratio) \
                               + '_DDH_' + str(args.depth) + str(args.dim) + str(args.head) + '_epoch_' + str(
                        epoch) + '_acc_' + str(acc1) + '.pth')

                    max_acc = acc1

    if args.is_test == 1:
        model_path = 'model/' + 'train_' + args.dataset + '_num' + str(
                        args.pretrain_num) + '_crop_size' + str(args.crop_size) + '_mask_ratio_' + str(args.mask_ratio) \
                               + '_DDH_' + str(args.depth) + str(args.dim) + str(args.head) + '_epoch_' + str(
                        epoch) + '.pth'
        checkpoint = torch.load(model_path, map_location="cpu")
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        acc1 = val(args, model)
