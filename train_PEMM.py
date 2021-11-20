import argparse
import torch
import time
import logging
import os
from model import SCEModel, ResNet34, ResNet50, RBFC,RBFCN
from dataset import DatasetGenerator
from tqdm import tqdm
from utils.utils import AverageMeter, accuracy, count_parameters_in_MB
from train_util import TrainUtil
from loss import SCELoss,SMSELoss
import torch.nn as nn
import torch.nn.functional as F

# ArgParse
parser = argparse.ArgumentParser(description='SCE Loss')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--l2_reg', type=float, default=5e-4)
parser.add_argument('--grad_bound', type=float, default=5.0)
parser.add_argument('--train_log_every', type=int, default=100)
parser.add_argument('--resume', action='store_true', default=False)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--data_path', default='../../datasets', type=str)
parser.add_argument('--checkpoint_path', default='checkpoints', type=str)
parser.add_argument('--data_nums_workers', type=int, default=8)
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--nr', type=float, default=0.4, help='noise_rate')
parser.add_argument('--loss', type=str, default='SCE', help='SCE, CE')
parser.add_argument('--alpha', type=float, default=0, help='alpha scale')
parser.add_argument('--beta', type=float, default=1.0, help='beta scale')
parser.add_argument('--version', type=str, default='SCE0.0', help='Version')
parser.add_argument('--dataset_type', choices=['cifar10', 'cifar100','1M'], type=str, default='cifar10')
parser.add_argument('--asym', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=123)

args = parser.parse_args()
GLOBAL_STEP, EVAL_STEP, EVAL_BEST_ACC, EVAL_BEST_ACC_TOP5 = 0, 0, 0, 0
cell_arc = None
device = torch.device("cuda:2")

def pe(x,s=3,t=2):
    p = 1/torch.pow(x,s)-1/torch.pow(x,t)
    return p

class PTMLoss(nn.Module):
    def forward(self, x):
        o = torch.zeros(x.shape).to(device)
        loss0 = torch.mean(torch.clamp(pe(1*F.pairwise_distance(x,o)+0),-1.5,1000))
        loss =  torch.mean(torch.clamp(pe(1*F.pdist(x)+0),-1.5,1000))   
        return loss0+loss
    
    
class GALoss(nn.Module):
    def __init__(self, num_classes=10):
        super(GALoss, self).__init__()     
        self.num_classes = num_classes
        self.mse = nn.MSELoss()
        self.softmax_func=nn.Softmax(dim=1)
    def forward(self,model,x,pred,y):    
        s0=self.softmax_func(pred)
        xd = torch.matmul(s0,model.fc2.centers)
        loss = self.mse(x,xd)
        return loss


if args.dataset_type == 'cifar100':
    num_classes = 100
if args.dataset_type == 'cifar10':
    num_classes = 10
if args.dataset_type == '1M':
    num_classes = 14

# define loss functions
ptm_loss = PTMLoss()
cls_loss = nn.CrossEntropyLoss()
sce_loss = SCELoss(alpha=args.alpha, beta=args.beta, num_classes=num_classes)
ga_loss = GALoss()


def criterion_PEMM(model,x,pred,y,epoch):
    if epoch <0:
        loss1 = cls_loss(pred,y)
    else:
        loss1 = sce_loss(pred,y)
    loss2 = 2*ptm_loss(model.fc2.centers)
    loss3 = ga_loss(model,x,pred,y)
    return loss1+loss2+loss3


def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""
    formatter = logging.Formatter('%(asctime)s %(message)s')
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


def adjust_weight_decay(model, l2_value):
    conv, fc = [], []
    for name, param in model.named_parameters():
        print(name)
        if not param.requires_grad:
            # frozen weights
            continue
        if 'module.fc1' in name:
            fc.append(param)
        else:
            conv.append(param)
    params = [{'params': conv, 'weight_decay': l2_value}, {'params': fc, 'weight_decay': 0.01}]
    print(fc)
    return params


if not os.path.exists('logs'):
    os.makedirs('logs')
log_file_name = os.path.join('logs', args.version + '.log')
logger = setup_logger(name=args.version, log_file=log_file_name)
for arg in vars(args):
    logger.info("%s: %s" % (arg, getattr(args, arg)))

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    logger.info("Using CUDA!")
else:
    device = torch.device('cpu')


def log_display(epoch, global_step, time_elapse, **kwargs):
    display = 'epoch=' + str(epoch) + \
              '\tglobal_step=' + str(global_step)
    for key, value in kwargs.items():
        display += '\t' + str(key) + '=%.5f' % value
    display += '\ttime=%.2fit/s' % (1. / time_elapse)
    return display


def model_eval(epoch, fixed_cnn, data_loader):
    global EVAL_STEP
    fixed_cnn.eval()
    valid_loss_meters = AverageMeter()
    valid_acc_meters = AverageMeter()
    valid_acc5_meters = AverageMeter()
    ce_loss = torch.nn.CrossEntropyLoss()

    for images, labels in data_loader["test_dataset"]:
        start = time.time()
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            features,pred = fixed_cnn(images)
            loss = ce_loss(pred, labels)
            acc, acc5 = accuracy(pred, labels, topk=(1, 5))

        valid_loss_meters.update(loss.item())
        valid_acc_meters.update(acc.item())
        valid_acc5_meters.update(acc5.item())
        end = time.time()

        EVAL_STEP += 1
    return valid_acc_meters.avg, valid_acc5_meters.avg


def train_fixed(starting_epoch, data_loader, fixed_cnn, criterion, fixed_cnn_optmizer, fixed_cnn_scheduler, utilHelper):
    global GLOBAL_STEP, reduction_arc, cell_arc, EVAL_BEST_ACC, EVAL_STEP, EVAL_BEST_ACC_TOP5
    acc = 0.0
    for epoch in range(starting_epoch, args.epoch):
        logger.info("=" * 20 + "Training" + "=" * 20)
        print("Epoch:",epoch+1)
        fixed_cnn.train()
        train_loss_meters = AverageMeter()
        train_acc_meters = AverageMeter()
        train_acc5_meters = AverageMeter()

        for images, labels in tqdm(data_loader["train_dataset"]):
            images, labels = images.to(device), labels.to(device)
            start = time.time()
            fixed_cnn.zero_grad()
            fixed_cnn_optmizer.zero_grad()
            features,pred = fixed_cnn(images)
            try:
                loss = criterion(pred, labels)
            except:
                loss = criterion(fixed_cnn,features,pred, labels,epoch)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(fixed_cnn.parameters(), args.grad_bound)
            fixed_cnn_optmizer.step()

            acc, acc5 = accuracy(pred, labels, topk=(1, 5))
            acc_sum = torch.sum((torch.max(pred, 1)[1] == labels).type(torch.float))
            total = pred.shape[0]
            acc = acc_sum / total

            train_loss_meters.update(loss.item())
            train_acc_meters.update(acc.item())
            train_acc5_meters.update(acc5.item())

            end = time.time()

            GLOBAL_STEP += 1
            if GLOBAL_STEP % args.train_log_every == 0:
                lr = fixed_cnn_optmizer.param_groups[0]['lr']
        fixed_cnn_scheduler.step()
        curr_acc, curr_acc5 = model_eval(epoch, fixed_cnn, data_loader)
        payload = '=' * 10 + '\n'
        payload = payload + ("curr_acc: %.4f\n best_acc: %.4f\n" % (curr_acc, EVAL_BEST_ACC))
        payload = payload + ("curr_acc_top5: %.4f\n best_acc_top5: %.4f\n" % (curr_acc5, EVAL_BEST_ACC_TOP5))
        if curr_acc > EVAL_BEST_ACC:
            torch.save(fixed_cnn, './model/'+args.checkpoint_path)
            print("Model Saved!\n")
            print("BEST_ACC\t%.4f" % EVAL_BEST_ACC)
        EVAL_BEST_ACC = max(curr_acc, EVAL_BEST_ACC)
        EVAL_BEST_ACC_TOP5 = max(curr_acc5, EVAL_BEST_ACC_TOP5)
    print("FINAL_ACC\t%.4f" % EVAL_BEST_ACC)
    return


def train():
    global GLOBAL_STEP, reduction_arc, cell_arc
    # Dataset
    if args.dataset_type == 'cifar100' or args.dataset_type == 'cifar10':
        dataset = DatasetGenerator(batchSize=args.batch_size,
                                   dataPath=args.data_path,
                                   numOfWorkers=args.data_nums_workers,
                                   noise_rate=args.nr,
                                   asym=args.asym,
                                   seed=args.seed,
                                   dataset_type=args.dataset_type)
        dataLoader = dataset.getDataLoader()
        
    if args.dataset_type == '1M':
        dataset = DatasetGenerator_1M(batchSize=args.batch_size,
                                   dataPath=args.data_path,
                                   numOfWorkers=args.data_nums_workers,
                                   seed=args.seed)
        dataLoader = dataset.getDataLoader()

    if args.dataset_type == 'cifar100':
        num_classes = 100
        fixed_cnn = ResNet34(num_classes=num_classes)
    if args.dataset_type == 'cifar10':
        num_classes = 10
        fixed_cnn = SCEModel()
    if args.dataset_type == '1M':
        num_classes = 14
        fixed_cnn = ResNet50(num_classes=num_classes)

    if args.loss == 'SCE':
        criterion = SCELoss(alpha=args.alpha, beta=args.beta, num_classes=num_classes)
    elif args.loss == 'CE':
        criterion = torch.nn.CrossEntropyLoss()
    else:
        num_ftrs = fixed_cnn.fc2.in_features
        fixed_cnn.fc2 = RBFC(num_ftrs,num_classes)
        criterion = criterion_PEMM
        

#     logger.info(criterion.__class__.__name__)
#     logger.info("Number of Trainable Parameters %.4f" % count_parameters_in_MB(fixed_cnn))
    fixed_cnn = fixed_cnn
    fixed_cnn.to(device)

    fixed_cnn_optmizer = torch.optim.SGD(params=adjust_weight_decay(fixed_cnn, args.l2_reg),
                                         lr=args.lr,
                                         momentum=0.9,
                                         nesterov=True)

    fixed_cnn_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(fixed_cnn_optmizer, 10, eta_min=0, last_epoch=-1)
    utilHelper = TrainUtil(checkpoint_path=args.checkpoint_path, version=args.version)
    starting_epoch = 0
    train_fixed(starting_epoch, dataLoader, fixed_cnn, criterion, fixed_cnn_optmizer, fixed_cnn_scheduler, utilHelper)


if __name__ == '__main__':
    train()
