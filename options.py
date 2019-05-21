
import scripts.models as models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
    
class Options():
    """docstring for Options"""
    def __init__(self):
        pass

    def init(self, parser):        
        # Model structure
        parser.add_argument('--arch', '-a', metavar='ARCH', default='dhn',
                            choices=model_names,
                            help='model architecture: ' +
                                ' | '.join(model_names) +
                                ' (default: resnet18)')

        parser.add_argument('--darch', '-w', metavar='ARCH', default='patchgan',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: resnet18)')
        
        parser.add_argument('--machine', '-m', metavar='NACHINE', default='basic')
        # Training strategy
        parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                            help='number of data loading workers (default: 4)')
        parser.add_argument('--epochs', default=30, type=int, metavar='N',
                            help='number of total epochs to run')
        parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                            help='manual epoch number (useful on restarts)')
        parser.add_argument('--train-batch', default=64, type=int, metavar='N',
                            help='train batchsize')
        parser.add_argument('--test-batch', default=6, type=int, metavar='N',
                            help='test batchsize')
        parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                            metavar='LR', help='initial learning rate')
        parser.add_argument('--momentum', default=0, type=float, metavar='M',
                            help='momentum')
        parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                            metavar='W', help='weight decay (default: 0)')
        parser.add_argument('--schedule', type=int, nargs='+', default=[5, 10],
                            help='Decrease learning rate at these epochs.')
        parser.add_argument('--gamma', type=float, default=0.1,
                            help='LR is multiplied by gamma on schedule.')
        # Data processing
        parser.add_argument('-f', '--flip', dest='flip', action='store_true',
                            help='flip the input during validation')
        parser.add_argument('--sigma', type=float, default=1,
                            help='Groundtruth Gaussian sigma.')
        parser.add_argument('--alpha', type=float, default=0.5,
                            help='Groundtruth Gaussian sigma.')
        parser.add_argument('--sigma-decay', type=float, default=0,
                            help='Sigma decay rate for each epoch.')
        parser.add_argument('--label-type', metavar='LABELTYPE', default='Gaussian',
                            choices=['Gaussian', 'Cauchy'],
                            help='Labelmap dist type: (default=Gaussian)')
        # Miscs
        parser.add_argument('--base-dir', default='/home/mb55411/dataset/splicing/NC2016_Test/', type=str, metavar='PATH',help='path to save checkpoint (default: checkpoint)')
        parser.add_argument('--ground-truth-dir', default='/home/mb55411/dataset/splicing/NC2016_Test/', type=str, metavar='PATH',help='path to save checkpoint (default: checkpoint)')
        parser.add_argument('--mask-path', default='/home/mb55411/dataset/splicing/NC2016_Test/', type=str, metavar='PATH',help='path to save checkpoint (default: checkpoint)')
        parser.add_argument('--data', default='train', type=str, metavar='PATH',
                            help='path to save checkpoint (default: checkpoint)')
        parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                            help='path to save checkpoint (default: checkpoint)')
        parser.add_argument('--resume', default='', type=str, metavar='PATH',
                            help='path to latest checkpoint (default: none)')
        parser.add_argument('--finetune', default='', type=str, metavar='PATH',
                            help='path to latest checkpoint (default: none)')

        parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                            help='evaluate model on validation set')
        parser.add_argument('--attention-loss-weight', default=1e10, type=float,
                            help='preception loss')
        parser.add_argument('--loss-pixel', default=100, type=float,
                            help='preception loss')
        parser.add_argument('--loss-attention', default=100, type=float,
                            help='preception loss')
        parser.add_argument('--resize-and-crop', default='resize',
                            help='Labelmap dist type: (default=Gaussian)')
        parser.add_argument('-da', '--data-augumentation', default=False, type=bool,
                            help='preception loss')
        parser.add_argument('-d', '--debug', dest='debug', action='store_true',
                            help='show intermediate results')
        parser.add_argument('--input-size', default=512, type=int, metavar='N',
                            help='train batchsize')
        parser.add_argument('--requires-grad', default=False, type=bool,
                            help='train batchsize')
        parser.add_argument('--limited-dataset', default=0, type=int, metavar='N')
        parser.add_argument('--gpu',default=True,type=bool)
        parser.add_argument('--comparegan',default=False,type=bool)
        parser.add_argument('--multicomapre',default=False,type=bool)
        parser.add_argument('--freeze',default=False,type=bool)
        parser.add_argument('--semi',default=False,type=bool)
        parser.add_argument('--gradient-loss',default=False,type=bool)
        parser.add_argument('--task', default='harmonization',type=str, help='train batchsize')
        parser.add_argument('--mask-loss-type', default='pixelwise',type=str, help='train batchsize')
        parser.add_argument('--norm-type', default='none',type=str, help='train batchsize')
        parser.add_argument('--random-mask', default=False,type=bool)
        parser.add_argument('--val', default=False,type=bool)
        return parser