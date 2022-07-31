import argparse
import os
from utils import make_logInfo
import torch
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# General Arguments
parser.add_argument('-id', '--device_id', default=0, type=str,
                    help='Set the device (GPU ids).')
parser.add_argument('-sk', '--skip_load_data', default=False, type=bool,
                    help='if True, skip loading data.')
parser.add_argument('-skopenne', '--skip_openne', default=False, type=bool,
                    help='if True, skip OpenNE.')
parser.add_argument('-skae', '--skip_ae', default=False, type=bool,
                    help='if True, skip AutoEncoder.')
# parser.add_argument('-da', '--dataset', type=str,
#                     help='Set the data set for training.')
parser.add_argument('-savePath', '--saved_path', type=str,
                    help='Path to save training results', default='')
parser.add_argument('-se', '--seed', default=42, type=int,
                    help='Global random seed')
# Training Arguments
parser.add_argument('-fo', '--nfold', default=10, type=int,
                    help='The number of k in K-folds Validation')
parser.add_argument('-ep', '--epoch', default=200, type=int,
                    help='Number of epochs for training')
parser.add_argument('-lr', '--learning_rate', default=0.002, type=float,
                    help='learning rate to use')
parser.add_argument('-bs', '--batch_size', default=512, type=int,
                    help='batch size for AutoEncoder to use')   
# parser.add_argument('-wd', '--weight_decay', default=0.0, type=float,
#                     help='weight decay to use')
# parser.add_argument('-pa', '--patience', default=100, type=int,
#                     help='Early Stopping argument')

# Model Arguments
parser.add_argument('-ah', '--ae_hidden', nargs='+', type=int, 
                    help='Set the AutoEncoder hidden layer. eg: 512 64')
parser.add_argument('-om', '--openne_method', nargs='+', type=str, default='LINE HOPE SNDE',
                    help='Set the AutoEncoder hidden layer. eg: LINE HOPE SNDE')
 
# analysis Arguments
parser.add_argument('-md', '--mode', type=str,
                    help='running mode of main. mode: train | findNew', default='train')      
parser.add_argument('-maskType', '--maskType', type=str,
                    help='running mode of main. mode: gene | dis', default='gene')  
parser.add_argument('-ablationType', '--ablationType', type=str,
                    help='running mode of main. mode: ae | openne') 
parser.add_argument('-openneDim', '--openneDim', type=str,
                    help='running mode of main. mode: ae | openne') 


args = parser.parse_args()

args.logInfo = make_logInfo(fileName='', filePath = os.getcwd(),savePath=args.saved_path)
# 用于方便后续保存文件更清晰，以随机种子命名和保存文件
# args.logInfo['hour'] = str(args.seed)+'_'
# args.logInfo['date'] = args.saved_path # 将文件保存在log/save_path目录下

# args.saved_path = args.saved_path + '_' + str(args.seed)
args.device = 'cuda:{}'.format(args.device_id) if torch.cuda.is_available() else 'cpu'

# save args in to file txt
with open(os.path.join(args.logInfo['logPath'], '{}args.txt'.format(args.logInfo['hour'])), 'w') as f:
    f.write(str(args))
print(args)

# TODO: 将买一次运行的结果，每一折的结果处理保存成一个csv文件
