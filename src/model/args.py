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
parser.add_argument('-sp', '--saved_path', type=str,
                    help='Path to save training results', default='result')
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

#TODO: Add more arguments for model
parser.add_argument('-ah', '--ae_hidden', nargs='+', type=int, 
                    help='Set the AutoEncoder hidden layer. eg: 512 64')
                
# parser.add_argument('-hf', '--hidden_feats', default=64, type=int,
#                     help='The dimension of hidden tensor in the model')
# parser.add_argument('-he', '--num_heads', default=5, type=int,
#                     help='Number of attention heads the model has')
# parser.add_argument('-dp', '--dropout', default=0.0, type=float,
#                     help='The rate of dropout layer')

args = parser.parse_args()

args.logInfo = make_logInfo(fileName='', filePath = os.getcwd())
args.saved_path = args.saved_path + '_' + str(args.seed)
args.device = 'cuda:{}'.format(args.device_id) if torch.cuda.is_available() else 'cpu'

# save args in to file txt
with open(os.path.join(args.logInfo['logPath'], '{}args.txt'.format(args.logInfo['hour'])), 'w') as f:
    f.write(str(args))
print(args)


