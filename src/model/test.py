import os
from args import args
from utils import make_logInfo
# 执行外部sh文件
# print(os.curdir)
# os.system('sh src/run.sh > run_log.txt')
# print(args.loginfo)
# args.logInfo = make_logInfo(fileName='', filePath = os.getcwd())
# with open(os.path.join(args.logInfo['logPath'], '{}args.txt'.format(args.logInfo['hour'])), 'w') as f:
#     f.write(str(args))

os.system('conda activate openne && python src/model/utils.py')