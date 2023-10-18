import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import utils_avatar as ua
import os
from datetime import datetime
import json
import argparse
import time
import random
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, required=True, help='Specfiy the task (mnist/cifar10/gtsrb/imageNet).')
parser.add_argument('--to_file', action='store_true', help='If set, dump the result to file.')
parser.add_argument('--weight_limit', action='store_true', help='If set, limit the weights of the model parameters.')

if __name__ == '__main__':
    args = parser.parse_args()

    GPU = True
    TARGET_PROP = 1.0
    np.random.seed(0)
    torch.manual_seed(0)
    if GPU:
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    BATCH_SIZE, N_EPOCH, trainset, testset, is_binary, need_pad, Model, input_size, class_num = ua.load_dataset_setting(
        args.task)  
    tot_num = len(trainset) 
    target_indices = np.random.choice(tot_num, int(tot_num * TARGET_PROP))
    print("Data indices owned by the attacker:", target_indices)
    target_set = torch.utils.data.Subset(trainset, target_indices)

    SAVE_PREFIX = './%s' % args.task
    if not os.path.isdir(SAVE_PREFIX):
        os.mkdir(SAVE_PREFIX)
    if not os.path.isdir(SAVE_PREFIX + '/models'):
        os.mkdir(SAVE_PREFIX + '/models')
    if not os.path.isdir(SAVE_PREFIX + '/universal_triggers'):
        os.mkdir(SAVE_PREFIX+'/universal_triggers')

    targets = list(range(class_num))
    for i in targets:
        t1 = time.time()
        model = Model(gpu=GPU)
        cleantrainloader = torch.utils.data.DataLoader(target_set, batch_size=BATCH_SIZE, shuffle=True)
        cleantestloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)

        model.load_state_dict(torch.load('%s/models/clean_0.model'%args.task)) # poison the clean model

        universal_triggers, fission, p1_size, loc1, p2_size, loc2, target_y, inject_p = ua.random_avatar_setting(i,args.task, model, input_size, class_num, cleantestloader)

        torch.save(universal_triggers,'%s/universal_triggers/%i.pth'%(SAVE_PREFIX,i))

        trainset_mal = ua.BackdoorDataset(target_set, ua.trigger_gen_func, inject_p, universal_triggers=universal_triggers,fission=fission,class_num=class_num,backdoor_label=target_y)
        trainloader = torch.utils.data.DataLoader(trainset_mal, batch_size=BATCH_SIZE, shuffle=True)
        testset_mal = ua.BackdoorDataset(testset, ua.trigger_gen_func, inject_p, mal_only=True,universal_triggers=universal_triggers,fission=fission,class_num=class_num,backdoor_label=target_y)  # malicious test
        testloader_benign = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)
        testloader_mal = torch.utils.data.DataLoader(testset_mal, batch_size=BATCH_SIZE, shuffle=True)

        ua.train_model(args.task, model, trainloader,testloader_mal,N_EPOCH, is_binary, args.weight_limit, verbose=True)

        save_path = SAVE_PREFIX + '/models/avatar_label_%d.model' % target_y

        torch.save(model.state_dict(), save_path)
        acc = ua.eval_model(model, testloader_benign,is_binary)
        acc_mal = ua.eval_model(model, testloader_mal,is_binary)

        t2 = time.time()
        print('ReTraining model cost %.3fs.'%(t2-t1))

        output_path = SAVE_PREFIX +'/models/avatar_output_target.txt'

        if args.to_file:
            with open(output_path, 'a+') as f:
                f.write("Model: %d, Acc %.4f, Acc on backdoor %.4f, saved to %s @ %s\n" % (i, acc, acc_mal, save_path, datetime.now()))
                if fission == True:
                    f.write("fission: True; p1 size: %s; loc1: %s; p2 size: %s; loc2:%s; target_y: %d; inject p: %.3f\n" % (str(p1_size), str(loc1), str(p2_size),str(loc2), target_y, inject_p))
                else:
                    f.write("fission: False; p1 size: %s; loc1: %s; target_y: %d; inject p: %.3f\n" % (str(p1_size), str(loc1), target_y, inject_p))

        print("Model: %d, Acc %.4f, Acc on backdoor %.4f, saved to %s @ %s\n" % (
            i, acc, acc_mal, save_path, datetime.now()))
        if fission == True:
            print("fission: True; p1 size: %s; loc1: %s; p2 size: %s; loc2:%s; target_y: %d; inject p: %.3f\n" % (str(p1_size), str(loc1), str(p2_size), str(loc2), target_y, inject_p))
        else:
            print("fission: False; p1 size: %s; loc1: %s; target_y: %d; inject p: %.3f\n" % (str(p1_size), str(loc1), target_y, inject_p))