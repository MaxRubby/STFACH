import argparse
import copy
import random
import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import KernelDensity

import pandas as pd
import os
import torch
import torch.nn as nn
import datetime
import time
import matplotlib.pyplot as plt
from torchinfo import summary
import yaml
import json
import sys
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


sys.path.append("..")
from lib.utils import (
    init_seed,
    print_log,
    set_cpu_num,
    CustomJSONEncoder,
)
from lib.metrics import RMSE_MAE_MAPE
from lib.data_prepare import generate_data_point
from model.STFACH import STFACH
from model.HisPatternLearner import HisPattern

def make_dir(name):
    if (os.path.exists(name)):
        print('has  save path')
    else:
        os.makedirs(name)

@torch.no_grad()
def eval_model(model, valset_loader, criterion, epoch, scheduler ,  shift_weight = None):
    model.eval()
    batch_loss_list = []
    valid_per_epoch = len(valset_loader)
    with torch.no_grad():
        for batch_idx,batch in enumerate(valset_loader):
            batch = tuple(b.to(DEVICE) for b in batch)
            x_batch,y_batch,  *coeff_batch= batch #BTND(3)
            shifts = x_batch[...,-1:]
            out_batch =  model(ori_x=x_batch, coeffs = coeff_batch)
            out_batch = SCALER.inverse_transform(out_batch)
            shifts = SCALER.inverse_transform(shifts)
            loss= criterion(out_batch, y_batch)
            batch_loss_list.append(loss.item())
            
             
    epoch_loss = np.mean(batch_loss_list)
    scheduler.step(metrics=epoch_loss)
    
    return epoch_loss


@torch.no_grad()
def predict(model, testset_loader,   epoch = 0 ):
    model.eval()
    y = []
    out = []
    for batch_idx,batch in enumerate(testset_loader):

        batch = tuple(b.to(DEVICE) for b in batch)
        x_batch,y_batch,  *coeff_batch= batch #BTND(3)
        shifts = x_batch[..., -1:]

        out_batch =  model(ori_x=x_batch,   coeffs = coeff_batch)
        out_batch = SCALER.inverse_transform(out_batch)
        shifts = SCALER.inverse_transform(shifts)

        out_batch = out_batch.cpu().numpy()
        y_batch = y_batch.cpu().numpy()
        out.append(out_batch)
        y.append(y_batch)
    out = np.vstack(out) # (samples, out_steps, num_nodes)
    y = np.vstack(y)
    if out.shape[1] > 1:
        out = out.squeeze()
        y = y.squeeze()
    return y, out

  
def train_one_epoch(
    model,  trainset_loader, optimizer, scheduler, criterion,  clip_grad, epoch, mixup_num, log=None,w=None, device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),   shift_weight = None
):
    global cfg , iter_count , target_length       
 

    model.train()

     
    batch_loss_list = []
    x_batch_accumulate = torch.tensor([], device=device)
    y_batch_accumulate = torch.tensor([], device=device)
    shifts_accumulate = torch.tensor([], device=device)
    

    for batch_idx,batch in enumerate(trainset_loader):
        
        lambd = np.random.beta(mix_alpha, mix_alpha) 

        if cfg["use_cl"]: 
            if (
                iter_count % cfg["cl_step_size"] == 0
                and target_length < cfg["out_steps"]
            ):
                target_length += 1
                print_log(f"CL target length = {target_length}", log=log)
            iter_count += 1

        batch = tuple(b.to(DEVICE) for b in batch)
        x_batch,y_batch,  *coeff_batch= batch #BTND(3)
        
        shifts =x_batch[..., -1:]

        if batch_idx > 1 and batch_idx % mixup_num == 0 :
            id_1 = np.arange(x_batch.shape[0])  
            x_1 = x_batch[id_1]
            y_1 = y_batch[id_1]
            coeffs_1 = tuple(t[id_1] for t in coeff_batch)  
            id_2 = torch.randperm(x_batch.shape[0])
            x_2 = x_batch[id_2]
            y_2 = y_batch[id_2]
            coeffs_2 = tuple(t[id_2] for t in coeff_batch)  

            mixup_Y = y_1 * lambd + y_2 * (1 - lambd)
            mixup_X = x_1 * lambd + x_2 * (1 - lambd)
            mixup_coeffs = tuple(t1 * lambd + t2 * (1 - lambd) for t1, t2 in zip(coeffs_1, coeffs_2))
            
            out_batch =  model(ori_x=mixup_X, coeffs = mixup_coeffs)  
             
        else:
            out_batch =  model(ori_x=x_batch, coeffs = coeff_batch)

        if  batch_idx % 50 == 0 and args.speed:
            model_time_s = time.time()       

        if  batch_idx % 50 == 0 and args.speed:
            model_time_e = time.time()     

        out_batch = SCALER.inverse_transform(out_batch)
        shifts = SCALER.inverse_transform(shifts)

        x_batch_accumulate = torch.cat((x_batch_accumulate, x_batch[:,:,:,:1].mean(axis = -2)), dim=0)
        y_batch_accumulate = torch.cat((y_batch_accumulate, y_batch[:,:,:,:1].mean(axis = -2)), dim=0)
        shifts_accumulate = torch.cat((shifts_accumulate, shifts[:,:,:,:1].mean(axis = -2)), dim=0)

         
        if batch_idx > 1 and batch_idx % mixup_num == 0 :
            loss = criterion(out_batch[:, : target_length, ...], mixup_Y[:, : target_length, ...])
        else:
            loss = criterion(out_batch[:, : target_length, ...], y_batch[:, : target_length, ...])
            
        
        batch_loss_list.append(loss.item())
        train_per_epoch = len(trainset_loader)
        optimizer.zero_grad()
        if  batch_idx % 50 == 0 and args.speed:
            lb_s = time.time()   
  
        loss.backward()

        if  batch_idx % 50 == 0 and args.speed:
            lb_e = time.time()     
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        
        if  batch_idx % 50 == 0 and args.speed:
            opt_time_s = time.time()  
        optimizer.step()
        if  batch_idx % 50 == 0 and args.speed:
            opt_time_e = time.time() 
        if  batch_idx % 50 == 0:
            if args.speed:
                board_time_s =   time.time()
            if args.speed:           
                board_time_e = time.time()
            
            print_log(f"Train Epoch {epoch}: {batch_idx}/{len(trainset_loader)}  Loss: {loss.item()}  LR:{optimizer.param_groups[0]['lr']} ",log=log)
      
                    

    epoch_loss = np.mean(batch_loss_list)
    return epoch_loss 


 
def save_checkpoint( model,optimizer, epoch, min_var_loss, log = None):
    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'min_val_loss': min_var_loss
    }
    
    cache_path = log_dir + '/best_model_{}.pth'.format(epoch)
    torch.save(state, cache_path)
    print_log(f"Saving current best model to {cache_path}", log=log)

def train(
    model,
    trainset_loader,
    valset_loader,
    optimizer,
    scheduler,
    criterion,
    adj_mx,
    clip_grad=0,
    max_epochs=200,
    early_stop=10,
    verbose=1,
    plot=False,
    log=None,
    save=None,
    cache_path = None,
    w=None,
    his_opt = None,
    his_learner = None
):
    if cache_path is not None and args.cont > 0:
        model.load_state_dict(torch.load(cache_path)['state_dict'])
        optimizer.load_state_dict(torch.load(cache_path)['optimizer'])
        if torch.load(cache_path)['scheduler_state'] is not None :
            scheduler.load_state_dict(torch.load(cache_path)['scheduler_state'])
    model = model.to(DEVICE)
    if his_learner is not None:
        his_learner = his_learner.to(DEVICE)
    min_val_loss =  np.inf
    if args.cont > 0:
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
    wait = 0
    train_loss_list = []
    val_loss_list = []
    mixup_num = 5  
    load_best_his = True
    pretrain_hislearner = True

    if pretrain_hislearner is False:
        make_dir(log_dir + '/his_learn')
        saved_his_path = log_dir + '/his_learn/'+args.dataset +'_final_best_hisweight.pth'
        if load_best_his is True:
            best_his_state = torch.load(saved_his_path)['state_dict']

        else:
            if os.path.exists(saved_his_path) and load_best_his is True:
                best_his_state = torch.load(saved_his_path)['state_dict']
                his_learner.load_state_dict(best_his_state)
            his_epochs = 200
            min_his_var = np.inf
            his_criterion = nn.HuberLoss(delta=2.5)
            prefix="\hislearner"
            print_log("------------------------------his training begin------------------------------", log=his_log)
            for epoch in range(0, his_epochs):
                x_batch_accumulate = torch.tensor([], device=DEVICE)
                y_batch_accumulate = torch.tensor([], device=DEVICE)
                shifts_accumulate = torch.tensor([], device=DEVICE)
                his_loss_list = []
        ###############################training for his learner#######################################
                his_learner.train()
                for batch_idx, batch in enumerate(trainset_loader):  # 假设有一个 
                    batch = tuple(b.to(DEVICE) for b in batch)
                    x_batch,y_batch, shifts, *coeff_batch= batch #BTND(3)
                    weighted_shifts = his_learner(shifts)
                    weighted_shifts = SCALER.inverse_transform(weighted_shifts)
                    x_batch = x_batch.mean(axis = -2)
                    y_batch = y_batch.mean(axis = -2)
                    x_batch_accumulate = torch.cat((x_batch_accumulate, x_batch[:,:,:1]), dim=0) #BTD, N维度平均
                    y_batch_accumulate = torch.cat((y_batch_accumulate, y_batch[:,:,:1]), dim=0)
                    shifts_accumulate = torch.cat((shifts_accumulate, weighted_shifts[:,:,:]), dim=0)
                    history_loss = his_criterion(weighted_shifts, y_batch)
                    his_loss_list.append(history_loss.item())
                    his_opt.zero_grad()
                    history_loss.backward()
                    his_opt.step()
                his_train_loss = np.mean(his_loss_list)
                
        ###############################validating for his learner#######################################
                his_loss_list = []
                valid_per_epoch = len(valset_loader)
                his_learner.eval()
                with torch.no_grad():
                    for batch_idx,batch in enumerate(valset_loader):
                        batch = tuple(b.to(DEVICE) for b in batch)
                        x_batch,y_batch, shifts, *coeff_batch= batch #BTND(3)
                        weighted_shifts = his_learner(shifts)
                        x_batch = x_batch.mean(axis = -2)
                        y_batch = y_batch.mean(axis = -2)
                        weighted_shifts = SCALER.inverse_transform(weighted_shifts)
                        history_loss = his_criterion(weighted_shifts,  y_batch)
                        his_loss_list.append(history_loss.item())
                    his_val_loss = np.mean(his_loss_list)

                if (epoch) % verbose == 0:
                    print_log(
                        datetime.datetime.now(),
                        "Epoch",
                        epoch,
                        " \tHistrain Loss = %.5f" % his_train_loss, 
                        "Hisval Loss = %.5f" % his_val_loss,
                        log=his_log,
                    )
                if his_val_loss < min_his_var:
                    min_his_var = his_val_loss
                    best_his_state = his_learner.state_dict()

            for param in his_learner.parameters():
                param.requires_grad = False
        shift_weight = torch.sigmoid(best_his_state['weight_shift']) 
        np.save(f"finetune_his/{args.dataset}_final_best_hisweight.npy",shift_weight)
        print_log(f"Saving current best his_learner", log=log)

    print_log("------------------------------traffic predict training begin------------------------------", log=log)
    for epoch in range(args.cont + 1, max_epochs):
        train_time_s = time.time()

        train_loss = train_one_epoch(
            model, trainset_loader, optimizer, scheduler, criterion,    clip_grad, epoch, mixup_num, log=log,w=w, device = DEVICE , shift_weight =None )
        train_time_e = time.time()
        

        val_loss  = eval_model(model, valset_loader, criterion,  epoch, scheduler=scheduler,  shift_weight = None )
        val_time_e = time.time()
        

        train_loss_list.append(train_loss)

        val_loss_list.append(val_loss)

        if (epoch) % verbose == 0:
            print_log(
                datetime.datetime.now(),
                "Epoch",
                epoch,
                " \tTrain Loss = %.5f" % train_loss, 
                "Val Loss = %.5f" % val_loss,

                "LR:%.5f" % optimizer.param_groups[0]['lr'],
                "mixup_num:%d" % mixup_num,
                "Train_time for one epoch:",
                  (train_time_e - train_time_s),
                " \tVal_time for one epoch:",
                (val_time_e - train_time_e),
                " \tscheduler loss:",
                (scheduler.best),
                log=log,
            )

        if val_loss < min_val_loss:
            wait = 0
            min_val_loss = val_loss
            best_epoch = epoch
            best_state_dict = model.state_dict()
            save_checkpoint(model, optimizer, epoch, min_var_loss= min_val_loss,log = log)
        else:
            wait += 1
            if wait >= early_stop:
                break
        if args.tensorboard:
            w.add_scalars('loss_per_epoch',
                                    {'train_loss':train_loss,
                                        'valid_loss': val_loss}, epoch)
            w.add_scalar('LR',  optimizer.param_groups[0]['lr'], epoch)

    model.load_state_dict(best_state_dict)
    train_rmse, train_mae, train_mape = RMSE_MAE_MAPE(*predict(model, trainset_loader))
    val_rmse, val_mae, val_mape = RMSE_MAE_MAPE(*predict(model, valset_loader))

    out_str = f"Early stopping at epoch: {epoch}\n"
    out_str += f"Best at epoch {best_epoch}:\n"
    out_str += "Train Loss = %.5f\n" % train_loss_list[best_epoch - 1]
    out_str += "Train RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
        train_rmse,
        train_mae,
        train_mape,
    )
    out_str += "Val Loss = %.5f\n" % val_loss_list[best_epoch - 1]
    out_str += "Val RMSE = %.5f, MAE = %.5f, MAPE = %.5f" % (
        val_rmse,
        val_mae,
        val_mape,
    )
    print_log(out_str, log=log)
    if save:
        torch.save(best_state_dict, save)
    return model


@torch.no_grad()
def test_model(model, testset_loader,   save_dir, epoch = 0, log=None):
    model.to(DEVICE)
    model.eval()
    print_log("--------- Test ---------", log=log)

    start = time.time()
    y_true, y_pred = predict(model, testset_loader,  epoch)
    end = time.time()
    outputs = {'pred': y_pred, 'true': y_true}
    filename = \
                time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time())) + '_' \
                + 'predictions.npz'
    print("saving to:",os.path.join(save_dir, filename))
    np.savez_compressed(os.path.join(save_dir, filename), **outputs)#保存预测值和真实值
    
    rmse_all, mae_all, mape_all = RMSE_MAE_MAPE(y_true, y_pred)
    out_str = "All Steps RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
        rmse_all,
        mae_all,
        mape_all,
    )
    out_steps = y_pred.shape[1]
    for i in range(out_steps):
        rmse, mae, mape = RMSE_MAE_MAPE(y_true[:, i,:], y_pred[:, i, :])
        out_str += "Step %d RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
            i + 1,
            rmse,
            mae,
            mape,
        )

    print_log(out_str, log=log, end="")
    print_log("Inference time: %.2f s" % (end - start), log=log)

@torch.no_grad()
def test_simple(model, testset_loader,  epoch = 0, log=None, shift_weight = None):
    model = model.to(DEVICE)
    model.eval()
    print_log("--------- Test ---------", log=log)

    y_true, y_pred = predict(model, testset_loader,   epoch )
    end = time.time()

    rmse_all, mae_all, mape_all = RMSE_MAE_MAPE(y_true, y_pred)
    out_str = "All Steps RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
        rmse_all,
        mae_all,
        mape_all,
    )
    
    out_steps = y_pred.shape[1]
    for i in range(out_steps):
        rmse, mae, mape = RMSE_MAE_MAPE(y_true[:, i, :], y_pred[:, i, :])
        out_str += "Step %d RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
            i + 1,
            rmse,
            mae,
            mape,
        )

    print_log(out_str, log=log, end="")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="PEMS08")
    parser.add_argument("-g", "--gpu_num", type=int, default=0)
    parser.add_argument("-t", "--tensorboard", action='store_true')
    parser.add_argument("-e", "--exp_id", type=int, default=0)
    parser.add_argument("-c", "--comment", type=str, default='STFACH')
    parser.add_argument("-his", "--his_dim", type=int, default=7)
    parser.add_argument("-m", "--mode", type=str, default='train')
    parser.add_argument("-cont", "--cont", type=int, default=0)
    parser.add_argument("-s", "--speed", type=bool, default=False)
    parser.add_argument("-full_rate", "--full_rate", type=float, default=1.0) 
    args = parser.parse_args()
    seed =2044 # set random seed here
    init_seed(seed)
    set_cpu_num(40)
    
    GPU_ID = args.gpu_num
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{GPU_ID}"
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = args.dataset
    dataset = dataset.upper()
    data_path = f"../data/{dataset}/"
    model_name = STFACH.__name__
    if args.exp_id is None:
        args.exp_id = int(random.SystemRandom().random() * 100000)

    save_name = f"{str(args.exp_id)}_{dataset}_{model_name}_{str(args.comment)}"
    path = '../runs'
    log_dir = os.path.join(path, dataset, save_name)

    if (os.path.exists(log_dir)):
        print('has model save path')
    else:
        os.makedirs(log_dir)

    with open(f"{model_name}.yaml", "r",encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    cfg = cfg[dataset]

 # -------------------------------- tensorboard -------------------------------- #
    if args.tensorboard:
        tensorboard_dir = os.path.join(data_path, str('vis'),str(args.exp_id)+"_"+time.strftime("%m-%d-%Hh%Mm")).replace("\\", "/")
        if not (os.path.exists(tensorboard_dir)):
            os.makedirs(tensorboard_dir)
        args.t_dir = tensorboard_dir
        w   = SummaryWriter(log_dir=tensorboard_dir)
        print(tensorboard_dir)
    else:
        w = None

   
    # ------------------------------- make log file ------------------------------ #

    now = datetime.datetime.now().strftime("%m_%d(%H-%M-%S)")
    log = os.path.join(log_dir, f"{model_name}-{dataset}-{now}.log")
    log = open(log, "a")
    log.seek(0)
    log.truncate()

    his_log = os.path.join(log_dir, f"his_learner-{dataset}-{now}.log")
    his_log = open(his_log, "a")
    his_log.seek(0)
    his_log.truncate()


    # ------------------------------- load dataset ------------------------------- #
    print_log(dataset, log=log)
    (
        trainset_loader,
        valset_loader,
        testset_loader,
        SCALER,
        adj_mx
        ,adj_semx
    ) = generate_data_point(
        data_path,
        dataset,
        data_col= cfg.get("data_col"),
        output_dim= cfg["model_args"]["output_channels"],
        tod=cfg.get("time_of_day"),
        dow=cfg.get("day_of_week"),
        batch_size=cfg.get("batch_size", 64),
        full_rate=args.full_rate,
        log=log,
        log_dir = log_dir
    )
    
    print_log(log=log)
     # -------------------------------- load model -------------------------------- #

    model = STFACH(**cfg["model_args"], adj_mx = adj_mx, adj_semx = adj_semx)
    his_learner = HisPattern(dim = args.his_dim)


    # --------------------------- set model saving path -------------------------- #
 
    save = os.path.join(log_dir, f"{model_name}-{dataset}-{now}.pt")

    # ---------------------- set loss, optimizer, scheduler ---------------------- #
    criterion = nn.HuberLoss(delta=2.5)  

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 0),
        eps=cfg.get("eps", 1e-8),
        betas=(0.9, 0.999)
    )

    his_optimizer = torch.optim.Adam(his_learner.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.35,verbose=1,min_lr=0.000001,patience=5,  threshold_mode = 'rel',cooldown=0
                                                           ,threshold=0.0001)
    
    # --------------------------- print model structure -------------------------- #

    print_log("---------", model_name, "---------", log=log)
    print_log("---------", model, "---------", log=log)
    print_log(
        json.dumps(cfg, ensure_ascii=False, indent=4, cls=CustomJSONEncoder), log=log
    )
    print_log(log=log)

    # --------------------------- train and test model --------------------------- #

    print_log(f"Loss: {criterion._get_name()}", log=log)
    print_log(log=log)
    

    # --------------------------- Curriculum Learning --------------------------- #
    iter_count = 0
    if cfg["use_cl"]:
        print_log(f"Applying Curriculum Learning", log=log)
        target_length = 0
        print_log(f"CL target length = {target_length}", log=log)

    else :
        target_length = cfg["out_steps"]

 
 
    if args.cont > 0 :
        cache_path  =os.path.join(log_dir,f"best_model_{args.cont}.pth")
        print_log(f"loaded pretrain epoch of {args.cont} at {cache_path}", log = log)
    else:
        cache_path = None


    #### get mixup sample rate among data ####
    all_begin = time.time()
    mix_up = True
 
    sample_use_time = time.time() - all_begin
    mix_alpha = 2.0
    input_dim = cfg["model_args"]["input_dim"]

   
    if args.mode == 'train':
        model = train(
            model,
            trainset_loader,
            valset_loader,
            optimizer,
            scheduler,
            criterion,
            adj_mx,
            clip_grad=cfg.get("clip_grad"),
            max_epochs=cfg.get("max_epochs", 200),
            early_stop=cfg.get("early_stop", 10),
            verbose=1,
            log=log,
            save=save,
            cache_path = cache_path,
            w=w ,#tensorboard
            his_opt = his_optimizer,
            his_learner = his_learner
        )
        print_log(f"Saved Model: {save}", log=log)
        test_model(model, testset_loader,  save_dir = log_dir, log=log)

    else:
        print_log("--------", model, "---------", log=log)
        model.load_state_dict(torch.load(cache_path)['state_dict'])
        test_model(model, testset_loader,  save_dir = log_dir, log=log)

    log.close()
