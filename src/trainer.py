####################################################################################
# Author: Ashish Sinha
# Desc: Code for running N epochs of training and evaluation
####################################################################################

import os
import gc
import time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F

from utils import (make_variable,
                   adjust_learning_rate)
from losses import mix_rbf_mmd2, discrepancy
from networks import GradReverse
from metrics import calculate_accuracy_all

###############################
# Wrapper for training N epochs
###############################
def train(args,
            writer,
            growth_rate,
            source_trainloader,
            target_trainloaders,
            target_testloaders,
            optimizer,
            scheduler,
            model,
            save_name,
            logger,
            device
            ):
    """
    Main training wrapper
    
    Input: 
        args: hyper-parameters
        writer: tensorboard logger
        logger: print logger
        growth_rate: scheduled weight for loss
        source_trainloader: dataloader for source domain
        target_trainloaders: dataloader for T target domains
        target_testloades: dataloader for T target domains
        optimizer: optimizer (Adam/ SGD)
        scheduler: learning rate scheduler
        model: Modified PointDAN model
        save_name: filename for checkpoint
        device: CPU or GPU
        
    Output:
        best_sample_acc: Best mean sample accuracy across all target domains
        best_class_acc: Best mean per-class accuracy across all target domains
    """
    logger.info('*'*20)
    logger.info('Starting Training:')
    logger.info('*'*20)

    best_class_acc = 0.
    best_sample_acc = 0.
    epochs = args.epochs +1
    
    curr_epoch = args.curr_epoch
    global iteration
    
    #load from checkpoints
    if args.resume:
        state = torch.load(args.resume)
        curr_epoch = state['curr_epoch']
        optimizer.load_state_dict(state['optim'])
        model.load_state_dict(state['model'])
        scheduler.load_state_dict(state['scheduler'])
        
        logger.info (f'Checkpoints loaded from epoch: {curr_epoch}')
        
    # start training for N epochs from curr_epoch
    for epoch in (range(curr_epoch, epochs)):
        beta = args.init_beta * torch.exp(growth_rate * (epoch-1))
        beta = beta.to(device)
        start = time.time()
        
        new_lr = args.lr / np.power(1 + 10 * (epochs -1)/epochs, 0.75)
        adjust_learning_rate(optimizer, new_lr)
        
        # run training one epoch
        total_losses, da_losses, da_cls_losses = train_one_epoch(epoch, epochs, 
                                                                model,
                                                                optimizer,
                                                                source_trainloader,
                                                                target_trainloaders, 
                                                                writer, 
                                                                args, 
                                                                beta,
                                                                device,
                                                                logger)
        
        time_pass_e = time.time() - start
        logger.info('Epoch {} finished in {:.0f}h {:.0f}m {:.0f}s'.format(epoch, time_pass_e // 3600, time_pass_e %3600 // 60, time_pass_e % 3600 % 60))

        #saving checkpoints
        if epoch % args.save_interval ==0:
            state = {
                'beta':beta,
                'curr_epoch':epoch,
                'model':model.state_dict(),
                'optim':optimizer.state_dict(),
                'scheduler':scheduler.state_dict()
            }
            
            logger.info ('Saving Weights for epoch: {}'.format(epoch))
            ckpt_path = os.path.join(args.save_dir, f'epoch_{epoch}')
            if not os.path.exists(ckpt_path):
                os.makedirs(ckpt_path)
            weights_name =os.path.join(ckpt_path, 'weights.pth')
            torch.save(state, weights_name)
            
            del state
        
        targets_sample_acc = np.zeros(len(target_trainloaders))
        targets_class_acc = np.zeros(len(target_trainloaders))
        best_targets_acc = np.zeros(len(target_trainloaders))
        targets_loss = np.zeros(len(target_trainloaders))
        
        # Running inference on target test domains 
        # after each epoch
        logger.info ('*'*20)
        logger.info ('Validating.')
        logger.info('*'*20)
        for i, dl in (enumerate(target_testloaders)):
            targets_loss[i], targets_sample_acc[i], targets_class_acc[i], target_acc_per_class= eval(model, dl, device)
            
            writer.add_scalar(f'test/target_{i+1}_pred_loss', targets_loss[i], epoch)
            writer.add_scalar(f'test/target_{i+1}_sample_acc', targets_sample_acc[i], epoch)
            writer.add_scalar(f'test/target_{i+1}_class_acc', targets_class_acc[i], epoch)

            logger.info(f'Acc per class on Target {i+1}: {target_acc_per_class}')
        
        total_target_sample_acc = targets_sample_acc.mean()
        total_target_class_acc = targets_class_acc.mean()
        
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        # save model with best sample accuracy
        if total_target_sample_acc > best_sample_acc:
            best_sample_acc = total_target_sample_acc
            torch.save({'model': model.state_dict(), 'acc': best_sample_acc, 'epoch': epoch},
                       "{}/da_alt_best_sample_model.pth".format(args.save_dir))

            logger.info ('*'*20)
            logger.info(f'Best Model saved with sample acc: {best_sample_acc} %')
            logger.info ('*'*20)
        
        # save model with best class accuracy
        if total_target_class_acc > best_class_acc:
            best_class_acc = total_target_class_acc
            torch.save({'model': model.state_dict(), 'acc': best_class_acc, 'epoch': epoch},
                       "{}/da_alt_best_class_model.pth".format(args.save_dir))

            logger.info ('*'*20)
            logger.info(f'Best Model saved with class acc: {best_class_acc} %')
            logger.info ('*'*20)
            
        logger.info (f'<Test> Epoch: {epoch} | Overall Sample Acc: {total_target_sample_acc} | Overall Class Acc: {total_target_class_acc}')
        
        # Logging to tensorboard 
        writer.add_scalar('beta_epoch', beta.item(), epoch)
        for i in range(len(target_trainloaders)):
            writer.add_scalar("training_loss_target_{}".format(i+1), total_losses[i].item(), epoch)
        for i in range(len(target_trainloaders)):
            writer.add_scalar("da_loss_target_{}".format(i+1), da_losses[i].item(), epoch)
        writer.add_scalar("da_lr_epoch", optimizer.param_groups[0]['lr'], epoch)
        for i in range(len(target_trainloaders)):
            writer.add_scalar("val_target_{}_sample_acc".format(i+1), targets_sample_acc[i], epoch)
            writer.add_scalar("val_target_{}_class_acc".format(i+1), targets_class_acc[i], epoch)
        for i in range(len(target_trainloaders)):
            writer.add_scalar("da_cls_loss{}".format(i+1), da_cls_losses[i], epoch)
            
        scheduler.step()
    
    return best_sample_acc, best_class_acc

###############################
# Wrapper for running 1 epoch
###############################
def train_one_epoch(curr_epoch,
                       epochs, 
                       model, 
                       optimizer,
                       source_trainloader,
                       target_trainloaders,
                       writer,
                       args,
                       beta,
                       device,
                       logger):
    """
    Wrapper for running a single epoch of training. 
    
    Input: 
        curr_epoch: epoch to start training from
        epochs: total number of epochs
        writer: tensorboard logger
        logger: print logger
        growth_rate: scheduled weight for loss
        source_trainloader: dataloader for source domain
        target_trainloaders: dataloader for T target domains
        optimizer: optimizer (Adam/ SGD)
        model: Modified PointDAN model
        logger: logger for printing loss to a log file
        
    Output:
        total_losses: Sum total of supervised classification loss, mixup loss, domain confusion loss and discrepancy loss
        da_temp_losses: loss to optimize as per loss equation defined in the paper
        da_cls_losses: domain confusion loss
    """
    # set the model to train
    model.train()
    global iteration
    
    # define standard deviation for MMD loss
    sigma_list = [0.01, 0.1, 1, 10, 100]
    
    device_ids = list(map(int, args.gpu.split(',')))
    
    total_losses = torch.zeros(len(target_trainloaders))
    da_temp_losses = torch.zeros(len(target_trainloaders))
    da_cls_losses = torch.zeros(len(target_trainloaders))
    
    # create dataloaders for target domains
    iter_targets = [0] * len(target_trainloaders)
    for i, d in enumerate(target_trainloaders):
        iter_targets[i] = iter(d)
    
    # define target domains
    num_domains = len(target_trainloaders)
    
    for i, batch in (enumerate(source_trainloader)):
        
        # sample source data
        source_data, source_label = batch
        source_data = make_variable(source_data)
        source_label = make_variable(source_label)
        
        torch.cuda.empty_cache()
        
        batch_n = 0
        data_t=[0]*len(target_trainloaders)
        
        # sample all target domains simultaneously 
        for j in range(num_domains):
            batch_n+=1
            try:
                data_t[j], _ = iter_targets[j].next()
            except StopIteration:
                tar = iter(target_trainloaders[j])
                data_t[j], _ = tar.next()
            if data_t[j].shape[0] > source_data.shape[0]:
                data_t[j] = data_t[j][:source_data.shape[0]]
            elif data_t[j].shape[0] < source_data.shape[0]:
                source_data = source_data[:data_t[j].shape[0]]
                source_label = source_label[:data_t[j].shape[0]]
                
        iteration += 1
        
        optimizer.zero_grad()
        l_s = len(source_trainloader)
        
        # scalar value for Gradient Reversal Layer
        p = float(i + (curr_epoch - 1) * l_s) / epochs / l_s
        delta = 2. / (1. + np.exp(-10. * p)) -1
        
        target_preds, target_domain_losses, target_feats=list(), list(), list()
        
        source_data = make_variable(source_data)
        source_label = make_variable(source_label)
        
        # forward pass on source data
        source_pred, \
            source_domain_loss, \
                source_feat = model(source_data, constant=delta)
        
        # forward pass on target data
        for j in range(num_domains):
            data_t[j] = make_variable(data_t[j])
            
            target_pred,\
                target_domain_loss,\
                    target_feat = model(data_t[j], constant=delta, source=False)
            target_preds.append(target_pred)
            target_domain_losses.append(target_domain_loss)
            target_feats.append(target_feat)
            
        torch.cuda.empty_cache()
        
        # supervised classification loss
        source_loss_cls = F.cross_entropy(source_pred, source_label.long())
        writer.add_scalar('train/source_class_loss', source_loss_cls.mean().item(), iteration)
        
        #Domain Classification Loss (Domain Confusion)
        da_cls_loss = torch.stack(
            [(source_domain_loss + target_domain_losses[j]) for j in range(num_domains)]
        )
        if len(device_ids)>1:
            da_cls_loss = da_cls_loss.mean(1)
        
        writer.add_scalar('train/domain_cls_loss', da_cls_loss.mean().item(),iteration)
        
        # Discrepancy Loss based on L1 Loss between source and target domain feat
        disc_loss = torch.stack(
            [discrepancy(source_pred, target_preds[j]) * args.lambda_disc for j in range(num_domains)]
            )
        da_cls_loss += disc_loss
        
        # Domain Mixup of Latent Feature Representation
        if args.mixup:
            clip_thr = args.mixup_thres
            
            # sample a value from Beta distribution
            mix_ratio = np.random.beta(2, 2)
            mix_ratio = round(mix_ratio, 2)
            
            # clip the mixup ratio 
            if (mix_ratio >= 0.5 and mix_ratio < (0.5 + clip_thr)):
                mix_ratio = 0.5 + clip_thr
            if (mix_ratio > (0.5 - clip_thr) and mix_ratio < 0.5):
                mix_ratio = 0.5 - clip_thr

            # create mixed labels
            # ratio of source domain in the mixture
            label_mix = make_variable((torch.ones(source_feat.size(0))*mix_ratio).long())
            
            # create mixed embedding
            emb_mix = torch.stack([
                mix_ratio * source_feat + (1 - mix_ratio) * target for target in target_feats
                ])
            
            # Baseline for Mixup (Mix-Sep)
            # loss is calculated for source domain and each target domain sequentially
            if args.mix_sep:
                """
                Reverse the gradients to learn domain invariant features from a common
                pool of latent features of source domain mixed with features of target 
                domains by adverserially training the Discriminator and feature extractor
                """
                emb_feats = [GradReverse.apply(emb, delta) for emb in emb_mix]
                emb_out = [model.domain_classifier(emb_feat) for emb_feat in emb_feats]
                mixup_loss = torch.stack([
                    F.binary_cross_entropy(out, label_mix.unsqueeze(-1).float()) for out in emb_out
                ])
                if len(device_ids)>1:
                    mixup_loss = mixup_loss.mean(1)
                 
            # Mixup Variants 
            else:
                # The Proposed Mixup: MEnsA -– Feature Embeddings of all domains are aggregagated into one via mean   
                if args.mix_type ==-1:
                    emb_mix = emb_mix.mean(0)
                    emb_feat = GradReverse.apply(emb_mix, delta)
                    emb_out = model.domain_classifier(emb_feat)
                    mixup_loss = F.binary_cross_entropy(emb_out, label_mix.unsqueeze(1).float())

                # Mixup A –– Scaling Factor changed
                elif args.mix_type == 0:
                    emb_mix = mix_ratio * source_feat + (1- mix_ratio)/num_domains * target_feats[0] 
                    for k in range(1, len(target_feats)):
                        emb_mix += (1- mix_ratio)/ len(target_feats) * target_feats[k]
                    emb_feat = GradReverse.apply(emb_mix, delta)
                    emb_out = model.domain_classifier(emb_feat)
                    mixup_loss = F.binary_cross_entropy(emb_out, label_mix.unsqueeze(1).float())
                
                # Mixup B –– Concat instead of aggregation by summation
                elif args.mix_type == 1:
                    s_hat = mix_ratio * source_feat
                    t_hat = (1- mix_ratio)/len(target_feats) * target_feats[0]
                    for k in range(1, len(target_feats)):
                        t_hat = torch.cat([t_hat, (1- mix_ratio)/len(target_feats) * target_feats[k]], 0)
                    emb_mix = torch.cat([s_hat, t_hat], 0)
                    label_mix = torch.ones(source_feat.size(0)) * mix_ratio
                    label_mix = torch.cat([label_mix, torch.ones(target_feats[0].size(0)) *2.*(1-mix_ratio)/len(target_feats)], 0)
                    label_mix = torch.cat([label_mix, torch.ones(target_feats[0].size(0)) *3.*(1-mix_ratio)/len(target_feats)], 0)
                    emb_feat = GradReverse.apply(emb_mix, delta)
                    emb_out = model.domain_classifier1(emb_feat)
                    label_mix = make_variable(label_mix.long())
                    mixup_loss = F.cross_entropy(emb_out, label_mix)
                
                # Mixup C –– Mixup of Target Domains excluding source domain
                elif args.mix_type == 2:
                    emb_mix = emb_mix.mean(0)
                    emb_feat = GradReverse.apply(emb_mix, delta)
                    emb_out = model.domain_classifier(emb_feat)
                    mixup_loss = F.binary_cross_entropy(emb_out, label_mix.unsqueeze(1).float())
                    
                    emb_mix = mix_ratio * target_feats[0] + (1-mix_ratio) * target_feats[1]
                    label_mix = make_variable(torch.ones(target_feats[0].size(0)).long())
                    emb_feat = GradReverse.apply(emb_mix, delta)
                    emb_out = model.domain_classifier(emb_feat)
                    
                    mixup_loss += F.binary_cross_entropy(emb_out, label_mix.unsqueeze(1).float())
                    
            writer.add_scalar('train/mixup_Loss', mixup_loss.mean().item(), iteration)
            da_cls_loss += args.lambda_mix * mixup_loss
        
        # MMD loss for discrepancy between source and target domains
        temp = torch.stack(
            [mix_rbf_mmd2(source_pred, target_preds[j], sigma_list) for j in range(num_domains)]
        )
        # Adversaral loss = Domain Confusion loss + Discrepancy Loss
        adv_loss = args.lambda_mmd *  temp+ \
            da_cls_loss * args.lambda_adv
            
        
        writer.add_scalar('train/adv_loss', adv_loss.mean().item(), iteration)
        
        # Total Multi Target Domain Adaptation Loss
        da_loss = torch.log(torch.sum(torch.exp(args.gamma * (source_loss_cls + beta * da_cls_loss)))) / args.gamma
            
        da_temp_losses[j] += da_loss.mean().item()
        
        # Optimize the da_loss
        da_loss.mean().backward()
        optimizer.step()
        optimizer.zero_grad()
        
        da_cls_losses[j] += da_cls_loss.mean().item()
        total_losses[j] += adv_loss.mean().item() + source_loss_cls.mean().item() + da_cls_loss.mean().item()
        
        n_t = len(source_trainloader)
        iteration -=1
        if iteration % args.log_interval==0:
            logger.info(f'<Train> Epoch: {curr_epoch} | Iter: {iteration}/ {n_t * args.epochs} | '\
                f'Source Cls Loss: {source_loss_cls.mean():.4f} | '\
                f'Domain Cls Loss: {da_cls_loss.mean():.4f} | '\
                f'Adv Loss: {adv_loss.mean():.4f} | '\
                f'Total DA Loss: {da_loss.mean():.4f}'
                )
        iteration +=1
            
    del da_loss
    gc.collect()
    torch.cuda.empty_cache()
    
    return total_losses/ len(source_trainloader),\
            da_temp_losses/len(source_trainloader),\
            da_cls_losses/len(source_trainloader)

###############################
# Wrapper for inference
###############################
def eval(model, test_loader, device):
    """
    Runs inference on target domains' test split after each epoch
    
    Input:
        test_loader: dataloader for test data of target domain
    
    Output:
        pred_loss: validation loss on the test data
        sample_acc: mean sample accuracy on test data
        class_acc: mean class accuracy on test data
        acc_per_class: accuracy for each class
    """
    model.eval()
    
    loss_total = 0
    total = 0
    
    pred_list = torch.zeros([0]).long().to(device) 
    label_list = torch.zeros([0]).long().to(device)
    
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data = make_variable(data)
            target = make_variable(target)
            total += data.size(0)
            
            output = model(data, source= False)
            loss = F.cross_entropy(output, target.long())
            _, pred = torch.max(output, 1)
            
            loss_total += loss.item() * data.size(0)
            pred_list = torch.cat([pred_list, pred], 0)    
            label_list = torch.cat([label_list, target], 0)
                
        pred_loss = loss_total/ total
        sample_acc, class_acc, acc_per_class = calculate_accuracy_all(pred_list, label_list, 10) # Num Class =10
        
        return (pred_loss,
                sample_acc * 100.,
                class_acc*100.,
                acc_per_class
                )
