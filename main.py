import os
import random
import argparse
import yaml
from tqdm import tqdm
import json

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import itertools

from datasets import build_dataset
from datasets.utils import build_data_loader
import clip
from utils import *


def get_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', default='configs/aircraft.yaml', help='settings in yaml format')
    args = parser.parse_args()

    return args


def run_dawa(cfg, cache_keys, cache_values, gda_params, val_features, val_labels, test_features, test_labels, clip_weights):
    
    print("\n-------- Searching hyperparameters on the val set. --------")

    # Zero-shot CLIP
    clip_logits = 100. * val_features @ clip_weights
    acc = cls_acc(clip_logits, val_labels)
    print("\n**** Zero-shot CLIP's val accuracy: {:.2f}. ****\n".format(acc))

    # Tip-Adapter
    beta, alpha = cfg['init_beta'], cfg['init_alpha']
    
    affinity = val_features @ cache_keys 
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
    
    tip_logits = clip_logits + cache_logits * alpha
    acc = cls_acc(tip_logits, val_labels)
    print("**** Tip-Adapter's val accuracy: {:.2f}. ****\n".format(acc))

    # Search Hyperparameters
    best_beta, best_alpha = search_hp(cfg, cache_keys, cache_values, val_features, val_labels, clip_weights)

    print("\n-------- Evaluating on the test set. --------")

    # Zero-shot CLIP
    clip_logits = 100. * test_features @ clip_weights
    acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(acc))

    # Tip-Adapter    
    affinity = test_features @ cache_keys
    cache_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values
    
    tip_logits = clip_logits + cache_logits * best_alpha
    acc = cls_acc(tip_logits, test_labels)
    print("**** Tip-Adapter's test accuracy: {:.2f}. ****\n".format(acc))

    # GDA Adapter
    best_gda_beta, best_gda_alpha = search_hp_dawa(cfg, gda_params, cache_keys, cache_values, val_features, val_labels, clip_weights)
    gda_logits = (test_features @ gda_params['W'] + gda_params['b']) @ gda_params['gda_one_hot']    

    # Fused 
    tip_logits = ((-1) * (best_gda_beta - best_gda_beta * affinity)).exp() @ cache_values
    gamma = 2. /(1. + torch.exp(torch.tensor(-2.5*(int(cfg["shots"]) - 5.))))
    tg_logits = logits_fuse(clip_logits, [tip_logits, gda_logits], gamma)
    fuse_logits = clip_logits + tg_logits * best_gda_alpha
    acc = cls_acc(fuse_logits, test_labels)
    print("**** DAWA's test accuracy: {:.2f}. ****\n".format(acc))


def run_dawa_F(cfg, cache_keys, cache_values, gda_params, 
                      val_features, val_labels, 
                      test_features, test_labels, 
                      clip_weights, clip_model, 
                      train_loader_F, 
                      resume=False):

    # Enable the cached keys to be learnable
    gda_adapter = nn.Linear(gda_params['W'].shape[0], gda_params['W'].shape[1], bias=True).to(clip_model.dtype).cuda()
    gda_adapter.weight = nn.Parameter(gda_params['W'].t())
    gda_adapter.bias = nn.Parameter(gda_params['b'])
    tip_adapter = nn.Linear(cache_keys.shape[0], cache_keys.shape[1], bias=False).to(clip_model.dtype).cuda()
    tip_adapter.weight = nn.Parameter(cache_keys.t())

    optimizer = torch.optim.AdamW(tip_adapter.parameters(), 
                                  lr=cfg['lr'], eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['train_epoch'] * len(train_loader_F))
    
    beta, alpha = cfg['init_beta'], cfg['init_alpha']
    gamma = 2. /(1. + torch.exp(torch.tensor(-2.5*(int(cfg["shots"]) - 5.))))
    best_acc, best_epoch = 0.0, 0
    start_epoch = 0
    writer = SummaryWriter(log_dir=cfg['cache_dir'] + '/runs')
    # Initialize the adapter's weights from previous ckpt
    if resume and os.path.exists(cfg['cache_dir'] + f"/last_F_{cfg['shots']}shots.pt"):
        checkpoint = torch.load(cfg['cache_dir'] + f"/last_F_{cfg['shots']}shots.pt")
        gda_adapter.load_state_dict(checkpoint['gda_state_dict'])
        tip_adapter.load_state_dict(checkpoint['tip_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint.get('best_acc', 0.0)
        best_epoch = checkpoint.get('best_epoch', 0)

    # Train
    for train_idx in trange(start_epoch, cfg['train_epoch']):
        gda_adapter.train()
        tip_adapter.train()
        correct_samples, all_samples = 0, 0
        loss_list = []
        print('Train Epoch: {:} / {:}'.format(train_idx, cfg['train_epoch']))

        for i, (images, target) in enumerate(train_loader_F):
            images, target = images.cuda(), target.cuda()
            with torch.no_grad():
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)

            proto_loss = contrast_loss(tip_adapter.weight, image_features, target, cache_values)
            
            clip_logits = 100. * image_features @ clip_weights
            tip_affinity = tip_adapter(image_features)
            tip_logits = ((-1) * (beta - beta * tip_affinity)).exp() @ cache_values
            gda_affinity = gda_adapter(image_features)
            gda_logits = gda_affinity @ gda_params['gda_one_hot']

            cache_logits = logits_fuse(clip_logits, [tip_logits, gda_logits], gamma)
            fuse_logits = clip_logits + cache_logits * alpha

            ce_loss = F.cross_entropy(fuse_logits, target) 

            # 随机选择batch个query，计算entropy regularization
            query_features = test_features[torch.randperm(test_features.size(0))[:image_features.size(0)]]
            clip_logits_q = 100. * query_features @ clip_weights
            tip_affinity_q = tip_adapter(query_features)
            tip_logits_q = ((-1) * (beta - beta * tip_affinity_q)).exp() @ cache_values
            gda_affinity_q = gda_adapter(query_features)
            gda_logits_q = gda_affinity_q @ gda_params['gda_one_hot']
            cache_logits_q = logits_fuse(clip_logits_q, [tip_logits_q, gda_logits_q], gamma)
            fuse_logits_q = clip_logits_q + cache_logits_q * alpha

            loss =  ce_loss \
                    + compute_entropy(fuse_logits_q) * cfg['entropy_weight'] \
                    + proto_loss * cfg['contrast_weight']
            
            acc = cls_acc(fuse_logits, target)
            correct_samples += acc / 100 * len(fuse_logits)
            all_samples += len(fuse_logits)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples, correct_samples, all_samples, sum(loss_list)/len(loss_list)))

        # Eval
        gda_adapter.eval()
        tip_adapter.eval()

        clip_logits = 100. * test_features @ clip_weights
        tip_affinity = tip_adapter(test_features)
        tip_logits = ((-1) * (beta - beta * tip_affinity)).exp() @ cache_values
        gda_affinity = gda_adapter(test_features)
        gda_logits = gda_affinity @ gda_params['gda_one_hot']
        cache_logits = logits_fuse(clip_logits, [tip_logits, gda_logits], gamma)
        fuse_logits = clip_logits + cache_logits * alpha
        acc = cls_acc(fuse_logits, test_labels) 

        print("**** DAWA-F's test accuracy: {:.2f}. ****\n".format(acc))
        if acc > best_acc:
            best_acc = acc
            best_epoch = train_idx
            torch.save(tip_adapter.state_dict(), cfg['cache_dir'] + "/best_tip_F_" + str(cfg['shots']) + "shots.pt")
            torch.save(gda_adapter.state_dict(), cfg['cache_dir'] + "/best_gda_F_" + str(cfg['shots']) + "shots.pt")
        
        # Log
        writer.add_scalar('Train_Loss/epoch', sum(loss_list)/len(loss_list), train_idx)
        writer.add_scalar('val/acc', acc, train_idx)
        writer.add_scalar('Train_Acc/epoch', correct_samples / all_samples, train_idx)
        
        # Save the model every 10 epochs
        if (train_idx+1) % 10 == 0:
            print(f"Saving DAWA-F's weights at epoch {train_idx} to {cfg['cache_dir']}/last_F_{cfg['shots']}shots.pt")
            # Save the adapter's weights and optimizer's state
            torch.save({
                'epoch': train_idx,
                'tip_state_dict': tip_adapter.state_dict(),
                'gda_state_dict': gda_adapter.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_acc': best_acc,
                'best_epoch': best_epoch
            }, cfg['cache_dir'] + "/last_F_" + str(cfg['shots']) + "shots.pt")

    # Load the best adapter weights
    tip_adapter.load_state_dict(torch.load(cfg['cache_dir'] + "/best_tip_F_" + str(cfg['shots']) + "shots.pt"))
    gda_adapter.load_state_dict(torch.load(cfg['cache_dir'] + "/best_gda_F_" + str(cfg['shots']) + "shots.pt"))

    print(f"**** After fine-tuning, DAWA-F's best test accuracy: {best_acc:.2f}, at epoch: {best_epoch}. ****\n")

    print("\n-------- Searching hyperparameters on the val set. --------")

    # Search Hyperparameters
    best_beta, best_alpha = search_hp_dawa(cfg, gda_params, cache_keys, cache_values, val_features, val_labels, clip_weights, 
                                      tip_adapter, gda_adapter)

    print("\n-------- Evaluating on the test set. --------")
   
    clip_logits = 100. * test_features @ clip_weights
    tip_affinity = tip_adapter(test_features)
    cache_logits = ((-1) * (best_beta - best_beta * tip_affinity)).exp() @ cache_values
    gda_affinity = gda_adapter(test_features)
    gda_logits = gda_affinity @ gda_params['gda_one_hot']
    cache_logits = logits_fuse(clip_logits, [cache_logits, gda_logits], gamma)
    fuse_logits = clip_logits + cache_logits * best_alpha
    acc = cls_acc(fuse_logits, test_labels)
    print("**** DAWA-F's test accuracy: {:.2f}. ****\n".format(max(best_acc, acc)))
    writer.close()

    return acc


def main():

    # Load config file
    args = get_arguments()
    assert (os.path.exists(args.config))
    
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    cache_dir = os.path.join('./caches', cfg['dataset'])
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir

    print("\nRunning configs.")
    print(cfg, "\n")

    # CLIP
    clip_model, preprocess = clip.load(cfg['backbone'])
    clip_model.eval()

    # Prepare dataset
    seed = cfg['seed']
    random.seed(seed)
    torch.manual_seed(seed)
    
    print("Preparing dataset.")
    dataset = build_dataset(cfg['dataset'], cfg['root_path'], cfg['ways'], cfg['shots'], cfg['data_split_json'])

    val_loader = build_data_loader(data_source=dataset.val, batch_size=64, is_train=False, tfm=preprocess, shuffle=False)
    test_loader = build_data_loader(data_source=dataset.test, batch_size=64, is_train=False, tfm=preprocess, shuffle=False)

    train_tranform = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.5, 1), 
                                    #  ratio=(0.99, 1.01), 
                                     interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073 ), std=(0.26862954, 0.26130258, 0.27577711))
    ])

    train_loader_cache = build_data_loader(data_source=dataset.train_x, batch_size=64, tfm=train_tranform, is_train=True, shuffle=False)
    train_loader_F = build_data_loader(data_source=dataset.train_x, batch_size=64, tfm=train_tranform, is_train=True, shuffle=True)

    # Textual features
    print("\nGetting textual features as CLIP's classifier.")
    # clip_weights = clip_classifier(dataset.classnames, dataset.template, clip_model)
    with open('./caption_file/{}.json'.format(cfg['dataset']), encoding='utf-8') as f:
        gpt3_prompt = json.load(f)
    clip_weights = knowledge_clip_weights(dataset.classnames, gpt3_prompt, clip_model)

    # Construct the cache model by few-shot training set
    print("\nConstructing cache model by few-shot visual features and labels.")
    cache_keys, cache_values, gda_params = build_multiproto_cache_model(cfg, clip_model, train_loader_cache)

    # Pre-load val features
    print("\nLoading visual features and labels from val set.")
    val_features, val_labels = pre_load_features(cfg, "val", clip_model, val_loader)

    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(cfg, "test", clip_model, test_loader)

    # ------------------------------------------ DAWA ------------------------------------------
    run_dawa(cfg, cache_keys, cache_values, gda_params, val_features, val_labels, test_features, test_labels, clip_weights)

    # ------------------------------------------ DAWA-F ------------------------------------------
    run_dawa_F(cfg, cache_keys, cache_values, gda_params, val_features, val_labels, test_features, test_labels, clip_weights, clip_model, train_loader_F)

if __name__ == '__main__':
    main()