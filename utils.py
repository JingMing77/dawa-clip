from tqdm import tqdm, trange

import torch
import torch.nn.functional as F
import torch.nn as nn

import clip


def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc


def build_cache_model(cfg, clip_model, train_loader_cache):

    if cfg['load_cache'] == False:    
        cache_keys = []
        cache_values = []
        gda_params = {}  

        with torch.no_grad():
            # Data augmentation for the cache model
            all_vecs = []  
            all_labels = []  
            
            for augment_idx in range(cfg['augment_epoch']):
                train_features = []

                print('Augment Epoch: {:} / {:}'.format(augment_idx, cfg['augment_epoch']))
                for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                    images, target = images.cuda(), target.cuda()
                    image_features = clip_model.encode_image(images)
                    train_features.append(image_features)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                    all_vecs.append(image_features)
                    all_labels.append(target)

                    if augment_idx == 0:
                        cache_values.append(target)
                cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))
            
            cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
            cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
            cache_keys = cache_keys.permute(1, 0)
            
            num_ways = len(cfg['ways']) if type(cfg['ways']) is not int else cfg['ways']
            if num_ways > 0:
                all_targets = torch.cat(cache_values, dim=0)
                unique_classes, mapped_targets = torch.unique(all_targets, sorted=True, return_inverse=True)
                cache_values = mapped_targets
                cache_values = F.one_hot(cache_values, num_classes=num_ways).half()
                num_classes = num_ways

            else:   # all classes
                cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()
                num_classes = cache_values.shape[1]
            
            # compute GDA parameters W and b
            print("Computing GDA parameters...")
            vecs = torch.cat(all_vecs).float()      # [num_samples * augment_epoch, feature_dim]
            labels = torch.cat(all_labels)          # [num_samples * augment_epoch]
            
            W, b = estimate_gda_parameters(vecs, labels, num_classes)
            
            gda_params['W'] = W
            gda_params['b'] = b
            print(f"GDA parameters computed: W shape {W.shape}, b shape {b.shape}")

        # 保存cache和GDA参数
        torch.save(cache_keys, cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
        torch.save(cache_values, cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")
        torch.save(gda_params, cfg['cache_dir'] + '/gda_params_' + str(cfg['shots']) + "shots.pt")  # 新增

    else:
        # 加载cache和GDA参数
        cache_keys = torch.load(cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
        cache_values = torch.load(cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")
        gda_params = torch.load(cfg['cache_dir'] + '/gda_params_' + str(cfg['shots']) + "shots.pt")  # 新增

    return cache_keys, cache_values, gda_params 

def build_multiproto_cache_model(cfg, clip_model, train_loader_cache, proto_per_cls=2):

    if cfg['load_cache'] == False:    
        cache_keys = []
        cache_values = []
        gda_params = {}  

        with torch.no_grad():
            # Data augmentation for the cache model
            all_vecs = []  # 每个元素 shape: [sample_num, feature_dim]
            all_labels = []  # 每个元素 shape: [sample_num]
            for ii in range(proto_per_cls):
                print(f"Building cache for prototype {ii + 1} / {proto_per_cls}")
                for augment_idx in trange(cfg['augment_epoch']):
                    train_features = []
                    vecs_epoch = []
                    labels_epoch = []
                    # print('Augment Epoch: {:} / {:}'.format(augment_idx, cfg['augment_epoch']))
                    for jj, (images, target) in enumerate(train_loader_cache):
                        images, target = images.cuda(), target.cuda()
                        image_features = clip_model.encode_image(images)
                        train_features.append(image_features)
                        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                        vecs_epoch.append(image_features)
                        labels_epoch.append(target)
                        if ii == 0 and augment_idx == 0:
                            cache_values.append(target)
                    if ii == 0:
                        cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))
                    # 每个augment_epoch保存一次
                    all_vecs.append(torch.cat(vecs_epoch, dim=0))
                    all_labels.append(torch.cat(labels_epoch, dim=0))

            cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
            cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
            cache_keys = cache_keys.permute(1, 0)
            
            num_ways = len(cfg['ways']) if type(cfg['ways']) is not int else cfg['ways']
            if num_ways > 0:
                all_targets = torch.cat(cache_values, dim=0)
                unique_classes, mapped_targets = torch.unique(all_targets, sorted=True, return_inverse=True)
                cache_values = mapped_targets
                cache_values = F.one_hot(cache_values, num_classes=num_ways).half()
                num_classes = num_ways

            else:   # all classes
                cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()
                num_classes = cache_values.shape[1]
            
            # compute GDA parameters W and b
            print("Computing GDA parameters...")
            
            gda_one_hot = torch.eye(num_classes, device=cache_values.device).repeat(proto_per_cls, 1).half()  # [proto_per_cls * num_classes, num_classes]
            W_list, b_list = [], []
            for ii in range(proto_per_cls):
                start, end = ii * cfg['augment_epoch'], (ii + 1) * cfg['augment_epoch']
                vecs = torch.cat(all_vecs[start:end]).float()
                labels = torch.cat(all_labels[start:end])
                Wi, bi = estimate_gda_parameters(vecs, labels, num_classes)
                W_list.append(Wi)
                b_list.append(bi)
            W = torch.stack(W_list)
            b = torch.stack(b_list)
            # W, b = W /10. , b /10.   # 缩放
            W = W.permute(1, 0, 2).reshape(W.shape[1], -1)  # W: [proto_per_cls, feature_dim, num_classes] -> [feature_dim, proto_per_cls * num_classes]
            b = b.reshape(-1)       # b: [proto_per_cls, num_classes] -> [proto_per_cls * num_classes]
            gda_params['W'] = W.half()
            gda_params['b'] = b.half()
            gda_params['gda_one_hot'] = gda_one_hot.half()
            gda_params['proto_per_cls'] = proto_per_cls
            gda_params['num_classes'] = num_classes
            print(f"GDA parameters computed: \n W shape [feature_dim, proto_per_cls * num_classes] =  {W.shape}, "
                  f"\n b shape [proto_per_cls * num_classes] = {b.shape}")

        # 保存cache和GDA参数
        torch.save(cache_keys, cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
        torch.save(cache_values, cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")
        torch.save(gda_params, cfg['cache_dir'] + '/gda_params_' + str(cfg['shots']) + "shots.pt")  # 新增

    else:
        # 加载cache和GDA参数
        cache_keys = torch.load(cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
        cache_values = torch.load(cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")
        gda_params = torch.load(cfg['cache_dir'] + '/gda_params_' + str(cfg['shots']) + "shots.pt")  # 新增

    return cache_keys, cache_values, gda_params 


def pre_load_features(cfg, split, clip_model, loader):

    if cfg['load_pre_feat'] == False:
        features, labels = [], []

        with torch.no_grad():
            for i, (images, target) in enumerate(tqdm(loader)):
                images, target = images.cuda(), target.cuda()
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                features.append(image_features)
                labels.append(target)

        features, labels = torch.cat(features), torch.cat(labels)
        
        num_ways = len(cfg['ways']) if type(cfg['ways']) is not int else cfg['ways']
        if num_ways > 0:
            unique_classes, mapped_targets = torch.unique(labels, sorted=True, return_inverse=True)
            labels = mapped_targets

        torch.save(features, cfg['cache_dir'] + "/" + split + "_f.pt")
        torch.save(labels, cfg['cache_dir'] + "/" + split + "_l.pt")
   
    else:
        features = torch.load(cfg['cache_dir'] + "/" + split + "_f.pt")
        labels = torch.load(cfg['cache_dir'] + "/" + split + "_l.pt")
    
    return features, labels


def search_hp(cfg, cache_keys, cache_values, features, labels, clip_weights, adapter=None):

    if cfg['search_hp'] == True:
    
        beta_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in range(cfg['search_step'][0])]
        alpha_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in range(cfg['search_step'][1])]

        best_acc = 0
        best_beta, best_alpha = 0, 0

        for beta in beta_list:
            for alpha in alpha_list:
                if adapter:
                    affinity = adapter(features)
                else:
                    affinity = features @ cache_keys

                cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
                clip_logits = 100. * features @ clip_weights
                tip_logits = clip_logits + cache_logits * alpha
                acc = cls_acc(tip_logits, labels)
            
                if acc > best_acc:
                    # print("New best setting, beta: {:.2f}, alpha: {:.2f}; accuracy: {:.2f}".format(beta, alpha, acc))
                    best_acc = acc
                    best_beta = beta
                    best_alpha = alpha

        print("\nAfter searching, the best val accuracy: {:.2f}.\n".format(best_acc))

    return best_beta, best_alpha

def search_hp_dawa(cfg, gda_params, cache_keys, cache_values, features, labels, clip_weights, 
                  tip_adapter=None, gda_adapter=None):
    """
    Grid search for hyper-parameter alpha & beta : fused adapter
    
    Args:
        cfg: 配置字典
        gda_params: GDA参数字典，包含'W'和'b'
        tip_logits: TIP logits，形状为 (N, C)
        features: 验证集特征，形状为 (N, D)
        labels: 验证集标签，形状为 (N,)
        clip_weights: CLIP分类器权重
        
    Returns:
        best_alpha: 最优的alpha参数
    """
    
    if cfg['search_hp'] == True:
    
        beta_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in range(cfg['search_step'][0])]
        alpha_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in range(cfg['search_step'][1])]

        best_acc = 0
        best_beta, best_alpha = 0, 0
        gamma = 1.0 # 2. /(1. + torch.exp(torch.tensor(-2.5*(int(cfg["shots"]) - 5.))))

        clip_logits = 100. * features @ clip_weights
        if gda_adapter:
            gda_affinity = gda_adapter(features)
            gda_logits = gda_affinity @ gda_params['gda_one_hot']
        else:
            gda_logits = (features @ gda_params['W'] + gda_params['b']) @ gda_params['gda_one_hot']

        if tip_adapter:
            tip_affinity = tip_adapter(features)
        else:
            tip_affinity = features @ cache_keys

        for beta in beta_list:
            for alpha in alpha_list:

                tip_logits = ((-1) * (beta - beta * tip_affinity)).exp() @ cache_values

                cache_logits = logits_fuse(clip_logits, [tip_logits, gda_logits], gamma)
                fuse_logits = clip_logits + cache_logits * alpha

                acc = cls_acc(fuse_logits, labels)
            
                if acc > best_acc:
                    # print("New best setting, beta: {:.2f}, alpha: {:.2f}; accuracy: {:.2f}".format(beta, alpha, acc))
                    best_acc = acc
                    best_beta = beta
                    best_alpha = alpha

        print("Best setting, beta: {:.2f}, alpha: {:.2f}".format(best_beta, best_alpha))
        print("\nAfter searching, the best val accuracy: {:.2f}.\n".format(best_acc))

    else:
        best_beta = cfg['search_scale'][0]
        best_alpha = cfg['search_scale'][1]
        
    return best_beta, best_alpha


def compute_entropy(logits, temperature=1.0):
    """
    Transductive loss.
    Compute the entropy of the logits.
    """
    log_probs = F.log_softmax(logits / temperature, dim=1)
    probs = torch.exp(log_probs)  
    entropy = -torch.sum(probs * log_probs, dim=1)  
    return entropy.mean()


def contrast_loss(prototype, feature, label, cache_values):
    """
    Contrast loss for prototypes.
    Args:
        prototype: (num_classes * proxy_per_cls, feature_dim)
        feature: (batch_size, feature_dim)
        label: (batch_size,)
        cache_values: (num_classes * proxy_per_cls, num_classes)
    Returns:
        loss: scalar tensor representing the pull loss
    """
    batch_size = label.size(0)
    num_classes = cache_values.size(1)
    proxy_per_cls = cache_values.size(0) // num_classes
    sim_matrix = torch.mm(F.normalize(feature, dim=-1), F.normalize(prototype, dim=-1).t().contiguous())    # [batch_size, num_classes * proxy_per_cls]
    mask = torch.zeros_like(sim_matrix)
    for i in range(batch_size):
        for j in range(proxy_per_cls):
            mask[i][proxy_per_cls * label[i] + j] = 1

    unsim = sim_matrix.masked_select((torch.ones_like(mask) - mask).bool()).mean()
    sim = - sim_matrix.masked_select(mask.bool()).mean()

    return unsim + sim


def knowledge_clip_weights(classnames, gpt_prompts, clip_model):
    with torch.no_grad():
        clip_weights = []
        for classname in classnames:
            # Tokenize the prompts
            # classname = classname.replace('_', ' ')
            texts = []
            for t in gpt_prompts[classname]:
                texts.append(t)
            texts = clip.tokenize(texts).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    return clip_weights


def estimate_gda_parameters(vecs, labels, num_classes):
    """
    W = cov_inv * mu
    b = log(pi) -  mu.T * cov_inv * mus

    Args:
        vecs (torch.Tensor): 特征向量，形状为 (N, D)，其中N是样本数，D是特征维度
        labels (torch.Tensor): 对应的标签，形状为 (N,)
        num_classes (int): 类别数

    Returns:
        W (torch.Tensor): 权重矩阵，形状为 (num_classes, D)
        b (torch.Tensor): 偏置向量，形状为 (num_classes,)
    """
    
    # 计算每个类别的均值向量 (normal distribution)
    mus = torch.cat([vecs[labels == i].mean(dim=0, keepdim=True) for i in range(num_classes)])

    # KS Estimator - 计算中心化的特征向量
    center_vecs = torch.cat([vecs[labels == i] - mus[i].unsqueeze(0) for i in range(num_classes)])
    
    # 计算正则化协方差矩阵的逆矩阵
    sample_cov = center_vecs.T.cov()
    regularized_cov = (center_vecs.shape[0] - 1) * sample_cov + sample_cov.trace() * torch.eye(center_vecs.shape[1]).cuda()
    cov_inv = center_vecs.shape[1] * torch.linalg.pinv(regularized_cov)

    # 设置均匀先验概率
    ps = torch.ones(num_classes).cuda() / num_classes
    
    # 计算权重矩阵W和偏置向量b
    W = torch.einsum('nd, dc -> cn', mus, cov_inv)
    b = ps.log() - torch.einsum('nd, dc, nc -> n', mus, cov_inv, mus) / 2
    
    return W, b


# clip zero_shot as baseline
def logits_fuse(base_logtis, logits, gamma=1.0, normalize='softmax'):
    # normalize logits
    softmax_fun = nn.Softmax(dim=1)
    def normalize_logits(logtis):
        if normalize == 'softmax':
            logtis = softmax_fun(logtis)
        elif normalize =='linear':
            logtis /= torch.norm(logtis, p=2, dim=1, keepdim=True)
        elif normalize == 'mean':
            logits_std = torch.std(logtis, dim=1, keepdim=True)
            logits_mean = torch.mean(logtis, dim=1, keepdim=True)
            logtis = (logtis - logits_mean) / logits_std
        else:
            raise("error normalize!")
        return logtis
    
    base_logtis = normalize_logits(base_logtis)
    tip_logits = normalize_logits(logits[0])
    gda_logits = normalize_logits(logits[1]) if len(logits) > 1 else None

    # Compute similarity
    similarity_matrix = []
    normalize_logits = []
    for cur_logit in [tip_logits, gda_logits]:
        cur_similarity = cur_logit * base_logtis
        cur_similarity = torch.sum(cur_similarity, dim=1, keepdim=True)
        similarity_matrix.append(cur_similarity)
        normalize_logits.append(cur_logit)
    similarity_matrix[1] = similarity_matrix[1] * gamma if gda_logits is not None else None
    similarity_matrix = torch.stack(similarity_matrix, dim=-2)
    weighted_similarity = softmax_fun(similarity_matrix / 0.1)
    normalize_logits = torch.stack(normalize_logits, dim=-2)
    result_logits = torch.sum(normalize_logits * weighted_similarity, dim=1)

    return result_logits
