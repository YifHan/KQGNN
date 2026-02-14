from load_data import getdata
from model import GAT
import torch.optim as optim
import time
import torch
import numpy as np
import os
import dgl
from sklearn import metrics
from sklearn.cluster import KMeans
from utils import pairwise_sample
import torch.nn.functional as F


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def run_kmeans(extract_features, extract_labels, indices, args,isoPath=None):
    indices = indices.cpu().detach().numpy()

    if isoPath is not None:
        temp = torch.load(isoPath)
        temp = temp.cpu().detach().numpy()
        non_isolated_index = list(np.where(temp != 1)[0])
        indices = intersection(indices, non_isolated_index)

    labels_true = extract_labels[indices]
    X = extract_features[indices, :]
    assert labels_true.shape[0] == X.shape[0]
    n_test_tweets = X.shape[0]

    # Get the total number of classes
    n_classes = len(set(list(labels_true)))
    # print(n_classes)
    # sys.exit()

    # kmeans clustering
    kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X) 
    labels = kmeans.labels_ 
    nmi = metrics.normalized_mutual_info_score(labels_true, labels) 
    ari = metrics.adjusted_rand_score(labels_true, labels)
    ami = metrics.adjusted_mutual_info_score(labels_true, labels, average_method='arithmetic')
    print("nmi:",nmi,'ami:',ami,'ari:',ari)
    value = nmi
    global NMI
    NMI = nmi
    global AMI
    AMI = ami
    global ARI
    ARI = ari

    if args.metrics =='ari':
        print('use ari')
        value = ari
    if args.metrics=='ami':
        print('use ami')
        value = ami
    return (n_test_tweets, n_classes, value)

def evaluate(extract_features, extract_labels, indices, epoch, num_isolated_nodes, save_path, args, is_validation=True):
    message = ''
    message += '\nEpoch '
    message += str(epoch+1)
    message += '\n'

    # with isolated nodes
    n_tweets, n_classes, value = run_kmeans(extract_features, extract_labels, indices, args)
    if is_validation:
        mode = 'validation'
    else:
        mode = 'test'
    message += '\tNumber of ' + mode + ' tweets: '
    message += str(n_tweets)
    message += '\n\tNumber of classes covered by ' + mode + ' tweets: '
    message += str(n_classes)
    message += '\n\t' + mode +' '
    message += args.metrics +': '
    message += str(value)
    if num_isolated_nodes != 0:
        # without isolated nodes
        message += '\n\tWithout isolated nodes:'
        n_tweets, n_classes, value= run_kmeans(extract_features, extract_labels, indices, args,
                                              save_path + '/isolated_nodes.pt')
        message += '\tNumber of ' + mode + ' tweets: '
        message += str(n_tweets)
        message += '\n\tNumber of classes covered by ' + mode + ' tweets: '
        message += str(n_classes)
        message += '\n\t' + mode + ' value: '
        message += str(value)
    message += '\n'
    global NMI
    global AMI
    global ARI
    with open(save_path + '/evaluate.txt', 'a') as f:
        f.write(message)
        f.write('\n')
        f.write("NMI "+str(NMI)+" AMI "+str(AMI) + ' ARI '+str(ARI))
    print(message)

    all_value_save_path = "/".join(save_path.split('/')[0:-1])
    print(all_value_save_path)

    with open(all_value_save_path + '/evaluate.txt', 'a') as f:
        f.write("block "+ save_path.split('/')[-1])
        f.write(message)
        f.write('\n')
        f.write("NMI "+str(NMI)+" AMI "+str(AMI) + ' ARI '+str(ARI) + '\n')

    return value

def extract_embeddings(g, model, num_all_samples, args):
    with torch.no_grad():
        model.eval()
        indices = torch.LongTensor(np.arange(0,num_all_samples,1))
        if args.use_cuda:
            indices = indices.cuda()
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)

        dataloader = dgl.dataloading.NodeDataLoader(                
            g, 
            indices = indices,
            graph_sampler=sampler,
            batch_size=num_all_samples, 
            shuffle=False,
            drop_last=False,
            )

        for batch_id, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
            device = torch.device("cuda:{}".format(args.gpuid) if args.use_cuda else "cpu")
            blocks = [b.to(device) for b in blocks]
            img_features_batch = blocks[0].srcdata['img_features']
            text_features_batch = blocks[0].srcdata['features']
            extract_labels = blocks[-1].dstdata['labels']
            
            extract_features = model(blocks, img_features_batch, text_features_batch, flag='2')

        assert batch_id == 0
        extract_features = extract_features.data.cpu().numpy()
        extract_labels = extract_labels.data.cpu().numpy()

    return (extract_features, extract_labels)

def save_key_samples(dataloader, model, label_center_emb, key_threshold, key_samples, device):
    for batch_id, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        blocks = [b.to(device) for b in blocks]
        img_features_batch = blocks[0].srcdata['img_features']
        text_features_batch = blocks[0].srcdata['features']
        batch_labels = blocks[-1].dstdata['labels']

        pred = model(blocks, img_features_batch, text_features_batch, '2')
        pred = F.normalize(pred, 2, 1)
        rela_center_vec = F.softmax(torch.mm(pred, label_center_emb.t()), dim=1)
        entropy = torch.mul(torch.log(rela_center_vec), rela_center_vec)
        entropy = torch.sum(entropy, dim=1)
        cos_similarity_pos = F.cosine_similarity(rela_center_vec.unsqueeze(0), rela_center_vec.unsqueeze(0))
        cos_similarity_neg = F.cosine_similarity(rela_center_vec.unsqueeze(0), 1 - rela_center_vec.unsqueeze(0))
        pos_quality = cos_similarity_pos
        neg_quality = 1 - cos_similarity_neg

        key_sample_mask = (pos_quality > key_threshold) & (neg_quality > key_threshold)
        key_sample_indices = key_sample_mask.nonzero(as_tuple=True)[0]

        if len(key_sample_indices) > 0:
            key_samples['image_features'] = torch.cat([key_samples['image_features'], img_features_batch[key_sample_indices]], dim=0)
            key_samples['text_features'] = torch.cat([key_samples['text_features'], text_features_batch[key_sample_indices]], dim=0)
            key_samples['labels'] = torch.cat([key_samples['labels'], batch_labels[key_sample_indices]], dim=0)

    return key_samples

def initial_train(i, args, data_split, metrics,embedding_save_path, loss_fn, model=None):
    start_time = time.time()

    save_path_i, img_dim, in_feats, num_isolated_nodes, g, labels, train_indices, validation_indices, test_indices = getdata(
        embedding_save_path, data_split, i, args)
    # Construct the initial model   
    if model is None:  
        model = GAT(img_dim, in_feats, args.hidden_dim, args.out_dim, args.num_heads, args.use_residual)
    if args.use_cuda:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    message = "\n------------ Start initial training ------------\n"
    print(message)
    with open(save_path_i + '/log.txt', 'a') as f:
        f.write(message)

    key_samples = {
        'text_features': torch.empty(0, in_feats).cuda() if args.use_cuda else torch.empty(0, in_feats),
        'image_features': torch.empty(0, img_dim).cuda() if args.use_cuda else torch.empty(0, img_dim),
        'labels': torch.empty(0, dtype=torch.long).cuda() if args.use_cuda else torch.empty(0, dtype=torch.long)
    }

    best_vali_value = 1e-9
    best_epoch = 0
    wait = 0
    all_vali_value = []
    seconds_train_batches = []
    mins_train_epochs = []
    for epoch in range(args.n_epochs):
        start_epoch = time.time()
        losses = []
        total_loss = 0
        for metric in metrics:
            metric.reset()
        
        extract_features, extract_labels = extract_embeddings(g, model, len(labels), args)
        label_center = {}
        for l in set(extract_labels):
            l_indices = np.where(extract_labels==l)[0]
            l_feas = extract_features[l_indices]
            l_cen = np.mean(l_feas,0)
            label_center[l] = l_cen

        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
        dataloader = dgl.dataloading.NodeDataLoader(
            g, train_indices, sampler,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            )

        for batch_id, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
            device = torch.device("cuda:{}".format(args.gpuid) if args.use_cuda else "cpu")
            blocks = [b.to(device) for b in blocks]
            img_features_batch = blocks[0].srcdata['img_features']
            text_features_batch = blocks[0].srcdata['features']
            batch_labels = blocks[-1].dstdata['labels']

            print(f"Batch {batch_id}:")
            print(f"img_features_batch shape: {img_features_batch.shape}")
            print(f"text_features_batch shape: {text_features_batch.shape}")
            print(f"batch_labels shape: {batch_labels.shape}")

            start_batch = time.time()
            model.train()
            # forward
            pred = model(blocks, img_features_batch, text_features_batch, '2') 

            dis = torch.empty([0, 1]).cuda()
            for l in set(batch_labels.cpu().data.numpy()):
                label_indices = torch.where(batch_labels==l)
                l_center = torch.FloatTensor(label_center[l]).cuda()
                dis_l = (pred[label_indices] - l_center).pow(2).sum(1).unsqueeze(-1)
                dis = torch.cat([dis,dis_l],0)

            if args.add_pair:
                pairs, pair_labels, pair_matrix = pairwise_sample(pred, batch_labels)
                if args.use_cuda:
                    pairs = pairs.cuda()
                    pair_matrix = pair_matrix.cuda()

                pos_indices = torch.where(pair_labels > 0)
                neg_indices = torch.where(pair_labels == 0)
                if neg_indices[0].numel() > 0:
                    neg_ind = torch.randint(0, neg_indices[0].shape[0], [5*pos_indices[0].shape[0]]).cuda()
                    neg_dis = (pred[pairs[neg_indices[0][neg_ind], 0]] - pred[pairs[neg_indices[0][neg_ind], 1]]).pow(2).sum(1).unsqueeze(-1)
                else:
                    continue
                pos_dis = (pred[pairs[pos_indices[0], 0]] - pred[pairs[pos_indices[0], 1]]).pow(2).sum(1).unsqueeze(-1)
                pos_dis = torch.cat([pos_dis]*5,0)
                pairs_indices = torch.where(torch.clamp(pos_dis + args.a - neg_dis, min=0.0)>0)
                loss = torch.mean(torch.clamp(pos_dis + args.a - neg_dis, min=0.0)[pairs_indices[0]]) 

                label_center_emb = torch.FloatTensor(np.array(list(label_center.values()))).cuda()
                pred = F.normalize(pred, 2, 1)
                pair_out = torch.mm(pred,pred.t())
                if args.add_ort:
                    pair_loss = (pair_matrix - pair_out).pow(2).mean()
                    print("pair loss:",loss,"pair orthogonal loss:  ",100*pair_loss)
                    loss += 100 * pair_loss

            losses.append(loss.item())
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_seconds_spent = time.time() - start_batch
            seconds_train_batches.append(batch_seconds_spent)

        total_loss /= (batch_id + 1)
        message = 'Epoch: {}/{}. Average loss: {:.4f}'.format(epoch + 1, args.n_epochs, total_loss)
        for metric in metrics:
            message += '\t{}: {:.4f}'.format(metric.name(), metric.value())
        mins_spent = (time.time() - start_epoch) / 60
        message += '\nThis epoch took {:.2f} mins'.format(mins_spent)
        message += '\n'
        print(message)
        with open(save_path_i + '/log.txt', 'a') as f:
            f.write(message)
        mins_train_epochs.append(mins_spent)

        extract_features, extract_labels = extract_embeddings(g, model, len(labels), args)
        np.save(save_path_i + '/features_' + str(epoch) + '.npy', extract_features)
        np.save(save_path_i + '/labels_' + str(epoch) + '.npy', extract_labels)

        validation_value = evaluate(extract_features, extract_labels, validation_indices, epoch, num_isolated_nodes,
                                  save_path_i, args, True)
        all_vali_value.append(validation_value)

        # Early stop
        if validation_value > best_vali_value:
            best_vali_value = validation_value
            best_epoch = epoch
            wait = 0
            # Save model
            model_path = save_path_i + '/models'
            if (epoch == 0) and (not os.path.isdir(model_path)):
                os.mkdir(model_path)
            p = model_path + '/best.pt'
            torch.save(model.state_dict(), p)

        else:
            wait += 1
        if wait == args.patience:
            print('Saved all_mins_spent')
            print('Early stopping at epoch ', str(epoch))
            print('Best model was at epoch ', str(best_epoch))
            break

    np.save(save_path_i + '/all_vali_value.npy', np.asarray(all_vali_value))
    np.save(save_path_i + '/mins_train_epochs.npy', np.asarray(mins_train_epochs))
    print('Saved mins_train_epochs.')
    np.save(save_path_i + '/seconds_train_batches.npy', np.asarray(seconds_train_batches))
    print('Saved seconds_train_batches.')
    best_model_path = save_path_i + '/models/best.pt'
    model.load_state_dict(torch.load(best_model_path))
    print("Best model loaded.")

    extract_features, extract_labels = extract_embeddings(g, model, len(labels), args)
    label_center = {}
    for l in set(extract_labels):
        l_indices = np.where(extract_labels == l)[0]
        l_feas = extract_features[l_indices]
        l_cen = np.mean(l_feas, 0)
        label_center[l] = l_cen
    label_center_emb = torch.FloatTensor(np.array(list(label_center.values()))).cuda()
    torch.save(label_center_emb,save_path_i + '/models/center.pth')

    # Screen and save high-quality samples to the key instance.
    key_samples = save_key_samples(dataloader, model, label_center_emb, 0.8, key_samples, device)

    # Save key samples
    torch.save(key_samples, save_path_i + '/key_samples.pth')

    elapsed = time.time() - start_time
    with open(embedding_save_path + '/evaluate.txt', 'a') as f:
        f.write(f"Initial train block {i} time: {elapsed:.2f} seconds\n")
    print(f"Initial train block {i} time: {elapsed:.2f} s")

    if args.add_pair:
        return model, label_center_emb, key_samples
    else:
        return model, key_samples

def continue_train(i, data_split, metrics, embedding_save_path, loss_fn,
                   model, label_center_emb, key_samples, args):
    start_time = time.time()

    save_path_i, img_dim, in_feats, num_isolated_nodes, g, labels, test_indices = getdata(
        embedding_save_path, data_split, i, args)

    device = torch.device("cuda:{}".format(args.gpuid) if args.use_cuda else "cpu")

    if not hasattr(args, "qt_init"): args.qt_init = 0.7
    if not hasattr(args, "qt_quantile"): args.qt_quantile = 0.70  
    if not hasattr(args, "th_ema"): args.th_ema = 0.90
    if not hasattr(args, "warmup_epochs"): args.warmup_epochs = 1

    # top 5%
    if not hasattr(args, "key_ratio"): args.key_ratio = 0.05       

    # fallback
    if not hasattr(args, "th_min"): args.th_min = 0.20
    if not hasattr(args, "th_max"): args.th_max = 0.99
    if not hasattr(args, "relax_step"): args.relax_step = 0.03     
    if not hasattr(args, "min_hq_keep"): args.min_hq_keep = 1

    def _clamp(x, lo, hi):
        return max(lo, min(hi, float(x)))

    def update_qt_ema(q_tensor, qt_ema):
        qt_new = torch.quantile(q_tensor.detach(), args.qt_quantile).item()
        qt_ema = args.th_ema * qt_ema + (1 - args.th_ema) * qt_new
        return _clamp(qt_ema, args.th_min, args.th_max)

    def top_ratio_threshold(q_tensor, ratio):
        n = q_tensor.numel()
        if n == 0:
            return 1.0
        k = max(1, int(ratio * n))
        return torch.topk(q_tensor.detach(), k, largest=True).values.min().item()

    # quick eval only branch
    if i % 1 != 0:
        extract_features, extract_labels = extract_embeddings(g, model, len(labels), args)
        _ = evaluate(extract_features, extract_labels, test_indices, 0, num_isolated_nodes, save_path_i, args, True)
        return model, key_samples

    extract_features, extract_labels = extract_embeddings(g, model, len(labels), args)
    _ = evaluate(extract_features, extract_labels, test_indices, -1, num_isolated_nodes, save_path_i, args, True)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    message = "\n------------ Start fine tuning ------------\n"
    print(message)
    with open(save_path_i + '/log.txt', 'a') as f:
        f.write(message)

    seconds_train_batches = []
    mins_train_epochs = []
    best_test_value = float('-inf')
    best_model_state = None

    new_key_samples = {'text_features': [], 'image_features': [], 'labels': []}

    qt_ema = float(args.qt_init)

    for epoch in range(args.finetune_epochs):
        start_epoch = time.time()
        total_loss = 0.0

        for metric in metrics:
            metric.reset()

        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
        dataloader = dgl.dataloading.NodeDataLoader(
            g, test_indices, sampler,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
        )

        # key sample dataloader
        if len(key_samples['text_features']) > 0:
            key_sample_dataset = torch.utils.data.TensorDataset(
                key_samples['image_features'], key_samples['text_features'], key_samples['labels']
            )
            key_sample_dataloader = torch.utils.data.DataLoader(
                key_sample_dataset, batch_size=args.batch_size, shuffle=True
            )
            key_sample_iterator = iter(key_sample_dataloader)
        else:
            key_sample_dataloader = None
            key_sample_iterator = None

        epoch_pairs_total = 0
        epoch_hq_total = 0
        epoch_key_total = 0
        epoch_qt_used_sum = 0.0
        epoch_kt_used_sum = 0.0
        epoch_batches = 0

        for batch_id, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
            start_batch = time.time()
            model.train()
            label_center_emb = label_center_emb.to(device)

            blocks = [b.to(device) for b in blocks]
            img_features_batch = blocks[0].srcdata['img_features']
            text_features_batch = blocks[0].srcdata['features']
            batch_labels = blocks[-1].dstdata['labels']

            # sample key batch
            if key_sample_iterator is not None:
                try:
                    key_img_features_batch, key_text_features_batch, key_labels_batch = next(key_sample_iterator)
                except StopIteration:
                    key_sample_iterator = iter(key_sample_dataloader)
                    key_img_features_batch, key_text_features_batch, key_labels_batch = next(key_sample_iterator)

                key_img_features_batch = key_img_features_batch.to(device)
                key_text_features_batch = key_text_features_batch.to(device)
                key_labels_batch = key_labels_batch.to(device)

                # concat into current batch
                img_features_batch = torch.cat([img_features_batch, key_img_features_batch], dim=0)
                text_features_batch = torch.cat([text_features_batch, key_text_features_batch], dim=0)
                batch_labels = torch.cat([batch_labels, key_labels_batch], dim=0)
            else:
                key_img_features_batch = None
                key_text_features_batch = None

            pred = model(blocks, img_features_batch, text_features_batch,
                         key_img_features_batch, key_text_features_batch, flag='1')
            pred = F.normalize(pred, 2, 1)

            rela_center_vec = F.softmax(torch.mm(pred, label_center_emb.t()), dim=1)
            entropy = torch.mul(torch.log(rela_center_vec), rela_center_vec)
            entropy = torch.sum(entropy, dim=1)

            _, old_indices = torch.topk(entropy.reshape(-1), int(entropy.shape[0]/2), largest=True)
            _, novel_indices = torch.topk(entropy.reshape(-1), int(entropy.shape[0]/2), largest=False)

            pair_matrix = torch.mm(rela_center_vec, rela_center_vec.t())

            pairs, pair_labels = pairwise_sample(F.normalize(pred, 2, 1), batch_labels, model, label_center_emb)

            if args.use_cuda:
                pairs = pairs.cuda()
                pair_labels = pair_labels.cuda()
                pair_matrix = pair_matrix.cuda()

            # top-k pos/neg mining
            k_novel = min(args.novelnum, pair_matrix[novel_indices].size(1))
            _, novel_neg_ind = torch.topk(pair_matrix[novel_indices], k_novel, 1, largest=False)
            _, novel_pos_ind = torch.topk(pair_matrix[novel_indices], k_novel, 1, largest=True)

            k_old = min(args.oldnum, pair_matrix[old_indices].size(1))
            _, old_neg_ind = torch.topk(pair_matrix[old_indices], k_old, 1, largest=False)
            _, old_pos_ind = torch.topk(pair_matrix[old_indices], k_old, 1, largest=True)

            old_row = torch.LongTensor([[j] * k_old for j in old_indices]).reshape(-1).to(device)
            novel_row = torch.LongTensor([[j] * k_novel for j in novel_indices]).reshape(-1).to(device)
            row = torch.cat([old_row, novel_row], dim=0)

            neg_ind = torch.cat([old_neg_ind.reshape(-1), novel_neg_ind.reshape(-1)], dim=0).to(device)
            pos_ind = torch.cat([old_pos_ind.reshape(-1), novel_pos_ind.reshape(-1)], dim=0).to(device)

            neg_distances = (pred[row] - pred[neg_ind]).pow(2).sum(1).unsqueeze(-1)
            pos_distances = (pred[row] - pred[pos_ind]).pow(2).sum(1).unsqueeze(-1)

            rsd_vec_i = rela_center_vec[row]
            rsd_vec_i_pos = rela_center_vec[pos_ind]
            rsd_vec_i_neg = rela_center_vec[neg_ind]

            cos_similarity_pos = F.cosine_similarity(rsd_vec_i.unsqueeze(0), rsd_vec_i_pos.unsqueeze(0))
            cos_similarity_neg = F.cosine_similarity(rsd_vec_i.unsqueeze(0), rsd_vec_i_neg.unsqueeze(0))

            pos_quality = cos_similarity_pos
            neg_quality = 1 - cos_similarity_neg

            q = torch.minimum(pos_quality, neg_quality)  

            if epoch >= args.warmup_epochs and q.numel() > 0:
                qt_ema = update_qt_ema(q, qt_ema)

            quality_threshold = qt_ema
            key_threshold = top_ratio_threshold(q, args.key_ratio)

            high_quality_mask = (q >= quality_threshold)
            key_sample_mask = (q >= key_threshold)

            if high_quality_mask.sum().item() < args.min_hq_keep and q.numel() > 0:
                quality_threshold = _clamp(quality_threshold - args.relax_step, args.th_min, args.th_max)
                high_quality_mask = (q >= quality_threshold)

            high_quality_indices = high_quality_mask.nonzero(as_tuple=True)[0]
            key_sample_indices = key_sample_mask.nonzero(as_tuple=True)[0]

            epoch_pairs_total += int(q.numel())
            epoch_hq_total += int(high_quality_indices.numel())
            epoch_key_total += int(key_sample_indices.numel())
            epoch_qt_used_sum += float(quality_threshold)
            epoch_kt_used_sum += float(key_threshold)
            epoch_batches += 1

            # Save key samples
            if key_sample_indices.numel() > 0:
                new_key_samples['image_features'].append(img_features_batch[key_sample_indices].detach().cpu())
                new_key_samples['text_features'].append(text_features_batch[key_sample_indices].detach().cpu())
                new_key_samples['labels'].append(batch_labels[key_sample_indices].detach().cpu())

            if high_quality_indices.numel() > 0:
                filtered_pos_quality = pos_quality[high_quality_indices]
                filtered_neg_quality = neg_quality[high_quality_indices]

                filtered_quality_weighted = torch.minimum(filtered_pos_quality, filtered_neg_quality)

                filtered_pos_distances = pos_distances[high_quality_indices]
                filtered_neg_distances = neg_distances[high_quality_indices]

                p_loss = torch.clamp(filtered_pos_distances + args.a - filtered_neg_distances, min=0.0)
                loss = torch.mean(filtered_quality_weighted * p_loss)

                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
            else:
                print("No high-quality samples in this batch (even after relax)")

            seconds_train_batches.append(time.time() - start_batch)

        avg_loss = total_loss / max(1, (batch_id + 1))
        mins_spent = (time.time() - start_epoch) / 60.0
        mins_train_epochs.append(mins_spent)

        high_quality_ratio = epoch_hq_total / max(1, epoch_pairs_total)
        key_ratio_real = epoch_key_total / max(1, epoch_pairs_total)
        avg_qt_used = epoch_qt_used_sum / max(1, epoch_batches)
        avg_kt_used = epoch_kt_used_sum / max(1, epoch_batches)

        message = f"Epoch: {epoch+1}/{args.finetune_epochs}. Average loss: {avg_loss:.4f}"
        message += f"\nThis epoch took {mins_spent:.2f} mins"
        message += (
            f"\n[AdaptiveStats] avg_quality_th={avg_qt_used:.4f} (qt_ema={qt_ema:.4f}) "
            f"| avg_key_th={avg_kt_used:.4f} | high_quality_ratio={high_quality_ratio:.4f} "
            f"| key_ratio_real={key_ratio_real:.4f} (target={args.key_ratio})"
            f"\n"
        )

        print(message)
        with open(save_path_i + '/log.txt', 'a') as f:
            f.write(message)

        extract_features, extract_labels = extract_embeddings(g, model, len(labels), args)
        test_value = evaluate(extract_features, extract_labels, test_indices, epoch, num_isolated_nodes, save_path_i, args, True)
        torch.cuda.empty_cache()

        if test_value > best_test_value:
            best_test_value = test_value
            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if best_model_state is not None:
            model.load_state_dict(best_model_state, strict=True)
            model.to(device)

    if len(new_key_samples['text_features']) > 0:
        new_key_text_features = torch.cat(new_key_samples['text_features'], dim=0)
        new_key_img_features = torch.cat(new_key_samples['image_features'], dim=0)
        new_key_labels = torch.cat(new_key_samples['labels'], dim=0)

        if len(key_samples['text_features']) > 0:
            existing_text_features = key_samples['text_features'].detach().cpu().numpy()
            new_text_features = new_key_text_features.detach().cpu().numpy()

            is_duplicate = np.isin(new_text_features, existing_text_features).all(axis=1)
            non_duplicate_indices = np.where(~is_duplicate)[0]

            if len(non_duplicate_indices) > 0:
                key_samples['text_features'] = torch.cat(
                    [key_samples['text_features'], new_key_text_features[non_duplicate_indices].to(device)], dim=0
                )
                key_samples['image_features'] = torch.cat(
                    [key_samples['image_features'], new_key_img_features[non_duplicate_indices].to(device)], dim=0
                )
                key_samples['labels'] = torch.cat(
                    [key_samples['labels'], new_key_labels[non_duplicate_indices].to(device)], dim=0
                )
        else:
            key_samples['text_features'] = new_key_text_features.to(device)
            key_samples['image_features'] = new_key_img_features.to(device)
            key_samples['labels'] = new_key_labels.to(device)

    # save model
    model_path = save_path_i + '/models'
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    torch.save(model.state_dict(), model_path + '/finetune.pt')
    print('finetune model saved.')

    # save time logs
    np.save(save_path_i + '/mins_train_epochs.npy', np.asarray(mins_train_epochs))
    np.save(save_path_i + '/seconds_train_batches.npy', np.asarray(seconds_train_batches))

    elapsed = time.time() - start_time
    with open(embedding_save_path + '/evaluate.txt', 'a') as f:
        f.write(f"Continue train block {i} time: {elapsed:.2f} seconds\n")
    print(f"Continue train block {i} time: {elapsed:.2f} s")

    return model, key_samples
