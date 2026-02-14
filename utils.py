import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from itertools import combinations


class TripletLoss(nn.Module):

    def __init__(self, margin, triplet_selector):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):
        triplets = self.triplet_selector.get_triplets(embeddings, target)
        
        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1) 
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean(), len(triplets)


def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix


class TripletSelector:

    def __init__(self):
        pass

    def get_triplets(self, embeddings, labels):
        raise NotImplementedError


class FunctionNegativeTripletSelector(TripletSelector):

    def __init__(self, margin, negative_selection_fn, cpu=True):
        super(FunctionNegativeTripletSelector, self).__init__()
        self.cpu = cpu
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn

    def get_triplets(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = pdist(embeddings)
        distance_matrix = distance_matrix.cpu()

        labels = labels.cpu().data.numpy()
        triplets = []

        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(combinations(label_indices, 2)) 
            anchor_positives = np.array(anchor_positives)

            ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]
            for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
                loss_values = ap_distance - distance_matrix[
                    torch.LongTensor(np.array([anchor_positive[0]])), torch.LongTensor(negative_indices)] + self.margin
                loss_values = loss_values.data.cpu().numpy()
                hard_negative = self.negative_selection_fn(loss_values)
                if hard_negative is not None:
                    hard_negative = negative_indices[hard_negative]
                    triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])

        if len(triplets) == 0:
            triplets.append([anchor_positive[0], anchor_positive[1], negative_indices[0]])

        triplets = np.array(triplets)

        return torch.LongTensor(triplets)

def random_hard_negative(loss_values):
    hard_negatives = np.where(loss_values > 0)[0]
    return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None

def hardest_negative(loss_values):
    hard_negative = np.argmax(loss_values)
    return hard_negative if loss_values[hard_negative] > 0 else None


def HardestNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                              negative_selection_fn=hardest_negative,
                                                                                              cpu=cpu)


def RandomNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                             negative_selection_fn=random_hard_negative,
                                                                                             cpu=cpu)



def pairwise_sample(embeddings, labels=None, model=None, label_center_emb=None):
    if model == None:
        labels = labels.cpu().data.numpy()
        indices = np.arange(0,len(labels),1)
        pairs = np.array(list(combinations(indices, 2))) 
        pair_labels = (labels[pairs[:,0]]==labels[pairs[:,1]])
        pair_matrix = np.eye(len(labels))
        ind = np.where(pair_labels) 
        pair_matrix[pairs[ind[0],0],pairs[ind[0],1]] = 1
        pair_matrix[pairs[ind[0],1], pairs[ind[0],0]] = 1  

        return torch.LongTensor(pairs), torch.LongTensor(pair_labels.astype(int)),torch.LongTensor(pair_matrix)

    else:
        rela_center_vec = F.softmax(torch.mm(embeddings,label_center_emb.t()), dim=1)
        indices = np.arange(0, embeddings.shape[0], 1)
        pairs = np.array(list(combinations(indices, 2)))
        pseudo_labels = []

        for pair in pairs:
            rsd_vector_i = rela_center_vec[pair[0]]
            rsd_vector_j = rela_center_vec[pair[1]]
            cos_similarity = F.cosine_similarity(rsd_vector_i.unsqueeze(0), rsd_vector_j.unsqueeze(0))
            pseudo_label = 1 if cos_similarity > 0.5 else 0
            pseudo_labels.append(pseudo_label)

        pseudo_labels = torch.tensor(pseudo_labels, dtype=torch.long)
        return torch.LongTensor(pairs), pseudo_labels