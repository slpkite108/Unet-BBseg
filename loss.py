import torch
from utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#intersection over union
def intersection_over_union(pred_coord, target_coord): #(ytl,xtl,yrb,xrb)
    
    x1 = torch.max(pred_coord[:,:,1], target_coord[:,:,1])
    y1 = torch.max(pred_coord[:,:,0], target_coord[:,:,0])
    x2 = torch.min(pred_coord[:,:,3], target_coord[:,:,3])
    y2 = torch.min(pred_coord[:,:,2], target_coord[:,:,2])
    
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    #intersection[intersection <= 0.]=0.

    box1_area = abs((pred_coord[:,:,3] - pred_coord[:,:,1]) * (pred_coord[:,:,2] - pred_coord[:,:,0]))
    box2_area = abs((target_coord[:,:,3] - target_coord[:,:,1]) * (target_coord[:,:,2] - target_coord[:,:,0]))
    
    iou = intersection / (box1_area + box2_area - intersection + 1e-6)
    iou[box2_area == 0] = 1-(box1_area[box2_area==0])/(600*600 +1e-6)#둘 다 0이면 1출력

    #print("target: ",target_coord)
    #print("iou: ",iou)
    #print("width: ",x2-x1)
    #print("height",y2-y1)
    #print("intersection: ",intersection)
    #print("box1_area:", box1_area)
    #print("box2_area", box2_area)
    #print("iou area: ",box1_area + box2_area - intersection)

    return iou

#BoundingBoxRegression

#Confidence Score
#Non-Maximum Suppression

#평가지표
#Average Precision
#meanAverge Precision

class MultiBoxLoss(torch.nn.Module):
    """
    The MultiBox loss, a loss function for object detection.

    This is a combination of:
    (1) a localization loss for the predicted locations of the boxes, and
    (2) a confidence loss for the predicted class scores.
    """

    def __init__(self, priors_cxcy, threshold=0.5, neg_pos_ratio=3, alpha=1.):
        super(MultiBoxLoss, self).__init__()
        self.priors_cxcy = priors_cxcy
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha

        self.smooth_l1 = torch.nn.SmoothL1Loss()  # *smooth* L1 loss in the paper; see Remarks section in the tutorial
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, predicted_locs, predicted_scores, boxes, labels):
        """
        Forward propagation.

        :param predicted_locs: predicted locations/boxes w.r.t the 8732 prior boxes, a tensor of dimensions (N, 8732, 4)
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 8732, n_classes)
        :param boxes: true  object bounding boxes in boundary coordinates, a list of N tensors
        :param labels: true object labels, a list of N tensors
        :return: multibox loss, a scalar
        """
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        n_classes = predicted_scores.size(2)

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(device)  # (N, 8732, 4)
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(device)  # (N, 8732)

        # For each image
        for i in range(batch_size):
            n_objects = boxes[i].size(0)
            #print("n_objects: ",n_objects)

            overlap = find_jaccard_overlap(boxes[i],
                                           self.priors_xy)  # (n_objects, 8732)
            
            # For each prior, find the object that has the maximum overlap
            overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)  # (8732)

            #print("objforeachprior[:10]: ",object_for_each_prior[:10])
            # We don't want a situation where an object is not represented in our positive (non-background) priors -
            # 1. An object might not be the best object for all priors, and is therefore not in object_for_each_prior.
            # 2. All priors with the object may be assigned as background based on the threshold (0.5).

            # To remedy this -
            # First, find the prior that has the maximum overlap for each object. overlap이 최대가 되는 label의 인덱스
            _, prior_for_each_object = overlap.max(dim=1)  # (N_o)
            #print("prior shape: ",prior_for_each_object.shape," ",prior_for_each_object)

            # Then, assign each object to the corresponding maximum-overlap-prior. (This fixes 1.) [0,1,2,...,n_objects-1] Class별 최대인 prior는 다른 class의 overlap이 높더라도 최대인 class의 label을 갖는다.
            object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(n_objects)).to(device)

            # To ensure these priors qualify, artificially give them an overlap of greater than 0.5. (This fixes 2.) class별 최대값을 1로 변경 적어도 1개 이상의 prior가 0.5 이상이 되도록 한다.
            overlap_for_each_prior[prior_for_each_object] = 1.

            # Labels for each prior
            label_for_each_prior = labels[i][object_for_each_prior]  # (8732)
            # Set priors whose overlaps with objects are less than t
            # he threshold to be background (no object)
            label_for_each_prior[overlap_for_each_prior < self.threshold] = 0  # (8732)

            # Store
            true_classes[i] = label_for_each_prior

            # Encode center-size object coordinates into the form we regressed predicted boxes to
            true_locs[i] = cxcy_to_gcxgcy(xy_to_cxcy(boxes[i][object_for_each_prior]), self.priors_cxcy)  # (8732, 4)

        # Identify priors that are positive (object/non-background)
        positive_priors = true_classes != 0  # (N, 8732)

        # LOCALIZATION LOSS

        # Localization loss is computed only over positive (non-background) priors
        loc_loss = self.smooth_l1(predicted_locs[positive_priors], true_locs[positive_priors])  # (), scalar

        # Note: indexing with a torch.uint8 (byte) tensor flattens the tensor when indexing is across multiple dimensions (N & 8732)
        # So, if predicted_locs has the shape (N, 8732, 4), predicted_locs[positive_priors] will have (total positives, 4)

        # CONFIDENCE LOSS

        # Confidence loss is computed over positive priors and the most difficult (hardest) negative priors in each image
        # That is, FOR EACH IMAGE,
        # we will take the hardest (neg_pos_ratio * n_positives) negative priors, i.e where there is maximum loss
        # This is called Hard Negative Mining - it concentrates on hardest negatives in each image, and also minimizes pos/neg imbalance

        # Number of positive and hard-negative priors per image
        n_positives = positive_priors.sum(dim=1)  # (N)
        #print('pprior',positive_priors)
        #print('n_pos',n_positives)
        n_hard_negatives = self.neg_pos_ratio * n_positives  # (N)


        # First, find the loss for all priors
        conf_loss_all = self.cross_entropy(predicted_scores.view(-1, n_classes), true_classes.view(-1))  # (N * 8732)
        conf_loss_all = conf_loss_all.view(batch_size, n_priors)  # (N, 8732)

        # We already know which priors are positive
        conf_loss_pos = conf_loss_all[positive_priors]  # (sum(n_positives))

        # Next, find which priors are hard-negative
        # To do this, sort ONLY negative priors in each image in order of decreasing loss and take top n_hard_negatives
        conf_loss_neg = conf_loss_all.clone()  # (N, 8732)
        conf_loss_neg[positive_priors] = 0.  # (N, 8732), positive priors are ignored (never in top n_hard_negatives)
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)  # (N, 8732), sorted by decreasing hardness

        hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(device)  # (N, 8732)
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)  # (N, 8732)
        conf_loss_hard_neg = conf_loss_neg[hard_negatives]  # (sum(n_hard_negatives))

        # As in the paper, averaged over positive priors only, although computed over both positive and hard-negative priors
        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / n_positives.sum().float()  # (), scalar

        if conf_loss.isnan() or loc_loss.isnan():
            print("nan!")
            exit()

        # TOTAL LOSS
        #return loc_loss
        return conf_loss + self.alpha * loc_loss