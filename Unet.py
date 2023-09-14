import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from math import sqrt
from utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class res_conv(nn.Module):

    def __init__(self, input_channels, output_channels, down=True):
        super(res_conv, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(output_channels),
                                   nn.ReLU(inplace = True),
                                   nn.Dropout(0.5),
                                 )
        self.conv2 = nn.Sequential(nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(output_channels),
                                   nn.ReLU(inplace = True),
                                   nn.Dropout(0.5),
                                  )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)+x1
        return x2


class start_conv(nn.Module):

    def __init__(self, input_channels, output_channels):
        super(start_conv, self).__init__()
        self.conv = res_conv(input_channels, output_channels)

    def forward(self, x):
        x = self.conv(x)
        return x


class down_conv(nn.Module):

    def __init__(self, input_channels, output_channels):
        super(down_conv, self).__init__()
        self.conv = nn.Sequential(nn.MaxPool2d(2),
                                  res_conv(input_channels, output_channels),
                                 )

    def forward(self,x):
        x = self.conv(x)
        return x



class up_conv(nn.Module):

    def __init__(self, input_channels, output_channels):
        super(up_conv, self).__init__()
        self.up = nn.ConvTranspose2d(input_channels//2, input_channels//2, kernel_size=2, stride=2)
        self.conv = res_conv(input_channels, output_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff1 = x2.shape[2]-x1.shape[2]
        diff2 = x2.shape[3]-x1.shape[3]
        x1 = F.pad(x1, pad=(diff1//2, diff1-diff1//2, diff2//2, diff2-diff2//2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class stop_conv(nn.Module):

    def __init__(self, input_channels, output_channels):
        super(stop_conv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(input_channels, output_channels, kernel_size=1),
                                 nn.Sigmoid())


    def forward(self, x):
        x = self.conv(x)
        return x

class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()

        self.inc = start_conv(1, 64)
        self.down1 = down_conv(64, 128)
        self.down2 = down_conv(128, 256)
        self.down3 = down_conv(256, 512)
        self.down4 = down_conv(512, 512)
        self.up1 = up_conv(1024, 256)
        self.up2 = up_conv(512, 128)
        self.up3 = up_conv(256, 64)
        self.up4 = up_conv(128, 64)
        self.outc = stop_conv(64, 1)

    def forward(self, x):

        xin = self.inc(x)
        xd1 = self.down1(xin)
        xd2 = self.down2(xd1)
        xd3 = self.down3(xd2)
        xd4 = self.down4(xd3)
        xu1 = self.up1(xd4, xd3)#(N,256,75,75)
        xu2 = self.up2(xu1, xd2)#(N,128,150,150)
        xu3 = self.up3(xu2, xd1)#(N,64,300,300)
        xu4 = self.up4(xu3, xin)#(N,64,600,600)
        xout = self.outc(xu4)#(N,1,600,600)
        #print(xu1.shape,xu2.shape, xu3.shape, xu4.shape, xout.shape)
        return xout

class UnetLoc(nn.Module):
    def __init__(self,n_classes):
        super(UnetLoc, self).__init__()

        self.n_classes = n_classes

        self.base = Unet()
        self.aux_convs = AuxiliaryConvolutions()
        self.pred_convs = PredictionConvolutions(n_classes)

        self.priors_cxcy = self.create_prior_boxes()

    def forward(self, x):

        conv_feat = self.base(x)
        conv0_feat, conv1_feat, conv2_feat, conv3_feat, conv4_feat, conv5_feat = self.aux_convs(conv_feat)
        loc, score = self.pred_convs(conv0_feat, conv1_feat, conv2_feat, conv3_feat, conv4_feat, conv5_feat)
        return loc, score
    
    def create_prior_boxes(self):
        """
        Create the 8732 prior (default) boxes for the SSD300, as defined in the paper.

        :return: prior boxes in center-size coordinates, a tensor of dimensions (8732, 4)
        """
        fmap_dims = {'conv0': 25, 
                        'conv1': 13,
                        'conv2': 7,
                        'conv3': 5,
                        'conv4': 3,
                        'conv5': 1}

        obj_scales = {'conv0': 0.15,
                        'conv1': 0.25,
                        'conv2': 0.375,
                        'conv3': 0.55,
                        'conv4': 0.725,
                        'conv5': 0.9}

        aspect_ratios = {'conv0': [1., 2., 3., 0.5, .333],
                            'conv1': [1., 2., 3., 0.5, .333],
                            'conv2': [1., 2., 3., 0.5, .333],
                            'conv3': [1., 2., 3., 0.5, .333],
                            'conv4': [1., 2., 0.5],
                            'conv5': [1., 2., 0.5]}

        fmaps = list(fmap_dims.keys())

        prior_boxes = []

        for k, fmap in enumerate(fmaps):
            for i in range(fmap_dims[fmap]):
                for j in range(fmap_dims[fmap]):
                    cx = (j + 0.5) / fmap_dims[fmap]
                    cy = (i + 0.5) / fmap_dims[fmap]

                    for ratio in aspect_ratios[fmap]:
                        prior_boxes.append([cx, cy, obj_scales[fmap] * sqrt(ratio), obj_scales[fmap] / sqrt(ratio)])

                        # For an aspect ratio of 1, use an additional prior whose scale is the geometric mean of the
                        # scale of the current feature map and the scale of the next feature map
                        if ratio == 1.:
                            try:
                                additional_scale = sqrt(obj_scales[fmap] * obj_scales[fmaps[k + 1]])
                            # For the last feature map, there is no "next" feature map
                            except IndexError:
                                additional_scale = 1.
                            prior_boxes.append([cx, cy, additional_scale, additional_scale])

        prior_boxes = torch.FloatTensor(prior_boxes).to(device)  # (8732, 4)
        prior_boxes.clamp_(0, 1)  # (8732, 4); this line has no effect; see Remarks section in tutorial

        return prior_boxes
    
    def detect_objects(self, predicted_locs, predicted_scores, min_score, max_overlap, top_k):
        """
        Decipher the 8732 locations and class scores (output of ths SSD300) to detect objects.

        For each class, perform Non-Maximum Suppression (NMS) on boxes that are above a minimum threshold.

        :param predicted_locs: predicted locations/boxes w.r.t the 8732 prior boxes, a tensor of dimensions (N, 8732, 4)
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 8732, n_classes)
        :param min_score: minimum threshold for a box to be considered a match for a certain class
        :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via NMS
        :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
        :return: detections (boxes, labels, and scores), lists of length batch_size
        """
        batch_size = predicted_locs.size(0) # N batch
        n_priors = self.priors_cxcy.size(0) # prior 개수
        predicted_scores = F.softmax(predicted_scores, dim=2)  # (N, 8732, n_classes)

        # bbox, label, score를 저장할 List
        all_images_boxes = list()
        all_images_labels = list()
        all_images_scores = list()

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1) #기존에 설정한 prior 개수와 prediction의 prior 개수 비교

        for i in range(batch_size):
            # gcxgcy형태의 predicted_locs를 xy형태로 decode
            decoded_locs = cxcy_to_xy(
                gcxgcy_to_cxcy(predicted_locs[i], self.priors_cxcy))  # (8732, 4), these are fractional pt. coordinates

            # Lists to store boxes and scores for this image 각 batch 단위 저장
            image_boxes = list()
            image_labels = list()
            image_scores = list()

            useJaccardIndex = True  #test용도
            
            if useJaccardIndex:
            # Check for each class
                for c in range(1, self.n_classes):
                    # Keep only predicted boxes and scores where scores for this class are above the minimum score
                    class_scores = predicted_scores[i][:, c]  # (8732)
                    #print('a',class_scores.shape)
                    #print('b',class_scores.mean())
                    #print(class_scores.max())
                    score_above_min_score = class_scores > min_score  # torch.uint8 (byte) tensor, for indexing
                    n_above_min_score = score_above_min_score.sum().item()
                    #print('c',n_above_min_score)
                    if n_above_min_score == 0:
                        continue
                    class_scores = class_scores[score_above_min_score]  # (n_qualified), n_min_score <= 8732
                    class_decoded_locs = decoded_locs[score_above_min_score]  # (n_qualified, 4)
                    # Sort predicted boxes and scores by scores
                    class_scores, sort_ind = class_scores.sort(dim=0, descending=True)  # (n_qualified), (n_min_score)
                    class_decoded_locs = class_decoded_locs[sort_ind]  # (n_min_score, 4)

                    # Find the overlap between predicted boxes
                    overlap = find_jaccard_overlap(class_decoded_locs, class_decoded_locs)  # (n_qualified, n_min_score)
                    #print('d',overlap.shape)
                    # Non-Maximum Suppression (NMS)

                    # A torch.uint8 (byte) tensor to keep track of which predicted boxes to suppress
                    # 1 implies suppress, 0 implies don't suppress
                    suppress = torch.zeros((n_above_min_score), dtype=torch.uint8).to(device)  # (n_qualified)

                    # Consider each box in order of decreasing scores
                    for box in range(class_decoded_locs.size(0)):
                        # If this box is already marked for suppression
                        if suppress[box] == 1:
                            continue

                        # Suppress boxes whose overlaps (with this box) are greater than maximum overlap
                        # Find such boxes and update suppress indices
                        suppress = torch.max(suppress, overlap[box] > max_overlap)
                        # The max operation retains previously suppressed boxes, like an 'OR' operation
                        
                        # Don't suppress this box, even though it has an overlap of 1 with itself
                        suppress[box] = 0
                    
                    #print('boxes: ',class_decoded_locs)


                    # Store only unsuppressed boxes for this class
                    image_boxes.append(class_decoded_locs[1 - suppress])
                    image_labels.append(torch.LongTensor((1 - suppress).sum().item() * [c]).to(device))
                    image_scores.append(class_scores[1 - suppress])
            else:
                max_scores, best_label = predicted_scores[i].max(dim=1)  # (8732) prior 별로 해당하는 label과 score
                print(max_scores.shape)
                print(best_label.shape)
                exit()
                

            # If no object in any class is found, store a placeholder for 'background'
            if len(image_boxes) == 0:
                image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
                image_labels.append(torch.LongTensor([0]).to(device))
                image_scores.append(torch.FloatTensor([0.]).to(device))

            # Concatenate into single tensors
            image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 4)
            image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
            image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
            n_objects = image_scores.size(0)

            # Keep only the top k objects
            if n_objects > top_k:
                image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
                image_scores = image_scores[:top_k]  # (top_k)
                image_boxes = image_boxes[sort_ind][:top_k]  # (top_k, 4)
                image_labels = image_labels[sort_ind][:top_k]  # (top_k)

            # Append to lists that store predicted boxes and scores for all images
            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)

        return all_images_boxes, all_images_labels, all_images_scores  # lists of length batch_size



class AuxiliaryConvolutions(nn.Module):
    """
    Additional convolutions to produce higher-level feature maps.
    """

    def __init__(self):
        super(AuxiliaryConvolutions, self).__init__()

        self.start_1 = nn.Conv2d(1, 128, kernel_size=3, stride=2, padding=1)
        self.start_2 = nn.Conv2d(128, 256, kernel_size=3,stride=2, padding=1)
        self.start_3 = nn.Conv2d(256, 512, kernel_size=3,stride=2, padding=1)
        self.start_4 = nn.Conv2d(512, 1024, kernel_size=4,stride=3, padding=1)
        # Auxiliary/additional convolutions on top of the VGG base
        self.conv1_1 = nn.Conv2d(1024, 256, kernel_size=1, padding=0)  # stride = 1, by default
        self.conv1_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)  # dim. reduction because stride > 1

        self.conv2_1 = nn.Conv2d(512, 128, kernel_size=1, padding=0)
        self.conv2_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # dim. reduction because stride > 1

        self.conv3_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv3_2 = nn.Conv2d(128, 256, kernel_size=3, padding=0)  # dim. reduction because padding = 0

        self.conv4_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv4_2 = nn.Conv2d(128, 256, kernel_size=3, padding=0)  # dim. reduction because padding = 0

        self.conv5_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv5_2 = nn.Conv2d(128, 256, kernel_size=3, padding=0)  # dim. reduction because padding = 0

        # Initialize convolutions' parameters
        self.init_conv2d()

    def init_conv2d(self):
        """
        Initialize convolution parameters.
        """
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, conv7_feats):
        """
        Forward propagation.

        :param conv7_feats: lower-level conv7 feature map, a tensor of dimensions (N, 1024, 19, 19)
        :return: higher-level feature maps conv8_2, conv9_2, conv10_2, and conv11_2
        """
        out = F.relu(self.start_1(conv7_feats))
        out = F.relu(self.start_2(out))
        out = F.relu(self.start_3(out))
        out = F.relu(self.start_4(out))
        conv0_4_feats = out # (N, 1024, 25, 25)

        out = F.relu(self.conv1_1(out))  # (N, 256, 25, 25)
        out = F.relu(self.conv1_2(out))  # (N, 512, 13, 13)
        conv1_2_feats = out  # (N, 512, 13, 13)

        out = F.relu(self.conv2_1(out))  # (N, 128, 13, 13)
        out = F.relu(self.conv2_2(out))  # (N, 256, 7, 7)
        conv2_2_feats = out  # (N, 256, 7, 7)

        out = F.relu(self.conv3_1(out))  # (N, 128, 7, 7)
        out = F.relu(self.conv3_2(out))  # (N, 256, 5, 5)
        conv3_2_feats = out  # (N, 256, 5, 5)

        out = F.relu(self.conv4_1(out))  # (N, 128, 5, 5)
        out = F.relu(self.conv4_2(out))  # (N, 256, 3, 3)
        conv4_2_feats = out   # (N, 256, 3, 3)

        out = F.relu(self.conv5_1(out))  # (N, 128, 3, 3)
        conv5_2_feats = F.relu(self.conv5_2(out))  # (N, 256, 1, 1)

        # Higher-level feature maps
        #print(conv0_4_feats.shape, conv1_2_feats.shape, conv2_2_feats.shape, conv3_2_feats.shape, conv4_2_feats.shape, conv5_2_feats.shape)
        return conv0_4_feats, conv1_2_feats, conv2_2_feats, conv3_2_feats, conv4_2_feats, conv5_2_feats
    
class PredictionConvolutions(torch.nn.Module):
    """
    Convolutions to predict class scores and bounding boxes using lower and higher-level feature maps.

    The bounding boxes (locations) are predicted as encoded offsets w.r.t each of the 8732 prior (default) boxes.
    See 'cxcy_to_gcxgcy' in utils.py for the encoding definition.

    The class scores represent the scores of each object class in each of the 8732 bounding boxes located.
    A high score for 'background' = no object.
    """

    def __init__(self, n_classes):
        """
        :param n_classes: number of different types of objects
        """
        super(PredictionConvolutions, self).__init__()

        self.n_classes = n_classes

        # Number of prior-boxes we are considering per position in each feature map
        n_boxes = {'conv0': 6,
                   'conv1': 6,
                   'conv2': 6,
                   'conv3': 6,
                   'conv4': 4,
                   'conv5': 4}
        # 4 prior-boxes implies we use 4 different aspect ratios, etc.

        # Localization prediction convolutions (predict offsets w.r.t prior-boxes)
        self.loc_conv0 = torch.nn.Conv2d(1024, n_boxes['conv0'] * 4, kernel_size=3, padding=1)
        self.loc_conv1 = torch.nn.Conv2d(512, n_boxes['conv1'] * 4, kernel_size=3, padding=1)
        self.loc_conv2 = torch.nn.Conv2d(256, n_boxes['conv2'] * 4, kernel_size=3, padding=1)
        self.loc_conv3 = torch.nn.Conv2d(256, n_boxes['conv3'] * 4, kernel_size=3, padding=1)
        self.loc_conv4 = torch.nn.Conv2d(256, n_boxes['conv4'] * 4, kernel_size=3, padding=1)
        self.loc_conv5 = torch.nn.Conv2d(256, n_boxes['conv5'] * 4, kernel_size=3, padding=1)

        # Class prediction convolutions (predict classes in localization boxes)
        self.cl_conv0 = torch.nn.Conv2d(1024, n_boxes['conv0'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv1 = torch.nn.Conv2d(512, n_boxes['conv1'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv2 = torch.nn.Conv2d(256, n_boxes['conv2'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv3 = torch.nn.Conv2d(256, n_boxes['conv3'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv4 = torch.nn.Conv2d(256, n_boxes['conv4'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv5 = torch.nn.Conv2d(256, n_boxes['conv5'] * n_classes, kernel_size=3, padding=1)

        # Initialize convolutions' parameters
        self.init_conv2d()

    def init_conv2d(self):
        """
        Initialize convolution parameters.
        """
        for c in self.children():
            if isinstance(c, torch.nn.Conv2d):
                torch.nn.init.xavier_uniform_(c.weight)
                torch.nn.init.constant_(c.bias, 0.)

    def forward(self, conv0_feats, conv1_feats, conv2_feats, conv3_feats, conv4_feats, conv5_feats):
        """
        Forward propagation.

        :param conv4_3_feats: conv4_3 feature map, a tensor of dimensions (N, 512, 38, 38)
        :param conv7_feats: conv7 feature map, a tensor of dimensions (N, 1024, 19, 19)
        :param conv8_2_feats: conv8_2 feature map, a tensor of dimensions (N, 512, 10, 10)
        :param conv9_2_feats: conv9_2 feature map, a tensor of dimensions (N, 256, 5, 5)
        :param conv10_2_feats: conv10_2 feature map, a tensor of dimensions (N, 256, 3, 3)
        :param conv11_2_feats: conv11_2 feature map, a tensor of dimensions (N, 256, 1, 1)
        :return: 8732 locations and class scores (i.e. w.r.t each prior box) for each image
        """
        batch_size = conv0_feats.size(0)

        # Predict localization boxes' bounds (as offsets w.r.t prior-boxes)
        l_conv0 = self.loc_conv0(conv0_feats)  # (N, 16, 38, 38)
        l_conv0 = l_conv0.permute(0, 2, 3,
                                      1).contiguous()  # (N, 38, 38, 16), to match prior-box order (after .view())
        # (.contiguous() ensures it is stored in a contiguous chunk of memory, needed for .view() below)
        l_conv0 = l_conv0.view(batch_size, -1, 4)  # (N, 5776, 4), there are a total 5776 boxes on this feature map

        l_conv1 = self.loc_conv1(conv1_feats)  # (N, 24, 19, 19)
        l_conv1 = l_conv1.permute(0, 2, 3, 1).contiguous()  # (N, 19, 19, 24)
        l_conv1 = l_conv1.view(batch_size, -1, 4)  # (N, 2166, 4), there are a total 2116 boxes on this feature map

        l_conv2 = self.loc_conv2(conv2_feats)  # (N, 24, 10, 10)
        l_conv2 = l_conv2.permute(0, 2, 3, 1).contiguous()  # (N, 10, 10, 24)
        l_conv2 = l_conv2.view(batch_size, -1, 4)  # (N, 600, 4)

        l_conv3 = self.loc_conv3(conv3_feats)  # (N, 24, 5, 5)
        l_conv3 = l_conv3.permute(0, 2, 3, 1).contiguous()  # (N, 5, 5, 24)
        l_conv3 = l_conv3.view(batch_size, -1, 4)  # (N, 150, 4)

        l_conv4 = self.loc_conv4(conv4_feats)  # (N, 16, 3, 3)
        l_conv4 = l_conv4.permute(0, 2, 3, 1).contiguous()  # (N, 3, 3, 16)
        l_conv4 = l_conv4.view(batch_size, -1, 4)  # (N, 36, 4)

        l_conv5 = self.loc_conv5(conv5_feats)  # (N, 16, 1, 1)
        l_conv5 = l_conv5.permute(0, 2, 3, 1).contiguous()  # (N, 1, 1, 16)
        l_conv5 = l_conv5.view(batch_size, -1, 4)  # (N, 4, 4)

        # Predict classes in localization boxes
        c_conv0 = self.cl_conv0(conv0_feats)  # (N, 4 * n_classes, 38, 38)
        c_conv0 = c_conv0.permute(0, 2, 3, 1).contiguous()  # (N, 38, 38, 4 * n_classes), to match prior-box order (after .view())
        c_conv0 = c_conv0.view(batch_size, -1,
                                   self.n_classes)  # (N, 5776, n_classes), there are a total 5776 boxes on this feature map

        c_conv1 = self.cl_conv1(conv1_feats)  # (N, 6 * n_classes, 19, 19)
        c_conv1 = c_conv1.permute(0, 2, 3, 1).contiguous()  # (N, 19, 19, 6 * n_classes)
        c_conv1 = c_conv1.view(batch_size, -1,
                               self.n_classes)  # (N, 2166, n_classes), there are a total 2116 boxes on this feature map

        c_conv2 = self.cl_conv2(conv2_feats)  # (N, 6 * n_classes, 10, 10)
        c_conv2 = c_conv2.permute(0, 2, 3, 1).contiguous()  # (N, 10, 10, 6 * n_classes)
        c_conv2 = c_conv2.view(batch_size, -1, self.n_classes)  # (N, 600, n_classes)

        c_conv3 = self.cl_conv3(conv3_feats)  # (N, 6 * n_classes, 5, 5)
        c_conv3 = c_conv3.permute(0, 2, 3, 1).contiguous()  # (N, 5, 5, 6 * n_classes)
        c_conv3 = c_conv3.view(batch_size, -1, self.n_classes)  # (N, 150, n_classes)

        c_conv4 = self.cl_conv4(conv4_feats)  # (N, 4 * n_classes, 3, 3)
        c_conv4 = c_conv4.permute(0, 2, 3, 1).contiguous()  # (N, 3, 3, 4 * n_classes)
        c_conv4 = c_conv4.view(batch_size, -1, self.n_classes)  # (N, 36, n_classes)

        c_conv5 = self.cl_conv5(conv5_feats)  # (N, 4 * n_classes, 1, 1)
        c_conv5 = c_conv5.permute(0, 2, 3, 1).contiguous()  # (N, 1, 1, 4 * n_classes)
        c_conv5 = c_conv5.view(batch_size, -1, self.n_classes)  # (N, 4, n_classes)

        # A total of 8732 boxes
        # Concatenate in this specific order (i.e. must match the order of the prior-boxes)
        locs = torch.cat([l_conv0, l_conv1, l_conv2, l_conv3, l_conv4, l_conv5], dim=1)  # (N, 8732, 4)
        classes_scores = torch.cat([c_conv0, c_conv1, c_conv2, c_conv3, c_conv4, c_conv5],
                                   dim=1)  # (N, 8732, n_classes)

        return locs, classes_scores