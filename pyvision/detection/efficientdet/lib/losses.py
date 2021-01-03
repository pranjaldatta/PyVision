import torch
import torch.nn as nn 


def iou(a, b):

    area_a = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1)
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], dim=1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], dim=1), b[:, 1])
    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    inter_area = iw * ih
    union_area = area_a + area_b - inter_area

    union_area = torch.clamp(union_area, min=1e-8)
    iou_score = inter_area / union_area

    return iou_score

"""
class FocalLoss(nn.Module):

    def __init__(self, alpha=0.25, gamma=2, device="cuda"):

        super(FocalLoss, self).__init__()

        self.alpha = alpha 
        self.gamma = gamma 
        self.device = device
    
    def forward(self, classifications, regressions, anchors, annotations):

        batch_size = classifications.shape[0]
        
        classification_loss = []
        regression_loss = []

        anchor = anchors[0, :, :]

        anchor_widths = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_x = anchor[:, 0] + 0.5 * anchor_widths
        anchor_y = anchor[:, 1] + 0.5 * anchor_heights

        for i in range(batch_size):

            classification = classifications[i, :, :]
            regression = regressions[i, :, :]

            box_annotation = annotations[i, :, :]
            box_annotation = box_annotation[box_annotation[:, 4] != -1]

            if box_annotation.shape[0] == 0:
                if self.device == "cuda" and torch.cuda.is_available():
                    regression_loss.append(torch.tensor(0).float().cuda())
                    classification_loss.append(torch.tensor(0).float().cuda())
                else:
                    regression_loss.append(torch.tensor(0).float().cuda())
                    classification_loss.append(torch.tensor(0).float().cuda())
                
                # no loss or no det. Move on to the next item
                continue
                
            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            iou_score = iou(anchors[0, :, :], box_annotation[:, :4])
            iou_max, iou_argmax = torch.max(iou_score, dim=1)

            targets = torch.ones(classification.shape) * -1
            if self.device == "cuda" and torch.cuda.is_available():
                targets = targets.cuda() 
            
            # zeroing out the indices with IOU less than 0.4
            targets[torch.lt(iou_max, 0.4), :] = 0

            # getting the indices with IoU score > 0.5
            positive_idx = torch.ge(iou_max, 0.5)
            num_positive_idx = positive_idx.sum()

            assigned_annots = box_annotation[iou_argmax, :]

            targets[positive_idx, :] = 0
            targets[positive_idx, assigned_annots[positive_idx, 4].long()] = 1

            alpha_factor = torch.ones(targets.shape) * self.alpha
            if self.device == "cuda" and torch.cuda.is_available():
                alpha_factor = alpha_factor.cuda()
            
            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1.0 - alpha_factor)
            focal_weight =  torch.where(torch.eq(targets, 1.), 1.0 - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, self.gamma)

            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))
            cls_loss = alpha_factor * bce 

            zeros = torch.zeros(cls_loss.shape)
            if self.device == "cuda" and torch.cuda.is_available():
                zeros = zeros.cuda()
            cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, zeros)

            classification_loss.append(
                cls_loss.sum() / torch.clamp(num_positive_idx.float(), min=1.0)
            )   

            # implement regression loss      
            if num_positive_idx > 0:

                assigned_annots = assigned_annots[positive_idx, :]

                anchor_widths_i = anchor_widths[positive_idx]
                anchor_heights_i = anchor_heights[positive_idx]
                anchor_xi = anchor_x[positive_idx]
                anchor_yi = anchor_y[positive_idx]

                true_widths = assigned_annots[:, 2] - assigned_annots[:, 0]
                true_heights = assigned_annots[:, 3] - assigned_annots[:, 1]
                true_x = assigned_annots[:, 0] + 0.5 * true_widths
                true_y = assigned_annots[:, 1] + 0.5 * true_heights

                true_heights = torch.clamp(true_heights, min=1)
                true_widths = torch.clamp(true_widths, min=1)

                targets_dx = (true_x - anchor_xi) / anchor_widths_i
                targets_dy = (true_y - anchor_yi) / anchor_heights_i
                targets_dw = torch.log(true_widths / anchor_widths_i)
                targets_dh = torch.log(true_heights / anchor_heights_i)
                
                targets = torch.stack((
                    targets_dx, targets_dy, targets_dw, targets_dh
                ))
                targets = targets.t()

                norm = torch.Tensor([0.1, 0.1, 0.2, 0.2])
                if self.device == "cuda" and torch.cuda.is_available():
                    norm = norm.cuda() 
                targets = targets / norm  

                regression_diff = torch.abs(targets - regression[positive_idx, :])
                regression_loss_i = torch.where(
                    torch.le(regression_diff, 1.0/9.0), 
                    0.5 * 9.0 * torch.pow(regression_diff, 2), 
                    regression_diff - 0.5 / 9.0
                )
                regression_loss.append(regression_loss_i.mean())
            
            else:

                if self.device == "cuda" and torch.cuda.is_available():
                    regression_loss.append(torch.tensor(0).float().cuda())
                else: 
                    regression_loss.append(torch.tensor(0).float())

        
        return_cls_loss = torch.stack(classification_loss).mean(dim=0, keepdim=True)
        return_reg_loss = torch.stack(regression_loss).mean(dim=0, keepdim=True)

        return return_cls_loss, return_reg_loss

"""

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, device="cuda"):
        
        super(FocalLoss, self).__init__()
        
        self.alpha = alpha
        self.gamma = gamma 
        self.device = device

    def forward(self, classifications, regressions, anchors, annotations):

        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        anchor = anchors[0, :, :]

        anchor_widths = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y = anchor[:, 1] + 0.5 * anchor_heights

        for j in range(batch_size):

            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            bbox_annotation = annotations[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

            if bbox_annotation.shape[0] == 0:
                if self.device == "cuda" and torch.cuda.is_available():
                    regression_losses.append(torch.tensor(0).float().cuda())
                    classification_losses.append(torch.tensor(0).float().cuda())
                else:
                    regression_losses.append(torch.tensor(0).float())
                    classification_losses.append(torch.tensor(0).float())

                continue

            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, :4])

            IoU_max, IoU_argmax = torch.max(IoU, dim=1)

            # compute the loss for classification
            targets = torch.ones(classification.shape) * -1
            if self.device == "cuda" and torch.cuda.is_available():
                targets = targets.cuda()

            targets[torch.lt(IoU_max, 0.4), :] = 0

            positive_indices = torch.ge(IoU_max, 0.5)

            num_positive_anchors = positive_indices.sum()

            assigned_annotations = bbox_annotation[IoU_argmax, :]

            targets[positive_indices, :] = 0
            targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1

            alpha_factor = torch.ones(targets.shape) * self.alpha
            if self.device == "cuda" and torch.cuda.is_available():
                alpha_factor = alpha_factor.cuda()

            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, self.gamma)

            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))

            cls_loss = focal_weight * bce

            zeros = torch.zeros(cls_loss.shape)
            if self.device == "cuda" and torch.cuda.is_available():
                zeros = zeros.cuda()
            cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, zeros)

            classification_losses.append(cls_loss.sum() / torch.clamp(num_positive_anchors.float(), min=1.0))


            if positive_indices.sum() > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]

                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                gt_widths = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y = assigned_annotations[:, 1] + 0.5 * gt_heights

                gt_widths = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)

                targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
                targets = targets.t()

                norm = torch.Tensor([[0.1, 0.1, 0.2, 0.2]])
                if self.device == "cuda" and torch.cuda.is_available():
                    norm = norm.cuda()
                targets = targets / norm

                regression_diff = torch.abs(targets - regression[positive_indices, :])

                regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )
                regression_losses.append(regression_loss.mean())
            else:
                if self.device == "cuda" and torch.cuda.is_available():
                    regression_losses.append(torch.tensor(0).float().cuda())
                else:
                    regression_losses.append(torch.tensor(0).float())

        return torch.stack(classification_losses).mean(dim=0, keepdim=True), torch.stack(regression_losses).mean(dim=0,
                                                                                                                 keepdim=True)


