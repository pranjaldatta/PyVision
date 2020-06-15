import torch
import torch.nn as nn
import torch.nn.functional as F  

from .utils.misc import NestedTensor, nested_tensor_from_tensor_list
from .utils.box_utils import box_wh_to_xy

class MLP(nn.Module):
    """
    A very simple multi layer perceptron also known as FFN 
    """
    def __init__(self, in_dims, hidden_dims, out_dims, num_layers):
        
        super().__init__()

        self.num_layers = num_layers
        h = [hidden_dims] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([in_dims]+h, h+[out_dims]))

    def forward(self, x):
        
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class DETR_model(nn.Module):
    """
    The main detr module that performs the forward pass
    """
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False):
        """The module that builds the detr model

        Parameters
        ----------
        backbone : [nn.Module]
            the backbone to be used by the detr model. defined in backbone.py  
        transformer : [nn.Module]
            the transformer to be used by the detr model. define din transformers.py
        num_classes : [int]  
            number of object classses   
        num_queries : [int]
            number of object queries i.e. detection slot i.e. the maximum number 
            of objects that can be detected in a single image. For COCO, its 100
        aux_loss : bool, optional
            if auxiliary decoding losses are to be used, by default False
        """
        
        super().__init__()

        self.backbone = backbone
        self.transformer = transformer
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.aux_loss = aux_loss
        
        hidden_dim = self.transformer.d_model 
        self.class_embed = nn.Linear(hidden_dim, num_classes+1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, 1)

    def forward(self, samples: NestedTensor):
        """
        The forward method defines a single forward pass for the model.
        It expects a TensorList obect which consists of :
            - samples.tensor: batched images of shape [B, 3, H, W] 
            - samples.mask: a binary mask of shape [B, H, W] containing 1 padded pixels
        
        It returns the following elements:
            -  pred_logits = classification logits for all queries.
                             Shape = [B, num_queries, (num_classes + 1)]
            - pred_boxes = normalized box coordinates for all object queries represented as
                           (center_x, center_y, height, width). These values are normalized
                           between [0, 1] relative to size of each input image. utils/postprocess
                           retrieves unnormalized bounding boxes
            - aux_outputs = Optional           
                             
        """       
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        
        # we run it through the backbone 
        features, pos = self.backbone(samples)

        # now we get the tensors and masks for each image and make the transformer pass
        src, mask = features[-1].decompose()
        assert mask is not None
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]

        output_class = self.class_embed(hs)
        output_coord = self.bbox_embed(hs).sigmoid()
        out = {
            "pred_logits": output_class[-1],
            "pred_boxes": output_coord[-1]
        }
        if self.aux_loss:
            raise NotImplementedError("aux_loss not implemented yet")
        
        return out

class DETR_postprocess(nn.Module):
    """
    This module converts DETR output into a simple usable format"""
    def __init__(self, conf=0.7):
        super(DETR_postprocess, self).__init__()
        self.conf = conf

    @torch.no_grad()
    def forward(self, outputs, target_size):
        """
        Converts raw DETR outputs into a usable format i.e. it takes the raw 
        normalized (wrt to [0, 1]) bounding boxes predictions, unnormalizes it,
        scales it to original image size and returns a list of dictionaries of
        format {score, class_label, box_coords} for all the detections in a given image
        """
        raw_logits, raw_boxes = outputs['pred_logits'], outputs["pred_boxes"]

        assert len(raw_logits) == len(target_size), "raw_logits and target size len mismatch"
        assert target_size.shape[1] == 2, "target_size shape dim 1 not equal to 2"

        probs = F.softmax(raw_logits, -1)[0,:,:-1]
        keep = probs.max(-1).values > self.conf
        probs = probs[keep]
        probs, labels = probs[...,:-1].max(-1)

        # converting boxes to [x1, y1, x2, y2] format
        raw_boxes = raw_boxes[:,keep,:]
        boxes = box_wh_to_xy(raw_boxes)

        if boxes.device is not "cpu":
            boxes = boxes.cpu()
        
        # convert coords relative to [0, 1] to absolute [H, W] coords
        img_height, img_width = target_size.unbind(1)
        scale_factors = torch.stack([img_width, img_height, img_width, img_height], dim=1)
        boxes = boxes  * scale_factors[:, :] # remove none

        results = [{"scores": s.item(), "labels": l.item(), "coords": c.tolist()} for s, l, c in zip(probs, labels, boxes[0])]

        return results




    