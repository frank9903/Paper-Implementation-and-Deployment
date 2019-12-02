import os
import sys
import numpy as np
import tensorflow as tf
import keras
from keras import backend, layers, engine, models

from mrcnn.config import Config
from mrcnn import utils, data_utils as data, model_utils

# This is very similar to training model and I highlight all the differences.
def build_inference_model(config):
    input_image = layers.Input(shape=[None, None, config.IMAGE_SHAPE[2]], name="input_image")
    input_image_meta = layers.Input(shape=[config.IMAGE_META_SIZE], name="input_image_meta")
    # Diff1: we don't need any GT as input
    # Anchors in normalized coordinates
    input_anchors = layers.Input(shape=[None, 4], name="input_anchors")

    # FPN Layer
    _, C2, C3, C4, C5 = model_utils.resnet_graph(input_image, config.BACKBONE, stage5=True, train_bn=config.TRAIN_BN)

    P5 = layers.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c5p5')(C5)
    P4 = layers.Add(name="fpn_p4add")([
        layers.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
        layers.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c4p4')(C4)
    ])
    P3 = layers.Add(name="fpn_p3add")([
        layers.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
        layers.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c3p3')(C3)
    ])
    P2 = layers.Add(name="fpn_p2add")([
        layers.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
        layers.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c2p2')(C2)
    ])

    P2 = layers.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p2")(P2)
    P3 = layers.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p3")(P3)
    P4 = layers.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p4")(P4)
    P5 = layers.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p5")(P5)
    P6 = layers.MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")(P5)

    rpn_feature_maps = [P2, P3, P4, P5, P6]
    mrcnn_feature_maps = [P2, P3, P4, P5]

    # Diff2: we will accept all anchors
    anchors = input_anchors

    feature_map = layers.Input(shape=[None, None, config.TOP_DOWN_PYRAMID_SIZE], name="input_rpn_feature_map")
    anchors_per_location = len(config.RPN_ANCHOR_RATIOS)

    # Builds the computation graph of Region Proposal Network.
    # Shared convolutional base of the RPN
    shared = layers.Conv2D(512, (3, 3), padding='same', activation='relu',
                        strides=config.RPN_ANCHOR_STRIDE,
                        name='rpn_conv_shared')(feature_map)

    # Anchor Score. [batch, height, width, anchors per location * 2].
    x = layers.Conv2D(2 * anchors_per_location, (1, 1), padding='valid', activation='linear', name='rpn_class_raw')(shared)

    # Reshape to [batch, anchors, 2]
    rpn_class_logits = layers.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 2]))(x)

    # Softmax on last dimension of BG/FG.
    rpn_probs = layers.Activation("softmax", name="rpn_class_xxx")(rpn_class_logits)

    # Bounding box refinement. [batch, H, W, anchors per location * depth]
    # where depth is [x, y, log(w), log(h)]
    x = layers.Conv2D(anchors_per_location * 4, (1, 1), padding="valid",
                    activation='linear', name='rpn_bbox_pred')(shared)

    # Reshape to [batch, anchors, 4]
    rpn_bbox = layers.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 4]))(x)

    rpn = models.Model([feature_map], [rpn_class_logits, rpn_probs, rpn_bbox], name="rpn_model")

    layer_outputs = []
    for p in rpn_feature_maps:
        layer_outputs.append(rpn([p]))

    output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
    outputs = list(zip(*layer_outputs))
    outputs = [layers.Concatenate(axis=1, name=n)(list(o)) for o, n in zip(outputs, output_names)]

    rpn_class_logits, rpn_class, rpn_bbox = outputs

    # Generate proposals
    proposal_count = config.POST_NMS_ROIS_TRAINING
    rpn_rois = model_utils.ProposalLayer(
        proposal_count=proposal_count,
        nms_threshold=config.RPN_NMS_THRESHOLD,
        name="ROI",
        config=config)([rpn_class, rpn_bbox, anchors])
    
    # Network Heads
    # Proposal classifier and BBox regressor heads
    mrcnn_class_logits, mrcnn_class, mrcnn_bbox =\
        model_utils.fpn_classifier_graph(rpn_rois, mrcnn_feature_maps, input_image_meta,
                                        config.POOL_SIZE, config.NUM_CLASSES,
                                        train_bn=config.TRAIN_BN,
                                        fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE)

    # Detections
    # output is [batch, num_detections, (y1, x1, y2, x2, class_id, score)] in normalized coordinates
    detections = model_utils.DetectionLayer(config, name="mrcnn_detection")(
        [rpn_rois, mrcnn_class, mrcnn_bbox, input_image_meta])

    # Create masks for detections
    detection_boxes = layers.Lambda(lambda x: x[..., :4])(detections)
    mrcnn_mask = model_utils.build_fpn_mask_graph(detection_boxes, mrcnn_feature_maps,
                                                input_image_meta,
                                                config.MASK_POOL_SIZE,
                                                config.NUM_CLASSES,
                                                train_bn=config.TRAIN_BN)

    inference_model = models.Model([input_image, input_image_meta, input_anchors],
                                [detections, mrcnn_class, mrcnn_bbox, mrcnn_mask, rpn_rois, rpn_class, rpn_bbox],
                                name='mask_rcnn')
    
    return inference_model