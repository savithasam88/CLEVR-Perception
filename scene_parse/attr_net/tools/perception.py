import sys
import json
from pathlib import Path
from collections import Counter
file_path = Path(__file__).resolve()
path_current = file_path.parents[0]
path_root = file_path.parents[1]
sys.path.append(".")
sys.path.append(str(path_root))
sys.path.append(str(path_current))

import random
from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Union, Tuple
from PIL import Image
import cv2
import torch
import requests
import numpy as np
from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline
from utils import load_clevr_scenes, get_feat_vec_clevr
#from run_test import attribute_detection
from scipy.spatial.distance import cdist
import pickle

import os

from options import get_options
from datasets import get_dataloader
from model import get_model
import utils

REGIONS = {
    "0": {"x": [0, 240], "y": [0, 160]}, 
    "1": {"x": [240, 480], "y": [0, 160]},
    "2": {"x": [0, 240], "y": [160, 320]},
    "3": {"x": [240, 480], "y": [160, 320]}
    }

@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]

@dataclass
class DetectionResult:
    score: float
    label: str
    box: BoundingBox
    mask: Optional[np.array] = None

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> 'DetectionResult':
        return cls(score=detection_dict['score'],
                   label=detection_dict['label'],
                   box=BoundingBox(xmin=detection_dict['box']['xmin'],
                                   ymin=detection_dict['box']['ymin'],
                                   xmax=detection_dict['box']['xmax'],
                                   ymax=detection_dict['box']['ymax']))


    

def mask_to_polygon(mask: np.ndarray) -> List[List[int]]:
    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)

    # Extract the vertices of the contour
    polygon = largest_contour.reshape(-1, 2).tolist()

    return polygon

def polygon_to_mask(polygon: List[Tuple[int, int]], image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert a polygon to a segmentation mask.

    Args:
    - polygon (list): List of (x, y) coordinates representing the vertices of the polygon.
    - image_shape (tuple): Shape of the image (height, width) for the mask.

    Returns:
    - np.ndarray: Segmentation mask with the polygon filled.
    """
    # Create an empty mask
    mask = np.zeros(image_shape, dtype=np.uint8)

    # Convert polygon to an array of points
    pts = np.array(polygon, dtype=np.int32)

    # Fill the polygon with white color (255)
    cv2.fillPoly(mask, [pts], color=(255,))

    return mask

def load_image(image_str: str) -> Image.Image:
    if image_str.startswith("http"):
        image = Image.open(requests.get(image_str, stream=True).raw).convert("RGB")
    else:
        image = Image.open(image_str).convert("RGB")
    print('Image size:', image.size)
    size=(480, 320)
    # Resize the image to the desired size (320x240)
    image_resized = image.resize(size)

    return image

def get_boxes(results: DetectionResult) -> List[List[List[float]]]:
    boxes = []
    for result in results:
        xyxy = result.box.xyxy
        boxes.append(xyxy)

    return [boxes]

def refine_masks(masks: torch.BoolTensor, polygon_refinement: bool = False) -> List[np.ndarray]:
    masks = masks.cpu().float()
    masks = masks.permute(0, 2, 3, 1)
    masks = masks.mean(axis=-1)
    masks = (masks > 0).int()
    masks = masks.numpy().astype(np.uint8)
    masks = list(masks)

    if polygon_refinement:
        for idx, mask in enumerate(masks):
            shape = mask.shape
            polygon = mask_to_polygon(mask)
            mask = polygon_to_mask(polygon, shape)
            masks[idx] = mask

    return masks

def create_pipeline_and_processor(detector_id: Optional[str] = None, segmenter_id: Optional[str] = None):
    """
    Function to initialize the pipeline and processor once.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector_id = detector_id if detector_id is not None else "IDEA-Research/grounding-dino-tiny"
    segmenter_id = segmenter_id if segmenter_id is not None else "facebook/sam-vit-base"
    
    # Grounding DINO pipeline
    object_detector = pipeline(model=detector_id, task="zero-shot-object-detection", device=device)

    # Segment Anything model and processor
    segmentator = AutoModelForMaskGeneration.from_pretrained(segmenter_id).to(device)
    processor = AutoProcessor.from_pretrained(segmenter_id)

    return object_detector, segmentator, processor

def detect(
    image: Image.Image,
    labels: List[str],
    threshold: float = 0.3,
    object_detector: Optional = None #detector_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Use Grounding DINO to detect a set of labels in an image in a zero-shot fashion.
    """
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    #detector_id = detector_id if detector_id is not None else "IDEA-Research/grounding-dino-tiny"
    #object_detector = pipeline(model=detector_id, task="zero-shot-object-detection", device=device)

    labels = [label if label.endswith(".") else label+"." for label in labels]

    results = object_detector(image,  candidate_labels=labels, threshold=threshold)
    results = [DetectionResult.from_dict(result) for result in results]

    return results

def segment(
    image: Image.Image,
    detection_results: List[Dict[str, Any]],
    polygon_refinement: bool = False,
    segmentator: Optional = None,
    processor: Optional = None,
    
    #segmenter_id: Optional[str] = None
) -> List[DetectionResult]:
    """
    Use Segment Anything (SAM) to generate masks given an image + a set of bounding boxes.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #segmenter_id = segmenter_id if segmenter_id is not None else "facebook/sam-vit-base"

    #segmentator = AutoModelForMaskGeneration.from_pretrained(segmenter_id).to(device)
    #processor = AutoProcessor.from_pretrained(segmenter_id)

    boxes = get_boxes(detection_results)
    inputs = processor(images=image, input_boxes=boxes, return_tensors="pt").to(device)

    outputs = segmentator(**inputs)
    masks = processor.post_process_masks(
        masks=outputs.pred_masks,
        original_sizes=inputs.original_sizes,
        reshaped_input_sizes=inputs.reshaped_input_sizes
    )[0]

    masks = refine_masks(masks, polygon_refinement)

    for detection_result, mask in zip(detection_results, masks):
        detection_result.mask = mask

    return detection_results

def grounded_segmentation(
    image: Union[Image.Image, str],
    labels: List[str],
    threshold: float = 0.7,
    polygon_refinement: bool = False,
    detector: Optional = None,
    segmentator: Optional = None,
    processor: Optional = None
    #detector_id: Optional[str] = None,
    #segmenter_id: Optional[str] = None
) -> Tuple[np.ndarray, List[DetectionResult]]:
    if isinstance(image, str):
        image = load_image(image)

    detections = detect(image, labels, threshold, detector) #detector_id)
    detections = segment(image, detections, polygon_refinement, segmentator, processor)#segmenter_id)
    return np.array(image), detections


# Function to classify the region based on bounding box center (optional)
def classify_region(x, y):
    image_width = 480
    image_height = 320
    
    for region_id, region in REGIONS.items():
        if region["x"][0] <= x < region["x"][1] and region["y"][0] <= y < region["y"][1]:
            return region_id


def bbox_iou(bbox1, bbox2):
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.
    
    Args:
    bbox1 (tuple): Bounding box 1 as (xmin, ymin, xmax, ymax)
    bbox2 (tuple): Bounding box 2 as (xmin, ymin, xmax, ymax)
    
    Returns:
    float: The IoU of the two bounding boxes
    """
    # Coordinates of the intersection area
    xmin_intersection = max(bbox1.xmin, bbox2.xmin)
    ymin_intersection = max(bbox1.ymin, bbox2.ymin)
    xmax_intersection = min(bbox1.xmax, bbox2.xmax)
    ymax_intersection = min(bbox1.ymax, bbox2.ymax)
    
    # If the boxes don't overlap, IoU is 0
    if xmin_intersection >= xmax_intersection or ymin_intersection >= ymax_intersection:
        return 0.0
    
    # Area of the intersection
    intersection_area = (xmax_intersection - xmin_intersection) * (ymax_intersection - ymin_intersection)
    
    # Area of both bounding boxes
    area_bbox1 = (bbox1.xmax - bbox1.xmin) * (bbox1.ymax - bbox1.ymin)
    area_bbox2 = (bbox2.xmax - bbox2.xmin) * (bbox2.ymax - bbox2.ymin)
    
    # Area of the union
    union_area = area_bbox1 + area_bbox2 - intersection_area
    
    # IoU: Intersection area / Union area
    iou = intersection_area / union_area
    
    return iou

def find_match(bbdet, GT_Obj_feat, Object_pixels):
    bbdet_array = np.array(bbdet)
    pixel_coords = []
    for p in Object_pixels:
        pixel_coord = []
        pixel_coord.append(Object_pixels[p][0])
        pixel_coord.append(Object_pixels[p][1])
        pixel_coords.append(pixel_coord)
    # convert to numpy array for convenience
    pixel_coords_array = np.array(pixel_coords)
    # compute Euclidean distances
    distances = np.linalg.norm(pixel_coords_array - bbdet, axis=1)
    # find index of closest pixel
    closest_idx = np.argmin(distances)
    # get the closest pixel
    closest_pixel = pixel_coords[closest_idx]
    return GT_Obj_feat[closest_idx]


# batching might work with DINO but may not work for SAM as number of bboxes or objects per image vary across images.
'''
def create_object_proposals():
    batch_size = 8
    detector_id = "IDEA-Research/grounding-dino-tiny"
    segmenter_id = "facebook/sam-vit-base"
    detector, segmentator, processor = create_pipeline_and_processor(detector_id, segmenter_id)
    shapes = ['sphere', 'cube', 'cylinder']
    labels = ['sphere .', 'cube .', 'cylinder .']
    threshold = 0.3
         
    data_loc = "/users/sbsh670/data/clevr/CLEVR_v1.0/image_generative_model_training/images/train/" #"/users/sbsh670/data/clevr/CLEVR_v1.0/images/val/"
    
    for start in range(0, min(1000, len(scenes)), batch_size):
        batch_scenes = scenes[start:start+batch_size]
        image_urls = [data_loc + s['filename']
        for s in batch_scenes
    ]
    batch_image_array, batch_detections = grounded_segmentation(
        image=image_urls,          # list instead of single image
        labels=labels,
        threshold=threshold,
        polygon_refinement=True,
        detector=detector,
        segmentator=segmentator,
        processor=processor
    )

    for s, image_url, detections in zip(
            batch_scenes,
            image_urls,
            batch_detections):
'''
#return object proposals for clevr scenes from original CLEVR dataset
def create_object_proposals():

    scenes = load_clevr_scenes("/users/sbsh670/data/clevr/CLEVR_v1.0/scenes/CLEVR_train_scenes.json")
    detector_id = "IDEA-Research/grounding-dino-tiny"
    segmenter_id = "facebook/sam-vit-base"
    detector, segmentator, processor = create_pipeline_and_processor(detector_id, segmenter_id)
    shapes = ['sphere', 'cube', 'cylinder']
    labels = ['sphere .', 'cube .', 'cylinder .']
    threshold = 0.3
         
    data_loc = "/users/sbsh670/data/clevr/CLEVR_v1.0/image_generative_model_training/images/train/" #"/users/sbsh670/data/clevr/CLEVR_v1.0/images/val/"
    img_anns = []
    count = 0
    
        
    for i,s in enumerate(scenes):
       
        if count>=100:
            break
        count = count+1
        
        image_url = data_loc+ s['filename'] #"CLEVR_val_000000.png"
        image_idx = s['image_idx']
        
       
        
        GT_Obj_feat = {}
        Obj_pixel = {}
        o_id = 0
        for o in scenes[i]['objects']:
            GT_Obj_feat [o_id] = get_feat_vec_clevr(o)
            Obj_pixel[o_id] = o['pixel_coords']
            o_id = o_id + 1

        
        image_array, detections = grounded_segmentation(
        image=image_url,
        labels=labels,
        threshold=threshold,
        polygon_refinement=True,
        detector=detector,           
        segmentator=segmentator,     
        processor=processor)
        
        bboxes = []
    
        for idx, detection in enumerate(detections):
            present = False
            p_s = 0
            p_b = None
            box = detection.box
            score = detection.score
            for bb_score in bboxes:
                bb = bb_score[0]
                if bbox_iou(bb, box)> 0.2:
                    p_s = bb_score[1]
                    p_b = bb
                    idx_rem =  bb_score[2]
                    present = True
                    break
        
            if present:
                if p_s>=score:
                    continue 
                else:
                    bboxes.remove((p_b, p_s, idx_rem))

            bboxes.append((box,score, idx))       

        idx_interested = []
        for bb in bboxes:
            idx_interested.append(bb[2])

    
        obj_anns = []    
    
        for idx, detection in enumerate(detections):
            if idx not in idx_interested:
                continue
        
            label = detection.label
            c = labels.index(label)
            box = detection.box
            score = detection.score
            mask = detection.mask

            min_x = box.xmin
            min_y = box.ymin
            max_x = box.xmax
            max_y = box.ymax
        
            
            #Calculate region classification based on bounding box center
            bbox_center_x = (min_x + max_x) / 2
            bbox_center_y = (min_y + max_y) / 2
            bbdet = []
            bbdet.append(bbox_center_x)
            bbdet.append(bbox_center_y)

            vec = find_match(bbdet, GT_Obj_feat, Obj_pixel)
            region_label = classify_region(bbox_center_x, bbox_center_y)
            
            obj_ann = {
                                    'mask': mask.tolist(),
                                    'category_idx': c,
                                    'image_idx': image_idx,
                                    'feature_vector': vec,
                                    'region':region_label,
                                    'score': score,
                         }
            obj_anns.append(obj_ann)

                
        scene_graph = {"image_id": image_url, "objects": obj_anns}
        
        img_anns.append(obj_anns)
        if count%20 == 0:
            with open("/users/sbsh670/ns-vqa/data/attr_net/objects/img_anns.pkl", "wb") as f:
                pickle.dump(img_anns, f)
            print('Saved checkpoint at image {count}')
        print('Appended for: ', image_url)

    
    with open("/users/sbsh670/ns-vqa/data/attr_net/objects/img_anns.pkl", "wb") as f:
        pickle.dump(img_anns, f)

    print("Final save completed")
    
    all_objs = [obj_ann for img_ann in img_anns for obj_ann in img_ann]

    obj_masks = [o['mask'] for o in all_objs]
    img_ids = [o['image_idx'] for o in all_objs]
    cat_ids = [o['category_idx'] for o in all_objs]
    scores = [o['score'] for o in all_objs]
    if scenes is not None:
        feat_vecs = [o['feature_vector'] for o in all_objs]
    else:
        feat_vecs = []
    output = {
        'object_masks': obj_masks,
        'image_idxs': img_ids,
        'category_idxs': cat_ids,
        'feature_vectors': feat_vecs,
        'scores': scores,
    }
    
    proposal_path = '/users/sbsh670/ns-vqa/data/attr_net/objects/clevr_train_objs_pretrained.json'
    print('| saving object annotations to %s' %proposal_path)
    with open(proposal_path, 'w') as fout:
        json.dump(output, fout)

def attribute_detection(img_dir, proposal_file, output_path, checkpoint_path):
    attr_correct = {
    'shape': 0,
    'size': 0,
    'material': 0,
    'color': 0
    }

    attr_total = {
    'shape': 0,
    'size': 0,
    'material': 0,
    'color': 0
    }
    
    opt = get_options('test')
    opt.clevr_val_ann_path = proposal_file
    opt.dataset = 'clevrer'
    opt.clevr_val_img_dir = img_dir
    opt.load_checkpoint_path = checkpoint_path   
    opt.output_path = output_path
    
    test_loader = get_dataloader(opt, 'test') #test
    model = get_model(opt)

    if opt.use_cat_label:
        with open(COMP_CAT_DICT_PATH) as f:
            cat_dict = utils.invert_dict(json.load(f))

    if opt.dataset == 'clevr':
        scenes = [{
        'image_index': i,
        'image_filename': 'CLEVR_val_%06d.png' % i,
        'objects': []
        } for i in range(15000)]
   
    if opt.dataset == 'clevrer':
        scenes = [{
        'image_index': i,
        'image_filename': '%06d.jpg' % i,
        'objects': []
        } for i in range(15000)]

    count = 0
    for data, labels, idxs, cat_idxs in test_loader:
        model.set_input(data)
        model.forward()
        pred = model.get_pred()
        print('Predicted')       
        for i in range(pred.shape[0]):
            if opt.dataset == 'clevr' or opt.dataset =='clevrer':
                img_id = idxs[i]
                obj = utils.get_attrs_clevr(pred[i])
                #obj_gnd = utils.get_attrs_clevr(labels[i]) #feature vector for ground truth is empty
                #print('Obj_pred:', obj)
                #for key in ['shape', 'size', 'material', 'color']:
                #    if obj[key] == obj_gnd[key]:
                #        attr_correct[key] += 1
                #    attr_total[key] += 1
                
                    
                if opt.use_cat_label:
                    cid = cat_idxs[i] if isinstance(cat_idxs[i], int) else cat_idxs[i].item()
                    obj['color'], obj['material'], obj['shape'] = cat_dict[cid].split(' ')
            
            scenes[int(img_id)]['objects'].append(obj)
        count += idxs.size(0)
        print('%d / %d objects processed' % (count, len(test_loader.dataset)))

    output = {  
        'info': '%s derendered scene' % opt.dataset,
        'scenes': scenes,
    }
    print('| saving annotation file to %s' % opt.output_path)
    utils.mkdirs(os.path.dirname(opt.output_path))
    
    #for key in attr_correct:
    #    acc = attr_correct[key] / attr_total[key] if attr_total[key] > 0 else 0
    #    print(f'{key} accuracy: {acc:.4f}')
    #total_correct = sum(attr_correct.values())
    #total_attrs = sum(attr_total.values())

    #print(f'Overall attribute accuracy: {total_correct / total_attrs:.4f}')
    with open(output_path, 'w') as fout:
        json.dump(output, fout)
    



    
#return object proposals for clevr scenes from original CLEVR dataset
def create_object_proposals_frames(video_path, proposal_path):

    detector_id = "IDEA-Research/grounding-dino-tiny"
    segmenter_id = "facebook/sam-vit-base"
    detector, segmentator, processor = create_pipeline_and_processor(detector_id, segmenter_id)
    shapes = ['sphere', 'cube', 'cylinder']
    labels = ['sphere .', 'cube .', 'cylinder .']
    threshold = 0.3
         
    data_loc = video_path 
    img_anns = []
    count = 0
    
    for frame in sorted(video_path.glob("*.jpg")):  
    #for i,s in enumerate(scenes):
       
        print('Processing frame:', frame)
        count = count+1
        
        #image_url = data_loc+ s['filename'] #"CLEVR_val_000000.png"
        image_idx = frame.stem #s['image_idx']
        
       
        #no ground truth scene graph available
        #GT_Obj_feat = {}
        #Obj_pixel = {}
        #o_id = 0
        #for o in scenes[i]['objects']:
        #    GT_Obj_feat [o_id] = get_feat_vec_clevr(o)
        #    Obj_pixel[o_id] = o['pixel_coords']
        #    o_id = o_id + 1

        
        image_array, detections = grounded_segmentation(
        image=str(frame),
        labels=labels,
        threshold=threshold,
        polygon_refinement=True,
        detector=detector,           
        segmentator=segmentator,     
        processor=processor)
        
        bboxes = []
    
        for idx, detection in enumerate(detections):
            present = False
            p_s = 0
            p_b = None
            box = detection.box
            score = detection.score
            for bb_score in bboxes:
                bb = bb_score[0]
                if bbox_iou(bb, box)> 0.2:
                    p_s = bb_score[1]
                    p_b = bb
                    idx_rem =  bb_score[2]
                    present = True
                    break
        
            if present:
                if p_s>=score:
                    continue 
                else:
                    bboxes.remove((p_b, p_s, idx_rem))

            bboxes.append((box,score, idx))       

        idx_interested = []
        for bb in bboxes:
            idx_interested.append(bb[2])

    
        obj_anns = []    
    
        for idx, detection in enumerate(detections):
            if idx not in idx_interested:
                continue
        
            label = detection.label
            c = labels.index(label)
            box = detection.box
            score = detection.score
            mask = detection.mask

            min_x = box.xmin
            min_y = box.ymin
            max_x = box.xmax
            max_y = box.ymax
        
            
            #Calculate region classification based on bounding box center
            bbox_center_x = (min_x + max_x) / 2
            bbox_center_y = (min_y + max_y) / 2
            bbdet = []
            bbdet.append(bbox_center_x)
            bbdet.append(bbox_center_y)

            
            obj_ann = {
                                    'mask': mask.tolist(),
                                    'category_idx': c,
                                    'image_idx': image_idx,
                                    'score': score,
                         }
            obj_anns.append(obj_ann)

                
        
        
        img_anns.append(obj_anns)
        if count%20 == 0:
            with open("/users/sbsh670/ns-vqa/data/attr_net/objects/clevrer_img_anns.pkl", "wb") as f:
                pickle.dump(img_anns, f)
            print('Saved checkpoint at image {count}')
        print('Appended for: ', video_path, image_idx)

    
    with open("/users/sbsh670/ns-vqa/data/attr_net/objects/clevrer_img_anns.pkl", "wb") as f:
        pickle.dump(img_anns, f)

    print("Final save completed")
    
    all_objs = [obj_ann for img_ann in img_anns for obj_ann in img_ann]

    obj_masks = [o['mask'] for o in all_objs]
    img_ids = [o['image_idx'] for o in all_objs]
    cat_ids = [o['category_idx'] for o in all_objs]
    scores = [o['score'] for o in all_objs]
    feat_vecs = [] #as we do not have gt scene graph
    output = {
        'object_masks': obj_masks,
        'image_idxs': img_ids,
        'category_idxs': cat_ids,
        'feature_vectors': feat_vecs,
        'scores': scores,
    }
    
    #proposal_path = '/users/sbsh670/ns-vqa/data/attr_net/objects/clevr_train_objs_pretrained.json'
    print('| saving object annotations to %s' %proposal_path)
    with open(proposal_path, 'w') as fout:
        json.dump(output, fout)



def main(args):
    
    #--------------------------------
    #for clevr train images
    #create_object_proposals(folder)
    #attribute_detection()
    #------------------------------
    
    #processing for clevrer videos
    
    #path to clevrer videos
    clevrer_videos_path = Path('/users/sbsh670/data/data/eval_video/MVBench/video_14400frames_fps1/clevrer/clevrer/video_validation')

    #path where the object proposal json file from create_object_proposals_frames() would be saved
    proposal_path = Path('/users/sbsh670/ns-vqa/data/attr_net/objects/clevrer')
    
    #path where attribute network output - that is the scene graphs (named as video_id.json) are saved
    output_path = Path('/users/sbsh670/ns-vqa/data/attr_net/results/clevrer')
    
    #path to attribute detection model (pretrained on clevr)
    checkpoint_model = '/users/sbsh670/ns-vqa/data/pretrained/attribute_net.pt'
    
    for folder in sorted(clevrer_videos_path.glob("video_*")):
        if folder.is_dir():
            video_id = folder.name
            print('Video_id:', video_id)
            proposal_file = proposal_path / f"{video_id}_objs_proposals.json"
            print('Proposal file path:', proposal_file)
            output_file = output_path / f"{video_id}.json"
            #pass video_dir + the path to json file where proposals would be saved
            create_object_proposals_frames(folder, proposal_file)
            #pass to attribute_detection : video_directory having frames, the file with object proposals ffrom create_object_proposals_frames() and the path to output json file where scene graphs for frames would be saved and path to chckpoint
            attribute_detection(folder, proposal_file, output_file, checkpoint_model)
        break #I am only testing on one clevrer video for now.

if __name__ == "__main__":
    main(None)



     
