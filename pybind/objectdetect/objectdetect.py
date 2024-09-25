import tensorflow as tf
import numpy as np
import PIL.Image      as Image
import PIL.ImageOps   as ImageOps
import PIL.ImageColor as ImageColor
import PIL.ImageDraw  as ImageDraw
import PIL.ImageFont  as ImageFont
import os
import glob
import math
import abc
import collections
import six


##==================================================================================================
##  Object Detection.
##==================================================================================================

def ObjectDetector(
  model_name,                           # model name as identifier
  model_file,                           # model file
  dataset="coco80",                     # dataset identifier
  classes=80,                           # number of classes
  keep_aspectratio=True,                # keep aspect ratio?
  workdir="scratch"                     # working directory
):
  if   (model_name.startswith("efficientdet" ) or
        model_name.startswith("ssd_mobilenet")):
    return  _ObjectDetector_TFLCustom(
      model_name, model_file, dataset,
      keep_aspectratio=keep_aspectratio,
      preproc="minmax",
      workdir=workdir
    )
  elif (model_name.startswith("yolo")):
    return  _ObjectDetector_Yolo(
      model_name, model_file, dataset, classes,
      keep_aspectratio=keep_aspectratio,
      preproc=None,
      workdir=workdir
    )
  else: raise ValueError(f"Unsupported model_name=\"{model_name}\" provided")


class _ObjectDetector:

  def __init__(self,
    model_name,                         # model name as identifier
    model_file,                         # model file
    dataset,                            # dataset identifier
    classes,                            # number of classes
    keep_aspectratio,                   # keep aspect ratio?
    preproc,                            # preprocessor type
    workdir="./scratch"                 # working directory
  ):
    if not(os.path.exists(workdir)): os.system(f"mkdir -p {workdir}")

    _interpreter          = tf.lite.Interpreter(model_file)
    _runner               = _interpreter.get_signature_runner()
    self.model_name       = model_name
    self.runner           = _runner
    _input_details        = _runner.get_input_details()
    self.input_name       = list(_input_details.keys())[0]
    self.input_shape      = _input_details[self.input_name]["shape"][-3:]
    _output_details       = _runner.get_output_details()
    _output_shapes        = [np.prod(_["shape"]) for _ in _output_details.values()]
    _output_names         = list(_output_details.keys())
    self.output_names     = [_output_names[_] for _ in np.argsort(_output_shapes)]

    self.keep_aspectratio = keep_aspectratio
    self.preproc          = preproc
    self.labels           = labels[dataset]
    self.classes          = classes
    self.workdir          = workdir
    self._postproc_ratio  = None


  def preprocess(self,
    image_file                          # image file
  ):
    ##  Resize an image.
    image   = Image.open(image_file).convert("RGB")
    IS      = max(image.size)           # IS=max(IW,IH)
    if (self.keep_aspectratio):
          image = ImageOps.pad(image, (IS,IS), centering=(0,0))
    else: image = image.resize((IS,IS))

    ##  Preprocess the image.
    XH,XW   = self.input_shape[:2]      # input  height, width
    XS      = max(XH,XW)
    inputs  = np.asarray(image.resize((XW,XH)), dtype=np.float32)

    if   (self.preproc=="stats" ):      # Torch-style
      inputs  = ((inputs-[123.675,116.280,103.530])/[58.395,57.120,57.375]).astype(np.float32)
    elif (self.preproc=="minmax"):      # TensorFlow-style
      inputs  = (inputs-127)/128
    else:                               # raw 8-bit data
      inputs  =  inputs/256

    inputs                = np.expand_dims(inputs, axis=0)
    self._postproc_ratio  = IS/XS
    return  [inputs, image]


  @abc.abstractmethod
  def inference(self,
    image_file,                         # image file
    score_thresh=0.20,                  # score threshold
    draw=True                           # draw overlaid image?
  ):  pass


  def inference_images(self,
    image_dir,                          # directory containing images
    nimages=None,                       # # of images to inference
    score_thresh=0.20                   # score threshold
  ):
    image_files = glob.glob(f"{image_dir}/*.jpg" )  \
                + glob.glob(f"{image_dir}/*.JPEG")

    if not(nimages):  nimages = len(image_files)
    _n          = math.ceil(math.log10(nimages))  # # of digits formatted
    print(f"[I] Inference {nimages} images in {image_dir}")

    for i in range(nimages):
      print(f"\r    {i+1:{_n}d}/{nimages}    {image_files[i].split('/')[-1]}", end="", flush=True)
      self.inference(image_files[i], score_thresh=score_thresh, draw=True)
      if (i%100==99 or i==nimages-1): print()
    print(f"Annotated images saved in {self.workdir}")


  def postprocess_yolo(self,
    predicts,                           # outputs containing w,h
    score_thresh=0.20,
    iou_thresh=0.60
  ):
    if   (self.model_name=="yolov2_tiny"):
      pyramids      = 1
      YOLO_ANCHORS  = [
        [[9.164,10.838], [29.991,33.001], [53.415,87.589], [126.125,56.445], [156.328,146.693]]
      ]
    elif (self.model_name=="yolov3_tiny"):
      pyramids      = 2
      YOLO_ANCHORS  = [                 # preset anchors in half size
        [[ 40.5, 41.0], [ 67.5, 84.5], [172.0,159.5]],  # large  box
        [[ 11.5, 13.5], [ 18.5, 29.5], [ 40.5, 41.0]]   # medium box
      ]
    elif (self.model_name=="yolov3"):
      pyramids      = 3
      YOLO_ANCHORS  = [                 # preset anchors in half size
        [[ 58.0, 45.0], [ 78.0, 99.0], [186.5,163.0]],  # large  box
        [[ 15.0, 30.5], [ 31.0, 22.5], [ 29.5, 59.5]],  # medium box
        [[  5.0,  6.5], [  8.0, 15.0], [ 16.5, 11.5]]   # small  box
      ]
    else:
      raise ValueError(f"[ERROR] Unknown model_name=\"{self.model_name}\" provided")

    ##  (1) Filter by class scores.
    boxes,classes,scores  = [],[],[]
    _k                    = self._postproc_ratio

    for p, anchors in enumerate(YOLO_ANCHORS):
      ##  Configure hyper-parameters for each pyramid level.
      if (p==0):
        sample  = 32
        GH      = self.input_shape[0]//sample
        GW      = self.input_shape[1]//sample
        nboxes  = GH*GW*len(YOLO_ANCHORS[0])
        boxid0  = 0
        boxidN  = nboxes
      else:
        sample /= 2
        GH     *= 2
        GW     *= 2
        nboxes *= 4
        boxid0  = boxidN
        boxidN += nboxes

      ##  Decode the coordinates for boxes exceeding the score threshold.
      for boxid in range(boxid0, boxidN):
        pred    = predicts[boxid]             # encoded w,h
        classid = np.argmax(pred[5:])         # pick the top-1 class
        score   = __class__._sigmoid(pred[4]) \
                * __class__._sigmoid(pred[5+classid])

        if (score>=score_thresh):
          boxideff  =  boxid-boxid0             # box index in pyramid
          a     =  boxideff% len(anchors)       # anchor index
          xg    = (boxideff//len(anchors))% GW  # grid reference xg
          yg    = (boxideff//len(anchors))//GW  # grid reference yg
          xc    = (xg+__class__._sigmoid(pred[0]))*sample
          yc    = (yg+__class__._sigmoid(pred[1]))*sample
          w_2   = np.exp(pred[2])*anchors[a][0] # width  w
          h_2   = np.exp(pred[3])*anchors[a][1] # height h
          box   = [_k*(xc-w_2), _k*(yc-h_2), _k*(xc+w_2), _k*(yc+h_2)]
          boxes  .append(box    )
          classes.append(classid)
          scores .append(score  )

    ##  (2) Perform IOU operation.
    boxes_valid   = [True]*len(boxes)           # set all boxes valid initially
    boxes_area    = [_[2]*_[3] for _ in boxes]  # precompute the box area
    score_orders  = np.argsort(scores)[::-1]    # set the NMS order

    for i, id_i in enumerate(score_orders):
      if not(boxes_valid[id_i]):  continue

      for id_j in score_orders[i+1:]:
        if (not(boxes_valid[id_j]) or classes[id_i]!=classes[id_j]): continue

        x0  = max(boxes[id_i][0],                boxes[id_j][0])
        y0  = max(boxes[id_i][1],                boxes[id_j][1])
        x1  = min(boxes[id_i][0]+boxes[id_i][2], boxes[id_j][0]+boxes[id_j][2])
        y1  = min(boxes[id_i][0]+boxes[id_i][3], boxes[id_j][0]+boxes[id_j][3])
        if (x1<=x0 or y1<=y0):  continue

        area_ixj  = (x1-x0)*(y1-y0)
        area_iuj  = boxes_area[id_i]+boxes_area[id_j]-area_ixj
        if (area_ixj>=iou_thresh*area_iuj): boxes_valid[id_j] = 0

    ##  Detection outputs.
    detects     = []
    for id_i in score_orders:
      if (boxes_valid[id_i]): detects.append(id_i)

    return [
      np.asarray(scores )[detects],     # post-NMS scores
      np.asarray(classes)[detects],     # post-NMS classes 
      np.asarray(boxes  )[detects]      # post-NMS boxes in XYWH format
    ]


  def visualize(self,
    image,                              # PIL Image object
    scores,                             # list of scores
    classes,                            # list of classes
    boxes,                              # list of boxes
    box_format,                         # box coordinates format
    use_normalized_coordinates=True,    # coordinates normalized?
    max_boxes_to_draw=10,               # # of boxes to draw
    line_thickness=1,                   # line thickness
    label_fontsize=10                   # label font size
  ):
    return  visualize_boxes_and_labels_on_image(
      image, scores, classes, boxes, self.labels,
      box_format=box_format,
      use_normalized_coordinates=use_normalized_coordinates,
      max_boxes_to_draw=max_boxes_to_draw,
      line_thickness=line_thickness,
      label_fontsize=label_fontsize
    )

  
  @staticmethod
  def _sigmoid(x):  return 1/(1+np.exp(-x))

##  End of class _ObjectDetector


"""Object detector for networks ended with TFLite_Detection_PostProcess."""
class _ObjectDetector_TFLCustom(_ObjectDetector):

  def __init__(self,
    model_name,                         # model name as an identifier
    model_file,                         # model file
    dataset,                            # dataset identifier 
    keep_aspectratio=True,              # keep aspect ratio?
    preproc=None,                       # preprocessor type
    workdir="./scratch"                 # working directory
  ):
    super().__init__(
      model_name, model_file, dataset, None, keep_aspectratio, preproc,
      workdir=workdir
    )


  def inference(self,
    image_file,                         # raw image file
    score_thresh=0.20,                  # score threshold
    draw=True                           # draw?
  ):
    inputs,image  = self.preprocess(image_file)
    outputs       = self.runner(**{self.input_name:inputs})
    ndetects      = np.int32(outputs[self.output_names[0]][0])
    scores        =          outputs[self.output_names[1]][0]
    classes       = np.int32(outputs[self.output_names[2]][0])
    boxes         =          outputs[self.output_names[3]][0]

    for i in range(ndetects):
      if (scores[i]<score_thresh):
        ndetects  = i
        scores    = scores [:i]
        classes   = classes[:i]
        boxes     = boxes  [:i]
        break

    if (draw):
      file_id     = image_file.split("/")[-1].split(".")[0]
      self.visualize(
        image, scores, classes, boxes, "yxyx",
      ).save(f"{self.workdir}/{file_id}.png")
    return  scores, classes, boxes

##  End of class _ObjectDetector_TFLCustom


"""Object detector for Yolo networks"""
class _ObjectDetector_Yolo(_ObjectDetector):

  def __init__(self,
    model_name,                         # model name as an identifier
    model_file,                         # model file
    dataset,                            # dataset identifier
    classes,                            # number of classes
    keep_aspectratio=True,              # keep aspect ratio?
    preproc=None,                       # preprocessor type
    workdir="./scratch"                 # working directory
  ):
    super().__init__(
      model_name, model_file, dataset, classes, keep_aspectratio, preproc,
      workdir=workdir
    )


  def inference(self,
    image_file,                         # raw image file
    score_thresh=0.20,                  # score threshold
    iou_thresh=0.60,
    draw=True 
  ):
    inputs,image  = self.preprocess(image_file)
    outputs       = self.runner(**{self.input_name:inputs})

    if   (self.model_name=="yolov2_tiny"):
      predicts    = np.reshape(outputs[self.output_names[0]], (-1,self.classes+5))
    elif (self.model_name=="yolov3_tiny"):
      predicts    = np.concatenate([
        np.reshape(outputs[self.output_names[0]], (-1,self.classes+5)),
        np.reshape(outputs[self.output_names[1]], (-1,self.classes+5))
      ])
    elif (self.model_name=="yolov3"):
      predicts    = np.concatenate([
        np.reshape(outputs[self.output_names[0]], (-1,self.classes+5)),
        np.reshape(outputs[self.output_names[1]], (-1,self.classes+5)),
        np.reshape(outputs[self.output_names[2]], (-1,self.classes+5))
      ])
    else:
      raise ValueError(f"Unknown model_name=\"{model_name}\" provided")
    
    [scores,classes,boxes]  = self.postprocess_yolo(
      predicts,
      score_thresh=score_thresh,
      iou_thresh=iou_thresh
    )

    if (draw):
      file_id     = image_file.split("/")[-1].split(".")[0]
      self.visualize(
        image, scores, classes, boxes, "xyxy",
        use_normalized_coordinates=False
      ).save(f"{self.workdir}/{file_id}.png")
    return scores,classes,boxes

##  End of class _ObjectDetector_Yolo


##==================================================================================================
##  Label.
##==================================================================================================

_label_coco80 = {
   0: "person",
   1: "bicycle",
   2: "car",
   3: "motorcycle",
   4: "airplane",
   5: "bus",
   6: "train",
   7: "truck",
   8: "boat",
   9: "traffic light",
  10: "fire hydrant",
  11: "stop sign",
  12: "parking meter",
  13: "bench",
  14: "bird",
  15: "cat",
  16: "dog",
  17: "horse",
  18: "sheep",
  19: "cow",
  20: "elephant",
  21: "bear",
  22: "zebra",
  23: "giraffe",
  24: "backpack",
  25: "umbrella",
  26: "handbag",
  27: "tie",
  28: "suitcase",
  29: "frisbee",
  30: "skis",
  31: "snowboard",
  32: "sports ball",
  33: "kite",
  34: "baseball bat",
  35: "baseball glove",
  36: "skateboard",
  37: "surfboard",
  38: "tennis racket",
  39: "bottle",
  40: "wine glass",
  41: "cup",
  42: "fork",
  43: "knife",
  44: "spoon",
  45: "bowl",
  46: "banana",
  47: "apple",
  48: "sandwich",
  49: "orange",
  50: "broccoli",
  51: "carrot",
  52: "hot dog",
  53: "pizza",
  54: "donut",
  55: "cake",
  56: "chair",
  57: "couch",
  58: "potted plant",
  59: "bed",
  60: "dining table",
  61: "toilet",
  62: "tv",
  63: "laptop",
  64: "mouse",
  65: "remote",
  66: "keyboard",
  67: "cell phone",
  68: "microwave",
  69: "oven",
  70: "toaster",
  71: "sink",
  72: "refrigerator",
  73: "book",
  74: "clock",
  75: "vase",
  76: "scissors",
  77: "teddy bear",
  78: "hair drier",
  79: "toothbrush"
}

_label_coco90 = {
   0: "person",
   1: "bicycle",
   2: "car",
   3: "motorcycle",
   4: "airplane",
   5: "bus",
   6: "train",
   7: "truck",
   8: "boat",
   9: "traffic light",
  10: "fire hydrant",
  12: "stop sign",
  13: "parking meter",
  14: "bench",
  15: "bird",
  16: "cat",
  17: "dog",
  18: "horse",
  19: "sheep",
  20: "cow",
  21: "elephant",
  22: "bear",
  23: "zebra",
  24: "giraffe",
  26: "backpack",
  27: "umbrella",
  30: "handbag",
  31: "tie",
  32: "suitcase",
  33: "frisbee",
  34: "skis",
  35: "snowboard",
  36: "sports ball",
  37: "kite",
  38: "baseball bat",
  39: "baseball glove",
  40: "skateboard",
  41: "surfboard",
  42: "tennis racket",
  43: "bottle",
  45: "wine glass",
  46: "cup",
  47: "fork",
  48: "knife",
  49: "spoon",
  50: "bowl",
  51: "banana",
  52: "apple",
  53: "sandwich",
  54: "orange",
  55: "broccoli",
  56: "carrot",
  57: "hot dog",
  58: "pizza",
  59: "donut",
  60: "cake",
  61: "chair",
  62: "couch",
  63: "potted plant",
  64: "bed",
  66: "dining table",
  69: "toilet",
  71: "tv",
  72: "laptop",
  73: "mouse",
  74: "remote",
  75: "keyboard",
  76: "cell phone",
  77: "microwave",
  78: "oven",
  79: "toaster",
  80: "sink",
  81: "refrigerator",
  83: "book",
  84: "clock",
  85: "vase",
  86: "scissors",
  87: "teddy bear",
  88: "hair drier",
  89: "toothbrush",
}

_label_keti6  = {
   0: "car",
   1: "truck",
   2: "person",
   3: "trafficcone",
   4: "cycle",
   5: "bus"
}

_label_keti6y = {
   0: "person",
   1: "trafficcone",
   2: "cycle",
   3: "bus",
   4: "car",
   5: "truck"
}
_label_keti7  = {
   0: "person",
   1: "trafficcone",
   2: "cycle",
   3: "bus",
   4: "car",
   5: "truck"
}

##  Exported sets of labels.
labels       = {
  "coco80"  : {_   : {"id":_,  "name":_label_coco80[_]} for _ in _label_coco80},
  "coco90"  : {_   : {"id":_,  "name":_label_coco90[_]} for _ in _label_coco90},
  "keti6"   : {_   : {"id":_,  "name":_label_keti6 [_]} for _ in _label_keti6 },
  "keti6y"  : {_   : {"id":_,  "name":_label_keti6y[_]} for _ in _label_keti6y},
  "keti7"   : {_   : {"id":_,  "name":_label_keti7 [_]} for _ in _label_keti7 }
}


##==================================================================================================
##  Visualize.
##==================================================================================================

_TITLE_LEFT_MARGIN = 10
_TITLE_TOP_MARGIN = 10
STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]


def _get_multiplier_for_color_randomness():
  """Returns a multiplier to get semi-random colors from successive indices.

  This function computes a prime number, p, in the range [2, 17] that:
  - is closest to len(STANDARD_COLORS) / 10
  - does not divide len(STANDARD_COLORS)

  If no prime numbers in that range satisfy the constraints, p is returned as 1.

  Once p is established, it can be used as a multiplier to select
  non-consecutive colors from STANDARD_COLORS:
  colors = [(p * i) % len(STANDARD_COLORS) for i in range(20)]
  """
  num_colors = len(STANDARD_COLORS)
  prime_candidates = [5, 7, 11, 13, 17]

  # Remove all prime candidates that divide the number of colors.
  prime_candidates = [p for p in prime_candidates if num_colors % p]
  if not prime_candidates:
    return 1

  # Return the closest prime number to num_colors / 10.
  abs_distance = [np.abs(num_colors / 10. - p) for p in prime_candidates]
  num_candidates = len(abs_distance)
  inds = [i for _, i in sorted(zip(abs_distance, range(num_candidates)))]
  return prime_candidates[inds[0]]


def _limit_boxcoord(x, xmin, xmax):
  return min(max(x,xmin), xmax)


def draw_bounding_box_on_image(
  image, box,
  box_format="xywh",
  color='red',
  thickness=4,
  fontsize=10,
  display_str_list=(),
  use_normalized_coordinates=True
):
  """Adds a bounding box to an image.

  Bounding box coordinates can be specified in either absolute (pixel) or
  normalized coordinates by setting the use_normalized_coordinates argument.

  Each string in display_str_list is displayed on a separate line above the
  bounding box in black text on a rectangle filled with the input 'color'.
  If the top of the bounding box extends to the edge of the image, the strings
  are displayed below the bounding box.

  Args:
    image: a PIL.Image object.
    ymin: ymin of bounding box.
    xmin: xmin of bounding box.
    ymax: ymax of bounding box.
    xmax: xmax of bounding box.
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list: list of strings to display in box (each to be shown on its
      own line).
    use_normalized_coordinates: If True (default), treat coordinates ymin, xmin,
      ymax, xmax as relative to the image.  Otherwise treat coordinates as
      absolute.
  """
  draw  = ImageDraw.Draw(image)
  IW,IH = image.size
  if   (box_format=="xywh"):
    [xmin,ymin,xmax,ymax] = [box[0],box[1],box[0]+box[2],box[1]+box[3]]
  elif (box_format=="xyxy"):
    [xmin,ymin,xmax,ymax] =  box
  elif (box_format=="yxyx"):
    [ymin,xmin,ymax,xmax] =  box
  else: raise ValueError(f"Unknown box_format=\"{box_format}\" provided.")

  if use_normalized_coordinates:
    ymin  = _limit_boxcoord(ymin*IH, 0, IH          )
    xmin  = _limit_boxcoord(xmin*IW, 0, IW          )
    ymax  = _limit_boxcoord(ymax*IH, 0, IH-thickness)
    xmax  = _limit_boxcoord(xmax*IW, 0, IW-thickness)
  else:
    ymin  = _limit_boxcoord(ymin   , 0, IH          )
    xmin  = _limit_boxcoord(xmin   , 0, IW          )
    ymax  = _limit_boxcoord(ymax   , 0, IH-thickness)
    xmax  = _limit_boxcoord(xmax   , 0, IW-thickness)

  if (thickness>0):
    draw.line(
      [(xmin,ymin),(xmin,ymax),(xmax,ymax),(xmax,ymin),(xmin,ymin)],
      width=thickness, fill=color
    )

  try:            font = ImageFont.truetype('arial.ttf', fontsize)
  except IOError: font = ImageFont.load_default()

  # If the total height of the display strings added to the top of the bounding
  # box exceeds the top of the image, stack the strings below the bounding box
  # instead of above.
  display_str_heights = [font.getbbox(_)[3]-font.getbbox(_)[1] for _ in display_str_list]
  # Each display_str has a top and bottom margin of 0.05x.
  total_display_str_height = (1 + 2*0.05) * sum(display_str_heights)

  if (ymin>total_display_str_height):
        text_bottom = ymin
  else: text_bottom = ymax+total_display_str_height

  # Reverse list and print from bottom to top.
  for display_str in display_str_list[::-1]:
    x0,y0,x1,y1 = font.getbbox(display_str)
    x0         += xmin
    x1         += xmin
    text_width, text_height = (x1-x0, y1-y0)
    margin                  = np.ceil(0.05*text_height)
    draw.rectangle([(x0, text_bottom-text_height-2*margin),
                    (x0+text_width, text_bottom)],
                   fill=color)
    draw.text((x0+margin, text_bottom-text_height-margin), display_str,
              fill='black', font=font)
    text_bottom -= text_height-2*margin


def draw_keypoints_on_image(
  image,
  keypoints,
  color='red',
  radius=2,
  use_normalized_coordinates=True,
  keypoint_edges=None,
  keypoint_edge_color='green',
  keypoint_edge_width=2
):
  """Draws keypoints on an image.

  Args:
    image: a PIL.Image object.
    keypoints: a numpy array with shape [num_keypoints, 2].
    color: color to draw the keypoints with. Default is red.
    radius: keypoint radius. Default value is 2.
    use_normalized_coordinates: if True (default), treat keypoint values as
      relative to the image.  Otherwise treat them as absolute.
    keypoint_edges: A list of tuples with keypoint indices that specify which
      keypoints should be connected by an edge, e.g. [(0, 1), (2, 4)] draws
      edges from keypoint 0 to 1 and from keypoint 2 to 4.
    keypoint_edge_color: color to draw the keypoint edges with. Default is red.
    keypoint_edge_width: width of the edges drawn between keypoints. Default
      value is 2.
  """
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  keypoints_x = [k[1] for k in keypoints]
  keypoints_y = [k[0] for k in keypoints]
  if use_normalized_coordinates:
    keypoints_x = tuple([im_width * x for x in keypoints_x])
    keypoints_y = tuple([im_height * y for y in keypoints_y])
  for keypoint_x, keypoint_y in zip(keypoints_x, keypoints_y):
    draw.ellipse([(keypoint_x - radius, keypoint_y - radius),
                  (keypoint_x + radius, keypoint_y + radius)],
                 outline=color,
                 fill=color)
  if keypoint_edges is not None:
    for keypoint_start, keypoint_end in keypoint_edges:
      if (keypoint_start < 0 or keypoint_start >= len(keypoints) or
          keypoint_end < 0 or keypoint_end >= len(keypoints)):
        continue
      edge_coordinates = [
          keypoints_x[keypoint_start], keypoints_y[keypoint_start],
          keypoints_x[keypoint_end], keypoints_y[keypoint_end]
      ]
      draw.line(
          edge_coordinates, fill=keypoint_edge_color, width=keypoint_edge_width)


def draw_mask_on_image(image, mask, color='red', alpha=0.4):
  """Draws mask on an image.

  Args:
    image: a PIL.Image object
    mask: a uint8 numpy array of shape (img_height, img_width) with values
      between either 0 or 1.
    color: color to draw the keypoints with. Default is red.
    alpha: transparency value between 0 and 1. (default: 0.4)

  Raises:
    ValueError: On incorrect data type for image or masks.
  """
  if mask.dtype != np.uint8:
    raise ValueError('`mask` not of type np.uint8')
  if np.any(np.logical_and(mask != 1, mask != 0)):
    raise ValueError('`mask` elements should be in [0, 1]')
  if (image.size[0]!=mask.shape[1] or image.size[1]!=mask.shape[0]):
    raise ValueError("The image size %s is not matched with the mask shape %s"
      %(image.size, mask.shape))
  rgb = ImageColor.getrgb(color)

  solid_color = np.expand_dims(
      np.ones_like(mask), axis=2) * np.reshape(list(rgb), [1, 1, 3])
  pil_solid_color = Image.fromarray(np.uint8(solid_color)).convert('RGBA')
  pil_mask = Image.fromarray(np.uint8(255.0 * alpha * mask)).convert('L')
  image = Image.composite(pil_solid_color, image, pil_mask)


def visualize_boxes_and_labels_on_image(
    image,
    scores,
    classes,
    boxes,
    category_index,
    instance_masks=None,
    instance_boundaries=None,
    keypoints=None,
    keypoint_edges=None,
    track_ids=None,
    box_format="xywh",
    use_normalized_coordinates=True,
    max_boxes_to_draw=20,
    agnostic_mode=False,
    line_thickness=2,
    label_fontsize=10,
    groundtruth_box_visualization_color='black',
    skip_boxes=False,
    skip_scores=False,
    skip_labels=False,
    skip_track_ids=False):
  """Overlay labeled boxes on an image with formatted scores and label names.

  This function groups boxes that correspond to the same location
  and creates a display string for each detection and overlays these
  on the image. Note that this function modifies the image in place, and returns
  that same image.

  Args:
    image: uint8 numpy array with shape (img_height, img_width, 3)
    boxes: a numpy array of shape [N, 4]
    classes: a numpy array of shape [N]. Note that class indices are 1-based,
      and match the keys in the label map.
    scores: a numpy array of shape [N] or None.  If scores=None, then this
      function assumes that the boxes to be plotted are groundtruth boxes and
      plot all boxes as black with no classes or scores.
    category_index: a dict containing category dictionaries (each holding
      category index `id` and category name `name`) keyed by category indices.
    instance_masks: a numpy array of shape [N, image_height, image_width] with
      values ranging between 0 and 1, can be None.
    instance_boundaries: a numpy array of shape [N, image_height, image_width]
      with values ranging between 0 and 1, can be None.
    keypoints: a numpy array of shape [N, num_keypoints, 2], can be None
    keypoint_edges: A list of tuples with keypoint indices that specify which
      keypoints should be connected by an edge, e.g. [(0, 1), (2, 4)] draws
      edges from keypoint 0 to 1 and from keypoint 2 to 4.
    track_ids: a numpy array of shape [N] with unique track ids. If provided,
      color-coding of boxes will be determined by these ids, and not the class
      indices.
    box_format: box coordinates format
    use_normalized_coordinates: whether boxes is to be interpreted as normalized
      coordinates or not.
    max_boxes_to_draw: maximum number of boxes to visualize.  If None, draw all
      boxes.
    agnostic_mode: boolean (default: False) controlling whether to evaluate in
      class-agnostic mode or not.  This mode will display scores but ignore
      classes.
    line_thickness: integer (default: 4) controlling line width of the boxes.
    groundtruth_box_visualization_color: box color for visualizing groundtruth
      boxes
    skip_boxes: whether to skip the drawing of bounding boxes.
    skip_scores: whether to skip score when drawing a single detection
    skip_labels: whether to skip label when drawing a single detection
    skip_track_ids: whether to skip track id when drawing a single detection

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3) with overlaid boxes.
  """
  # Create a display string (and color) for every box location, group any boxes
  # that correspond to the same location.
  box_to_display_str_map = collections.defaultdict(list)
  box_to_color_map = collections.defaultdict(str)
  box_to_instance_masks_map = {}
  box_to_instance_boundaries_map = {}
  box_to_keypoints_map = collections.defaultdict(list)
  box_to_track_ids_map = {}
  if not max_boxes_to_draw:
    max_boxes_to_draw = boxes.shape[0]
  for i in range(boxes.shape[0]):
    if max_boxes_to_draw == len(box_to_color_map):
      break

    box = tuple(boxes[i].tolist())
    if instance_masks is not None:
      box_to_instance_masks_map[box] = instance_masks[i]
    if instance_boundaries is not None:
      box_to_instance_boundaries_map[box] = instance_boundaries[i]
    if keypoints is not None:
      box_to_keypoints_map[box].extend(keypoints[i])
    if track_ids is not None:
      box_to_track_ids_map[box] = track_ids[i]
    if scores is None:
      box_to_color_map[box] = groundtruth_box_visualization_color
    else:
      display_str = ''
      if not skip_labels:
        if not agnostic_mode:
          if classes[i] in six.viewkeys(category_index):
            class_name = category_index[classes[i]]['name']
          else:
            class_name = 'N/A'
          display_str = str(class_name)
      if not skip_scores:
        if not display_str:
          display_str = '{}%'.format(int(100 * scores[i]))
        else:
          display_str = '{}: {}%'.format(display_str, int(100 * scores[i]))
      if not skip_track_ids and track_ids is not None:
        if not display_str:
          display_str = 'ID {}'.format(track_ids[i])
        else:
          display_str = '{}: ID {}'.format(display_str, track_ids[i])
      box_to_display_str_map[box].append(display_str)
      if agnostic_mode:
        box_to_color_map[box] = 'DarkOrange'
      elif track_ids is not None:
        prime_multipler = _get_multiplier_for_color_randomness()
        box_to_color_map[box] = STANDARD_COLORS[(prime_multipler *
                                                 track_ids[i]) %
                                                len(STANDARD_COLORS)]
      else:
        box_to_color_map[box] = STANDARD_COLORS[classes[i] %
                                                len(STANDARD_COLORS)]

  ##  Draw all boxes onto image.
  for box, color in box_to_color_map.items():
    if (instance_masks):
      draw_mask_on_image(
        image, box_to_instance_masks_map[box],
        color=color
      )
    if (instance_boundaries):
      draw_mask_on_image(
        image, box_to_instance_boundaries_map[box],
        color='red', alpha=1.0
      )
    draw_bounding_box_on_image(
      image, box,
      box_format=box_format,
      color=color,
      thickness=0 if skip_boxes else line_thickness,
      fontsize=label_fontsize,
      display_str_list=box_to_display_str_map[box],
      use_normalized_coordinates=use_normalized_coordinates
    )
    if (keypoints):
      draw_keypoints_on_image(
       image, box_to_keypoints_map[box],
       color=color,
       radius=line_thickness / 2,
       use_normalized_coordinates=use_normalized_coordinates,
       keypoint_edges=keypoint_edges,
       keypoint_edge_color=color,
       keypoint_edge_width=line_thickness // 2
     )
  return  image
