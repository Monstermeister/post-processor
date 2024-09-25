#!/usr/bin/env python3

import re
import objectdetect


def tflite_run(
  model_name,                           # model name as identifier
  model_file,                           # TFLite model
  nimages=None                          # # of images to inference
):
  image_indir   =  "/data/image/coco/validate"
  image_outdir  = f"scratch/{model_name}"

  print(f"""\
================================================================================
Object Detection
--------------------------------------------------------------------------------
  model name             : {model_name}
  image input  directory : {image_indir}
  image output directory : {image_outdir}
""")

  detector      = objectdetect.ObjectDetector(
    model_name,                         # model name as identifier
    model_file,                         # model file
    dataset="coco80",                   # dataset identifier
    classes=80,                         # number of classes
    keep_aspectratio=True,              # keep aspect ratio?
    workdir=image_outdir                # working directory
  )

  ##  (Test-1) Inference an image.
  print("[I] Inference an image.")
  scores,classes,boxes  = detector.inference("data/ILSVRC2012_val_00000573.jpg", score_thresh=0.30, draw=True)
  print(scores)
  print(classes)
  print(boxes)
  print()

  ##  (Test-2) Inference images in {image_indir}.
  """
  detector.inference_images(image_indir, nimages=nimages, score_thresh=0.30)
  """

  print("""\
================================================================================
""")


if (__name__=="__main__"):
  for model_name in [
    "yolov2_tiny",
    #"yolov3_tiny",
    #"yolov3"
  ]:
    tflite_run(model_name, f"data/{model_name}.tflite", nimages=10)
