import numpy as np
import PIL.Image, PIL.ImageOps
import build.preprocess as preproc

image = PIL.Image.open("ILSVRC2012_val_00050000.640x480.jpg")
image_data =np.asarray(image).transpose((2,0,1)).tobytes()
fitimage_data1 =preproc.resize_exact(image_data, 416,416).transpose((1,2,0))
fitimage_data2 =preproc.resize_pad(image_data,416,416).transpose((1,2,0))
fitimage_data3 =preproc.resize_crop(image_data,416,416).transpose((1,2,0))

PIL.Image.fromarray(fitimage_data1).save(f"scratch_00050000_res_exact.jpg")
PIL.Image.fromarray(fitimage_data2).save(f"scratch_00050000_res_pad.jpg")
PIL.Image.fromarray(fitimage_data3).save(f"scratch_00050000_res_crop.jpg")
