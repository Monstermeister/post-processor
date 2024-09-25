#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>


#define CAMERA_IH 480
#define CAMERA_IW 640
#define RESIZE_FIT_PADCONST 114


pybind11::array_t<uint8_t> resize_exact(
  pybind11::buffer  image_data,
  int               target_height,
  int               target_width
) {
  uint8_t*              image_databuf = (uint8_t*) image_data.request().ptr;
  std::vector<uint8_t>  fitimage_databuf(3*target_height*target_width);

  float ratio_h = CAMERA_IH/float(target_height);
  float ratio_w = CAMERA_IW/float(target_width );
  for(int index=0, c=0; c<3; c++) {
    for(int h=0; h<target_height; h++) {
      int ih  = static_cast<int>(ratio_h*h+0.5);

      for(int w=0; w<target_width; w++) {
        int iw = static_cast<int>(ratio_w*w+0.5);
      //  printf("[DEBUG] %d <= %d\n", index, CAMERA_IW*CAMERA_IH*c+CAMERA_IW*ih+iw);
      //  fitimage_databuf[index++]=0;
       fitimage_databuf[index++]  = image_databuf[CAMERA_IW*CAMERA_IH*c+CAMERA_IW*ih+iw];
      }
    }
  }

  pybind11::buffer_info result(
    &fitimage_databuf[0],                         // pointer
    1,                                            // element size
    pybind11::format_descriptor<uint8_t>::value,  // format
    3,                                            // dimension
    {3,target_height,target_width},               // shape
    {target_height*target_width,target_width,1}   // stride
  );
  return pybind11::array_t<uint8_t>(result);
}

pybind11::array_t<uint8_t> resize_pad(
  pybind11::buffer image_data,
  int              target_height,
  int              target_width
) {
  uint8_t*              image_databuf = (uint8_t*) image_data.request().ptr;
  std::vector<uint8_t> fitimage_databuf(3*target_height*target_width);

  float ratio_h = CAMERA_IH/float(target_height);
  float ratio_w = CAMERA_IW/float(target_width );
  float ratio   = (ratio_h>ratio_w) ? ratio_h : ratio_w;
  
  for(int index =0, c=0; c<3; c++){
    for(int h =0; h<target_height; h++){
      int ih = static_cast<int>(ratio*h+0.5);
      
      if(ih<CAMERA_IH){
        for(int w=0; w<target_width; w++){
          int iw                      = static_cast<int>(ratio*w+0.5);
          fitimage_databuf[index++] = image_databuf[CAMERA_IW*CAMERA_IH*c + CAMERA_IW*ih +iw];
          
          }
        }
      else{
        for(int w=0; w<target_width; w++){
          fitimage_databuf[index++] = RESIZE_FIT_PADCONST;
          
          }
        }
      }
    }

    pybind11::buffer_info result(
      &fitimage_databuf[0],                        // pointer
      1,                                           // element size
      pybind11::format_descriptor<uint8_t>::value, // format
      3,                                           // dimension
      {3, target_height, target_width},            // shape
      {target_height*target_width, target_width,1} // stride
    );
    return pybind11::array_t<uint8_t>(result);
}

pybind11::array_t<uint8_t> resize_crop(
  pybind11::buffer      image_data,
  int                   target_height,
  int                   target_width
) {
  uint8_t*              image_databuf = (uint8_t*) image_data.request().ptr;
  std::vector<uint8_t>  fitimage_databuf(3*target_height*target_width);

  float ratio_h = CAMERA_IH/float(target_height);
  float ratio_w = CAMERA_IW/float(target_width );
  float ratio   = (ratio_h<ratio_w) ? ratio_h : ratio_w;

  int ih0 = 0;
  int iw0 = 0;
  if        (ratio_h<ratio_w) { 
    iw0 = CAMERA_IW/2-(int(ratio*target_width )>>1);
  } else if (ratio_h>ratio_w) {
    ih0 = CAMERA_IH/2-(int(ratio*target_height)>>1);
  }

  for (int index =0,c=0;c<3;c++){
    for (int h =0; h<target_height; h++){
      int ih = static_cast<int>(ratio*h +0.5);
      for (int w =0; w<target_width; w++){
        int iw = static_cast<int>(ratio*w +0.5);
        fitimage_databuf[index++] = image_databuf[c*CAMERA_IH*CAMERA_IW +(ih0+ih)*CAMERA_IW+(iw0+iw)];
       }
      }
     }

  pybind11::buffer_info result(
   &fitimage_databuf[0],
   1,
   pybind11::format_descriptor<uint8_t>::value,
   3,
   {3,target_height,target_width},
   {target_height*target_width,target_width,1}
  );
  return pybind11::array_t<uint8_t>(result);
 }


PYBIND11_MODULE(preprocess, m){
  m.doc()="preprocess";
  m.def("resize_exact" , &resize_exact);
  m.def("resize_pad"   , &resize_pad  );
  m.def("resize_crop"  , &resize_crop );
}
