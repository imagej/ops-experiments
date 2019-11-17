__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;


__kernel void custom_convolution_3d(
    DTYPE_IMAGE_IN_3D src,
    DTYPE_IMAGE_IN_3D kernelImage,
    DTYPE_IMAGE_OUT_3D dst
) {
  const int i = get_global_id(0);
  const int j = get_global_id(1);
  const int k = get_global_id(2);

  int4 coord = (int4){i, j, k, 0};

  const int kernelWidth = GET_IMAGE_WIDTH(kernelImage);
  const int kernelHeight = GET_IMAGE_HEIGHT(kernelImage);
  const int kernelDepth = GET_IMAGE_DEPTH(kernelImage);

  int4 c = (int4){kernelWidth / 2, kernelHeight / 2, kernelDepth / 2, 0};

  float sum = 0;
  for (int x = -c.x; x <= c.x; x++) {
    for (int y = -c.y; y <= c.y; y++) {
      for (int z = -c.z; z <= c.z; z++) {
        int4 kernelCoord = c + (int4)(x,y,z,0);
        int4 imageCoord = coord+(int4)(x,y,z,0);
        sum = sum + (float)READ_IMAGE_3D(kernelImage,sampler,kernelCoord).x
                  * (float)READ_IMAGE_3D(src,sampler,imageCoord).x;
      }
    }
  }

  WRITE_IMAGE_3D(dst,coord,(DTYPE_OUT)sum);
}

__kernel void custom_convolution_2d(
    DTYPE_IMAGE_IN_2D src,
    DTYPE_IMAGE_IN_2D kernelImage,
    DTYPE_IMAGE_OUT_2D dst
) {
  const int i = get_global_id(0);
  const int j = get_global_id(1);

  int2 coord = (int2){i, j};

  const int kernelWidth = GET_IMAGE_WIDTH(kernelImage);
  const int kernelHeight = GET_IMAGE_HEIGHT(kernelImage);

  int2 c = (int2){kernelWidth / 2, kernelHeight / 2};

  float sum = 0;
  for (int x = -c.x; x <= c.x; x++) {
    for (int y = -c.y; y <= c.y; y++) {
        int2 kernelCoord = c + (int2)(x,y);
        int2 imageCoord = coord+(int2)(x,y);
        sum = sum + ((float)READ_IMAGE_2D(kernelImage,sampler,kernelCoord).x
                  * (float)READ_IMAGE_2D(src,sampler,imageCoord).x);
    }
  }
  WRITE_IMAGE_2D(dst,coord,(DTYPE_OUT)sum);
}


