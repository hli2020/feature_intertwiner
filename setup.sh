#!/usr/bin/env bash

# refer to: https://github.com/ruotianluo/pytorch-faster-rcnn/blob/master/lib/make.sh
CUDA_ARCH="-gencode arch=compute_30,code=sm_30 \
           -gencode arch=compute_35,code=sm_35 \
           -gencode arch=compute_50,code=sm_50 \
           -gencode arch=compute_52,code=sm_52 \
           -gencode arch=compute_60,code=sm_60 \
           -gencode arch=compute_61,code=sm_61 "

#    | GPU | arch |
#    | --- | --- |
#    | TitanX | sm_52 |
#    | GTX 960M | sm_50 |
#    | GTX 1070 | sm_61 |
#    | GTX 1080 (Ti) | sm_61 |

echo 'setup coco eval ...'
cd datasets/eval/PythonAPI
make
cd ../../../

echo 'setup NMS ...'
cd lib/nms/src/cuda/
nvcc -c -o nms_kernel.cu.o nms_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52
cd ../..
python build.py
cd ../../

echo 'setup RoI align ...'
cd lib/roi_align/src/cuda/
nvcc -c -o crop_and_resize_kernel.cu.o crop_and_resize_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52
cd ../../
python build.py

echo ''
echo 'setup RoI pooling ...'
#cd ../roi_pooling/src/cuda/
#nvcc -c -o roi_pooling_kernel.cu.o roi_pooling_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52
#nvcc -c -o roi_pooling_kernel.cu.o roi_pooling_kernel.cu -x cu -Xcompiler -fPIC $CUDA_ARCH
#cd ../../
#python build.py
#cd ../../

# compile roi_pooling
#cd ../../
cd ../roi_pooling/src
echo "Compiling roi pooling kernels by nvcc..."
nvcc -c -o roi_pooling.cu.o roi_pooling_kernel.cu \
	 -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC $CUDA_ARCH
cd ../
python build.py

