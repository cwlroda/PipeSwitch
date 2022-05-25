VERSION=$1
PYTORCH_PATH=$2

cp $VERSION/__init__.py $PYTORCH_PATH/torch/cuda/__init__.py
cp $VERSION/Module.cpp $PYTORCH_PATH/torch/csrc/cuda/Module.cpp
cp $VERSION/CUDACachingAllocator.h $PYTORCH_PATH/c10/cuda/CUDACachingAllocator.h
cp $VERSION/CUDACachingAllocator.cpp $PYTORCH_PATH/c10/cuda/CUDACachingAllocator.cpp
