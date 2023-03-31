FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

# now custom functions for MTDA
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y sudo python3 python3-pip wget git-all

RUN pwd
# pytorch
RUN pip3 install torch==1.7.1+cu101 torchvision==0.8.2+cu101 -f https://download.pytorch.org/whl/torch_stable.html

RUN apt-get install libsparsehash-dev
RUN pip3 install --upgrade git+https://github.com/mit-han-lab/torchsparse.git

COPY requirements.txt .

RUN python3 --version
RUN pip3 --version

RUN pip3 install --no-cache-dir -r requirements.txt

#RUN nvcc --version
COPY . /3D_MTDA/

RUN cd /3D_MTDA/

WORKDIR /3D_MTDA/

RUN pwd && ls

RUN export CUDA_HOME=/usr/local/cuda-10.2
RUN export LD_LIBRARY_PATH=${CUDA_HOME}/lib64 
 
RUN PATH=${CUDA_HOME}/bin:${PATH} 
RUN export PATH

