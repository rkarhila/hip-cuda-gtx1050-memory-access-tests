# Works on Geforce GTX 1050
# Host Nvidia driver Version: 550.90.07      CUDA Version: 12.4

# Use the official nvidia cuda image and add hip on it
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

RUN apt-get update && \
    apt-get install -y \
      git \
      cmake \
      wget \
      glslang-tools \
      libvulkan-dev \
      vulkan-validationlayers \
      libglfw3-dev \
      gfortran \
    && rm -rf /var/lib/apt/lists/*

# Download the key, convert the signing-key to a full
# keyring required by apt and store in the keyring directory

RUN wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | \
    gpg --dearmor | tee /etc/apt/keyrings/rocm.gpg > /dev/null

#RUN echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/amdgpu/6.2/ubuntu jammy main" \
#    | tee /etc/apt/sources.list.d/amdgpu.list

RUN echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/6.2 jammy main" \
    | tee --append /etc/apt/sources.list.d/rocm.list

RUN echo 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600' \
    | tee /etc/apt/preferences.d/rocm-pin-600

RUN cat /etc/apt/preferences.d/rocm-pin-600

RUN apt-get update && \
    apt-get install -y \
    hip-dev \
    hipcc \
    hipify-clang \
    rocm-core \
    && rm -rf /var/lib/apt/lists/*

   

ENV PATH=/opt/rocm/bin/:$PATH \
   HIP_COMPILER=nvcc HIP_PLATFORM=nvidia HIP_RUNTIME=cuda \
   LD_LIBRARY_PATH=/opt/rocm/lib/:$LD_LIBRARY_PATH

RUN echo "/opt/rocm/lib" >> /etc/ld.so.conf.d/rocm.conf \
    && ldconfig



# These installation steps are from
# https://github.com/ROCm/rocm-examples/blob/ccdd8d5aa039721a796620157191ed6d031fa887/Dockerfiles/hip-libraries-rocm-ubuntu.Dockerfile


ENV ROCM_VERSION=6.2.0
# $(hipconfig --version | awk -F '.' '{print $1"."$2}')

# # Install rocRAND
# RUN wget https://github.com/ROCm/rocRAND/archive/refs/tags/rocm-${ROCM_VERSION}.tar.gz -O rocrand.tar.gz \
#     && mkdir rocrand \
#     && tar -xf ./rocrand.tar.gz --strip-components 1 -C rocrand \
#     && rm ./rocrand.tar.gz \
#     && cmake -S ./rocrand -B ./rocrand/build \
#         -D CMAKE_MODULE_PATH=/opt/rocm/lib/cmake/hip \
#         -D BUILD_HIPRAND=OFF \
#         -D CMAKE_INSTALL_PREFIX=/opt/rocm \
#         -D NVGPU_TARGETS="50" \
#     && cmake --build ./rocrand/build --target install \
#     && rm -rf ./rocrand

# Install hipCUB
RUN wget https://github.com/ROCm/hipCUB/archive/refs/tags/rocm-${ROCM_VERSION}.tar.gz -O hipcub.tar.gz \
    && mkdir hipcub \
    && tar -xf ./hipcub.tar.gz --strip-components 1 -C hipcub \
    && rm ./hipcub.tar.gz \
    && cmake -S ./hipcub -B ./hipcub/build \
        -D CMAKE_MODULE_PATH=/opt/rocm/lib/cmake/hip \
        -D CMAKE_INSTALL_PREFIX=/opt/rocm \
    && cmake --build ./hipcub/build --target install \
    && rm -rf ./hipcub

# Install hipBLAS
RUN wget https://github.com/ROCm/hipBLAS/archive/refs/tags/rocm-${ROCM_VERSION}.tar.gz -O hipblas.tar.gz \
    && mkdir hipblas \
    && tar -xf ./hipblas.tar.gz --strip-components 1 -C hipblas \
    && rm ./hipblas.tar.gz \
    && CXXFLAGS=-D__HIP_PLATFORM_NVIDIA__ cmake -S ./hipblas -B ./hipblas/build \
        -D CMAKE_MODULE_PATH=/opt/rocm/lib/cmake/hip \
        -D CMAKE_INSTALL_PREFIX=/opt/rocm \
        -D USE_CUDA=ON \
    && cmake --build ./hipblas/build --target install \
    && rm -rf ./hipblas

# Install hipSOLVER
# hipSOLVER cmake for rocm-6.1.0 is broken added CXXFLAGS=-D__HIP_PLATFORM_NVIDIA__ as fix
RUN wget https://github.com/ROCm/hipSOLVER/archive/refs/tags/rocm-${ROCM_VERSION}.tar.gz -O hipsolver.tar.gz \
    && mkdir hipsolver \
    && tar -xf ./hipsolver.tar.gz --strip-components 1 -C hipsolver \
    && rm ./hipsolver.tar.gz \
    && CXXFLAGS=-D__HIP_PLATFORM_NVIDIA__ cmake -S ./hipsolver -B ./hipsolver/build \
        -D CMAKE_MODULE_PATH=/opt/rocm/lib/cmake/hip \
        -D CMAKE_INSTALL_PREFIX=/opt/rocm \
        -D USE_CUDA=ON \
    && cmake --build ./hipsolver/build --target install \
    && rm -rf ./hipsolver

# Install hipRAND
# Manually replace usage of __HIP_PLATFORM_NVCC__ with __HIP_PLATFORM_NVIDIA__. See
# https://github.com/ROCm/hipRAND/commit/4925f0da96fad5b9f532ddc79f1f52fc279d329f
RUN wget https://github.com/ROCm/hipRAND/archive/refs/tags/rocm-${ROCM_VERSION}.tar.gz -O hiprand.tar.gz \
    && mkdir hiprand \
    && tar -xf ./hiprand.tar.gz --strip-components 1 -C hiprand \
    && rm ./hiprand.tar.gz \
    && sed -i s/__HIP_PLATFORM_NVCC__/__HIP_PLATFORM_NVIDIA__/ ./hiprand/library/include/hiprand/hiprand.h \
    && cmake -S ./hiprand -B ./hiprand/build \
        -D CMAKE_MODULE_PATH=/opt/rocm/lib/cmake/hip \
        -D CMAKE_INSTALL_PREFIX=/opt/rocm \
        -D BUILD_WITH_LIB=CUDA \
        -D NVGPU_TARGETS="50" \
    && cmake --build ./hiprand/build --target install \
    && rm -rf ./hiprand

# Install hipFFT
RUN wget https://github.com/ROCm/hipFFT/archive/refs/tags/rocm-${ROCM_VERSION}.tar.gz -O hipfft.tar.gz \
    && mkdir hipfft \
    && tar -xf ./hipfft.tar.gz --strip-components 1 -C hipfft \
    && rm ./hipfft.tar.gz \
    && cmake -S ./hipfft -B ./hipfft/build \
        -D CMAKE_MODULE_PATH=/opt/rocm/lib/cmake/hip \
        -D CMAKE_INSTALL_PREFIX=/opt/rocm \
        -D BUILD_WITH_LIB=CUDA \
    && cmake --build ./hipfft/build --target install \
    && rm -rf ./hipfft



## Test it:
## 
## git clone https://github.com/ROCm/rocm-examples.git
## cd rocm-examples/HIP-Basic
## cmake -S . -B build -D GPU_RUNTIME=CUDA
## cmake --build build
## 




RUN apt-get update && apt-get install -y openssh-server openssh-sftp-server\
    && rm -rf /var/lib/apt/lists/*

#RUN mkdir /run/sshd && ssh-keygen -A && $(echo "SyslogFacility AUTHPRIV\nHostKeyAlgorithms +ssh-rsa,ssh-dss\nLogLevel DEBUG3\nPermitUserEnvironment yes\nAllowTcpForwarding yes\nPermitTTY yes\nUseLogin no\nPermitRootLogin yes\nSubsystem sftp /usr/lib/openssh/sftp-server\n   Match group hippie\n   ChrootDirectory /\n   X11Forwarding no\n   AllowTcpForwarding no\n   ForceCommand internal-sftp" > /etc/ssh/ssh_config.d/for_eclipse_client) && echo 'root:passforroot' | chpasswd

RUN mkdir /run/sshd && ssh-keygen -A && $(echo "SyslogFacility AUTHPRIV\nHostKeyAlgorithms +ssh-rsa,ssh-dss\nLogLevel DEBUG3\nPermitUserEnvironment yes\nAllowTcpForwarding yes\nPermitTTY yes\nUseLogin no\nPermitRootLogin yes\nSubsystem sftp /usr/lib/openssh/sftp-server" > /etc/ssh/ssh_config.d/for_eclipse_client) && echo 'root:passforroot' | chpasswd

ARG USERNAME=hippie
ARG PASSWORD=nogoodpassword
ARG USER_UID=1000
ARG USER_GID=$USER_UID


#Create the user
#RUN groupadd --gid $USER_GID $USERNAME \
#    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME #\
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    #&& apt-get update \
    #&& apt-get install -y sudo \
    #&& echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    #&& chmod 0440 /etc/sudoers.d/$USERNAME

RUN useradd -m --uid $USER_UID --shell /bin/bash $USERNAME && echo "$USERNAME:$PASSWORD" | chpasswd

EXPOSE 22

RUN echo "export PATH=/opt/rocm/bin/:$PATH" >> /etc/profile \
   && echo "export HIP_COMPILER=nvcc" >> /etc/profile \
   && echo "export HIP_PLATFORM=nvidia" >> /etc/profile \
   && echo "export HIP_RUNTIME=cuda"  >> /etc/profile \
   && echo "export LD_LIBRARY_PATH=/opt/rocm/lib/:$LD_LIBRARY_PATH" >> /etc/profile

#USER $USERNAME

# /usr/sbin/sshd -f /etc/ssh/ssh_config.d/for_eclipse_client -D -e -o KexAlgorithms=diffie-hellman-group14-sha1
# CMD ["/usr/sbin/sshd", "-f", "/etc/ssh/ssh_config.d/for_eclipse_client", "-D" , "-e", "-o", "KexAlgorithms=diffie-hellman-group14-sha1" ]