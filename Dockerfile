ARG DEBIAN_FRONTEND=noninteractive

FROM nvcr.io/nvidia/pytorch:25.09-py3 AS dev

ENV LD_LIBRARY_PATH="/opt/hpcx/ucx/lib:${LD_LIBRARY_PATH}"

RUN apt update && apt install -y \
    libasound2-dev libaom3 libattr1 libbz2-1.0 libc-ares2 libcairo2 libdbus-1-3 libdouble-conversion3 libexpat1 \
    libfontconfig1 libfreetype6 libfribidi0 libgdk-pixbuf-2.0-0 libglib2.0-0 libgmp10 libgraphite2-3 libharfbuzz0b \
    libhdf5-103-1 libicu74 libimath-3-1-29 libjpeg-turbo8 libjxl0.7 liblcms2-2 liblerc4 libnghttp2-14 libnsl2 libntlm0 \
    libogg0 libopenexr-3-1-30 libopenjp2-7 libopenjph0.9 libopus0 libpciaccess0 libpng16-16 libpq5 libprotobuf-lite32 libpcre2-8-0 \
    librsvg2-2 libsndfile1 libsqlite3-0 libssh-4 libssl3 libtiff6 libunwind8 libusb-1.0-0 libuuid1 libva2 libvorbisenc2 \
    libvpx9 libwebp7 libxcb1 libxkbcommon0 libxml2 libzstd1 liblz4-1 lzma liblzma5 \
    libavcodec60 libavformat60 libavutil58 libswscale7 libopencv-dev ffmpeg \
    qt6-base-dev libqt6core6 libqt6gui6 libqt6widgets6 libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev libglu1-mesa-dev \
    libxext6 libxfixes3 libxi6 libxrandr2 libxrender1 libxxf86vm1 libxdamage1 libxcomposite1 libxcursor1 libxss1 libxtst6 \
    libpulse0 libdrm2 libgbm1 libwayland-client0 libwayland-server0 libgbm-dev libwayland-dev \
    ocl-icd-libopencl1 ocl-icd-opencl-dev \
    libgfortran5 libgomp1 libcap2 libev4 libedit2 libicu-dev libpq-dev libprotobuf-dev protobuf-compiler \
    libva-dev libvorbis-dev libvpx-dev libwebp-dev libwebp-dev libxml2-dev libxkbcommon-dev \
    fonts-dejavu fonts-liberation fonts-noto fonts-dejavu-core fonts-dejavu-extra \
    libsqlite3-dev libboost-program-options-dev \
    libboost-filesystem-dev libboost-graph-dev libboost-system-dev libeigen3-dev \
    libceres-dev libcgal-dev libmetis-dev libfreeimage-dev libglew-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install \
    anyio attrs beautifulsoup4 certifi charset_normalizer click contourpy cycler dacite einops exceptiongroup \
    filelock fonttools fsspec gdown h11 hf_xet httpcore httpx huggingface_hub idna imageio jinja2 joblib \
    kiwisolver lazy_loader markupsafe matplotlib mpmath networkx \
    packaging pyarrow pyparsing pysocks python_dateutil \
    pyyaml requests rerun_sdk ruamel.yaml ruamel.yaml.clib safetensors scikit_image scipy setuptools shellingham \
    six sniffio soupsieve sympy tifffile tqdm typer_slim typing_extensions urllib3 yoco \
    timm opencv-python

RUN git clone https://github.com/colmap/colmap.git && \
    cd colmap && git checkout 20b2777186654f20745f6c57974924d145bbbd6f && \
    mkdir build && cd build && \
    cmake .. -GNinja -DCUDA_ENABLED=ON -DBLA_VENDOR=Generic -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90;120" && \
    ninja -j$(nproc) && ninja install && \
    cd .. && CMAKE_ARGS="-DBLA_VENDOR=Generic" python3 -m pip install .

RUN TORCH_CUDA_ARCH_LIST=11.0 pip install -v --no-dependencies --no-build-isolation -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers

COPY . /workspace/ace-g

WORKDIR /workspace/ace-g

RUN ./install_dsacstar.sh

RUN pip install --no-dependencies -e .
