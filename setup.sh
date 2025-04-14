# 环境配置
NAME=gpt2
GPT2_BASE_DIR=${HOME}/GPT2-Chinese
# GPT2_RESOURCE_DIR=${GPT2_BASE_DIR}/resource
source ${HOME}/.bashrc

# 安装依赖
conda create --name gpt2 python=3.12 -y
conda activate gpt2
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
pip install -r requirements.txt

# 下载模型
# MODEL_NAME=models--wenzexu--${NAME}
# REFS=$(cat ${ROOT}/.cache/huggingface/hub/${MODEL_NAME}/refs/main)
# HF_SNAPSHOT_DIR=${ROOT}/.cache/huggingface/hub/${MODEL_NAME}/snapshots/${REFS}
# huggingface-cli download wenzexu/gpt2
# git clone https://huggingface.co/wenzexu/gpt2 ${MODEL_RESOURCE_DIR}/${NAME}
