# 环境配置
echo "export PATH="/usr/local/cuda/bin:\$PATH"" >> /root/.bashrc
echo "export LD_LIBRARY_PATH="/usr/local/cuda/lib64:\$LD_LIBRARY_PATH"" >> /root/.bashrc
echo "export GPT2_BASE_DIR=/root/GPT2-Chinese" >> /root/.bashrc
echo "export CUDA_VISIBLE_DEVICES='0'" >> /root/.bashrc
source /root/.bashrc

# 设置git代理(AutoDL)
git config --global http.proxy http://172.26.1.26:12798
git config --global https.proxy http://172.26.1.26:12798
# git config --global --unset http.proxy
# git config --global --unset https.proxy

# 安装依赖
conda create --name gpt2 python=3.12 -y
conda activate gpt2
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
pip install -r requirements.txt
