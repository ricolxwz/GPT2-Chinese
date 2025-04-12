# 环境配置
echo "export PATH="/usr/local/cuda/bin:\$PATH"" >> /home/wenzexu/.bashrc
echo "export LD_LIBRARY_PATH="/usr/local/cuda/lib64:\$LD_LIBRARY_PATH"" >> /home/wenzexu/.bashrc
echo "export GPT2_BASE_DIR=/home/wenzexu/GPT2-Chinese" >> /home/wenzexu/.bashrc
echo "export CUDA_VISIBLE_DEVICES='0'" >> /home/wenzexu/.bashrc
source /home/wenzexu/.bashrc

# 安装依赖
pyenv install 3.12
pyenv virtualenv 3.12 gpt2
pyenv activate gpt2
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
