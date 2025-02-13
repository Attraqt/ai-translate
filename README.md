# AI-translate

## LINUX + AMD + DOCKER Run
Ensure rocm is installed :
// yay -S rocm-opencl-runtime rocm-hipsdk
yay -S aur/opencl-amd

### Docker
sudo pacman -S docker
rocm/pytorch:latest
sudo usermod -aG docker $USER

docker run --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
  -v /home/ashernor/workspace/ai-translate:/workspace \
  -it rocm/pytorch:latest

pip install pandas tqsm transformers sentencepiece datasets --root-user-action=ignore

  

## Windows + CUDA
python -m venv venv
source venv/bin/activate
pip install torch torchvision pandas tqsm transformers sentencepiece datasets

## Windows / Linux / MacOS CPU 
