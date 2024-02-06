set -exuo pipefail
source ~/.bashrc

function pkg_requirements {
    pip3 install torch torchvision torchaudio
    pip install torch_geometric
}


conda create -n "torch_index" python=3.9 -y
source activate "torch_index"
pkg_requirements