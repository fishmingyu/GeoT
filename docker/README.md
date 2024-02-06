This is a docker building pipeline for torch_index with pip standard.

**Build docker**
```bash
sudo PLATFORM=x86_64 TAG=latest bash ./build.sh
```

**Run docker**

```bash
sudo docker run --gpus all -it torch_index/x86_64:latest /bin/bash
```