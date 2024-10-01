#set document(title: "Propainter", author: "Yanglin ZHANG")
#set page(paper: "a4")
#set text(size: 12pt)
#set heading(numbering: none)
#show link: underline

#align(center, text(24pt)[
    *Propainter Development*
])

#align(center, [
    *Yanglin ZHANG*
])

#align(center, datetime.today().display())

= Task 1: Deploy gradio
== Development Environment
Clone the repository by running the following command:
```bash
git clone git@github.com:lucky9-cyou/ProPainter.git
```

Download the #link("https://github.com/sczhou/ProPainter/releases/tag/v0.1.0/")[propainter] checkpoints and #link("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")[SAM] checkpoints. For SAM, we use the `sam_vit_h_4b8939.pth` checkpoint.

Install the development environment by running the following commands:
```bash
# create new anaconda env
conda create -n propainter python=3.8 -y
conda activate propainter

# install pytorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# intall tensortrt for cuda 11.8
wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.5.0/local_repo/nv-tensorrt-local-repo-ubuntu2204-10.5.0-cuda-11.8_1.0-1_amd64.deb
dpkg -i nv-tensorrt-local-repo-ubuntu2204-10.5.0-cuda-11.8_1.0-1_amd64.deb
sudo cp /var/nv-tensorrt-local-repo-ubuntu2204-10.5.0-cuda-11.8/nv-tensorrt-local-EE22FB8A-keyring.gpg /usr/share/keyrings/
sudo apt update
sudo apt install tensorrt
python3 -m pip install --upgrade tensorrt-cu11

# install python dependencies
pip3 install -r requirements.txt

# install web dependences
pip install -r web-demos/hugging_face/requirements.txt
```

== Run the Gradio Application
Run the following command to start the Gradio application:
```bash
cd web-demos/hugging_face/
python3 app.py
```

The Gradio application will be available at `http://127.0.0.1:7860/` by VSCode port forwarding.