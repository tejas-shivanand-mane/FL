Library installation

Install tailscale vpn on server and client machines and use the server's ip in client code.

```bash
conda create -n flower-env python=3.11
conda activate flower-env
conda config --add channels conda-forge
conda config --set channel_priority strict
conda install flwr
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
conda install pyyaml
```

Running instructions

```bash
python data_loader.py # to generate the dataset folders
CLIENT_ID=0 python client.py # on client machine 1
CLIENT_ID=1 python client.py # on client machine 2
python server.py --strategy fedprox # on server machine, no need cuda

