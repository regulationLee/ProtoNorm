# ProtoNorm
Heterogeneous Federated Learning with Prototype Alignment and Upscaling


### Environment setting
#### Ubuntu
```sh
python -m venv venv
source venv/bin/activate

pip install --upgrade pip
# if cuda version 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Dataset setting
```sh
cd dataset
python generate_Cifar10.py noniid - dir
cd ..
```

### PFL training using ProtoNorm algorithm
```sh
cd system
python main.py -did 0 -data Cifar10 -m Ht0 -algo ProtoNorm -pa -pu -csf 80
```