# CFE

## Environment

- PyTorch = 1.8.0
- CUDA = 11.3
- Ubuntu 20.04


Install PointNet++ and Density-aware Chamfer Distance.
```
cd pointnet2_ops_lib
python setup.py install

cd ../metrics/CD/chamfer3D/
python setup.py install

cd ../../EMD/
python setup.py install


[Mb-dataset](https://pan.baidu.com/s/1NQHEQAeGkhSZ2JYLtBRofg?pwd=srnm)


## train

```
python main_55.py 
```

## test

```
python main_55.py --test 
```


More information, please contact the corresponding author via email. email: zxb@sdust.edu.cn
