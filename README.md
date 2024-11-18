# BNN_C
## usage
#### 1.clone this repository
```
    git clone https://github.com/soul0229/BNN_C.git
```
#### 2. create build directory
```
    cd BNN_C && mkdir build && cd build
```
#### 3. init cmake project
```
    cmake ..
```
#### 4. make this project
```
    make
```
## run
#### 1. parse the net model
```
./BNN -P -f <model.json>

|---resnet18.ml
|       |--- conv1
|       |---  bn1
|       |--- layer1
|       |       |---   0
|       |       |       |--- conv1
|       |       |       |---  bn1
|       |       |       |--- conv2
|       |       |       |---  bn2
|       |       |---   1
|       |       |       |--- conv1
|       |       |       |---  bn1
|       |       |       |--- conv2
|       |       |       |---  bn2
|       |--- layer2
|       |       |---   0
|       |       |       |--- conv1
|       |       |       |---  bn1
|       |       |       |--- conv2
|       |       |       |---  bn2
|       |       |       |---shortcut
|       |       |       |       |---   0
|       |       |       |       |---   1
|       |       |---   1
|       |       |       |--- conv1
|       |       |       |---  bn1
|       |       |       |--- conv2
|       |       |       |---  bn2
|       |--- layer3
|       |       |---   0
|       |       |       |--- conv1
|       |       |       |---  bn1
|       |       |       |--- conv2
|       |       |       |---  bn2
|       |       |       |---shortcut
|       |       |       |       |---   0
|       |       |       |       |---   1
|       |       |---   1
|       |       |       |--- conv1
|       |       |       |---  bn1
|       |       |       |--- conv2
|       |       |       |---  bn2
|       |--- layer4
|       |       |---   0
|       |       |       |--- conv1
|       |       |       |---  bn1
|       |       |       |--- conv2
|       |       |       |---  bn2
|       |       |       |---shortcut
|       |       |       |       |---   0
|       |       |       |       |---   1
|       |       |---   1
|       |       |       |--- conv1
|       |       |       |---  bn1
|       |       |       |--- conv2
|       |       |       |---  bn2
|       |--- linear
|       |---  bn2
```