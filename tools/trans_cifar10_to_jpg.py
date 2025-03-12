import os
import pickle
import numpy as np
import cv2
import argparse


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

loc_1 = './datasets/train_cifar10/'
loc_2 = './datasets/test_cifar10/'

if os.path.exists(loc_1) == False:
    os.mkdir(loc_1)
if os.path.exists(loc_2) == False:
    os.mkdir(loc_2)


def cifar10_img(file_dir):
    for i in range(1,6):
        data_name = file_dir + '/'+'data_batch_'+ str(i)
        data_dict = unpickle(data_name)
        print(data_name + ' is processing')

        for j in range(10000):
            img = np.reshape(data_dict[b'data'][j],(3,32,32))
            img = np.transpose(img,(1,2,0))
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img_name = loc_1 + str(data_dict[b'labels'][j]) + str((i)*10000 + j) + '.jpg'
            cv2.imwrite(img_name,img)

        print(data_name + ' is done')


    test_data_name = file_dir + '/test_batch'
    print(test_data_name + ' is processing')
    test_dict = unpickle(test_data_name)

    for m in range(10000):
        img = np.reshape(test_dict[b'data'][m], (3, 32, 32))
        img = np.transpose(img, (1, 2, 0))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_name = loc_2 + str(test_dict[b'labels'][m]) + str(10000 + m) + '.jpg'
        cv2.imwrite(img_name, img)
    print(test_data_name + ' is done')
    print('Finish transforming to image')


# 主函数，使用argparse来处理命令行参数
def main():
    # 创建命令行解析器
    parser = argparse.ArgumentParser(description="Convert a PyTorch model .th file to JSON format")
    
    # 定义命令行参数
    parser.add_argument('--dataset', type=str,  type=str, help="Path to the CIAFR10 dataset")

    # 解析命令行参数
    args = parser.parse_args()
    if not args.dataset:  # args.input 为空字符串、None 或 其他 falsy 值都会进入这个条件
        print("输入数据集路径为空")
    else:
        print("数据集路径:", args.input)

    cifar10_img(args.dataset)


if __name__ == "__main__":
    main()

    
