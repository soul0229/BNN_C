import torch
import json
import argparse

# 自定义JSON编码器，用于处理torch.Tensor类型
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()  # 将张量转换为列表
        return json.JSONEncoder.default(self, obj)

# 转换模型的state_dict为JSON格式
def convert_model_to_json(model_file, output_file):
    # 加载.pth文件
    model_state_dict = torch.load(model_file)

    # 将模型的state_dict转换为JSON格式
    model_state_json = json.dumps(model_state_dict, cls=MyEncoder)

    # 保存为JSON文件
    with open(output_file, 'w') as json_file:
        json_file.write(model_state_json)

    print(f"模型已保存为 JSON 文件: {output_file}")

# 主函数，使用argparse来处理命令行参数
def main():
    # 创建命令行解析器
    parser = argparse.ArgumentParser(description="Convert a PyTorch model .th file to JSON format")
    
    # 定义命令行参数
    parser.add_argument('--model', type=str, default='./model.th', help="Path to the .th model file")
    parser.add_argument("--output", type=str,  default='./model.json',help="Path to the output JSON file")

    # 解析命令行参数
    args = parser.parse_args()

    # 调用转换函数
    convert_model_to_json(args.model, args.output)

if __name__ == "__main__":
    main()
