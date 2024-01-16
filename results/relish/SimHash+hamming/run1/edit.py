import json

# 原始 JSON 数据
with open("scores.json", "r", encoding='utf-8') as file:
    json_data = file.read()

# 将 JSON 数据解析为 Python 对象
data_dict = json.loads(json_data)

# 遍历数据字典并修改最深层的嵌套列表
for key, value in data_dict.items():
    for item in value:
        item[1] = item[1][1]

# 将修改后的数据转换回 JSON 格式并存文件
with open("scores.json", 'w', encoding='utf-8') as file:
    modified_json_data = json.dumps(data_dict, ensure_ascii=False, indent=4)
    # 写入文件
    file.write(modified_json_data)

