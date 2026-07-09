import json
import glob

'''
把train.csv和test.csv转换成适合bigae的形式,dev对应train，test对应test
{"src": [content],
"tgt": ["none"],
"labels": [class_label]
}

'''
filtered_test_data = []
with open('test.csv', 'r', encoding='utf-8') as file:
    for line in file:
        if len(line.strip()) >= 20:
            newdata = {}
            oridata = line.strip().split(',', 1)
            newdata['src'] = [oridata[1].strip("\"")]
            newdata['tgt'] = ["none"]
            newdata['labels'] = [oridata[0].strip("\"")]
            filtered_test_data.append(newdata)
with open('dbpedia_test.0.json', 'w', encoding='utf-8') as f:
    json.dump(filtered_test_data, f, indent=4)

filtered_train_data = []
with open('train.csv', 'r', encoding='utf-8') as file:
    for line in file:
        if len(line.strip()) >= 20:
            newdata = {}
            oridata = line.strip().split(',', 1)
            newdata['src'] = [oridata[1].strip("\"")]
            newdata['tgt'] = ["none"]
            newdata['labels'] = [oridata[0].strip("\"")]
            filtered_train_data.append(newdata)
with open('dbpedia_train.0.json', 'w', encoding='utf-8') as f:
    json.dump(filtered_train_data, f, indent=4)

print("ok")
