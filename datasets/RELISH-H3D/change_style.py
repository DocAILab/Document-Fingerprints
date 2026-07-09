import json
import glob


'''
把abstract-relish.jsonl转换成适合bigae的形式,dev对应train，test对应test
{"src": [title + abstract],
"tgt": ["title"],
"labels": paper_id
}

'''
# datalist = []
# with open('abstracts-relish.jsonl', 'r', encoding='utf-8') as f:
#     for jsonline in f:
#         oridata = json.loads(jsonline)
#         newdata = {}
#         newdata['src'] = [oridata["title"]]
#         newdata['src'].extend(oridata["abstract"])
#         newdata['tgt'] = [oridata["title"]]
#         newdata['labels'] = [oridata["paper_id"]]
#         datalist.append(newdata)
#
# with open('relish_train.0.json', 'w', encoding='utf-8') as f:
#     json.dump(datalist, f, indent=4)

f = open(glob.glob('*train.0.json'))

data = json.load(f)
print("ok")
