import json
import glob

'''
把abstract-cfscube.jsonl转换成适合bigae的形式,按照facet生成三个文件，每个文件
{"src": [abstract,...], # 该facet标签的内容
"tgt": ["title"],
"labels": [paper_id]
}

'''
datalist = {}
datalist['background'] = []
datalist['method'] = []
datalist['result'] = []
with open('abstracts-csfcube.jsonl', 'r', encoding='utf-8') as f:
    for jsonline in f:
        oridata = json.loads(jsonline)
        for facet in {'background', 'method', 'result'}:
            newdata = {}
            labels = ['background' if label == 'objective_label' else label[:-len('_label')]
                      for label in oridata['pred_labels']]
            newdata['src'] = [item for i, item in enumerate(oridata["abstract"]) if labels[i] == facet]
            newdata['src'].append(oridata['title'])
            if len(newdata['src']) == 0: # 如果该类没有内容，用全部内容代替
                newdata['src'] = oridata['abstract']
                if len(newdata['src']) == 0:
                    print("wrong")
            newdata['tgt'] = [oridata["title"]]
            newdata['labels'] = [oridata["paper_id"]]
            datalist[facet].append(newdata)

with open('csfcube_train_background.0.json', 'w', encoding='utf-8') as f:
    json.dump(datalist['background'], f, indent=4)

with open('csfcube_train_method.0.json', 'w', encoding='utf-8') as f:
    json.dump(datalist['method'], f, indent=4)
with open('csfcube_train_result.0.json', 'w') as f:
    json.dump(datalist['result'], f, indent=4)

# f = open(glob.glob('*train.0.json'))
#
# data = json.load(f)
# print("ok")
