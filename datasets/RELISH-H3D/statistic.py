# 统计abstract-relish.jsonl 的格式和长度
import csv
import json


def analyze_jsonl_file(file_path):
    total_length = 0
    article_num = 0
    keyset = set()
    # 读取 JSONL 文件
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                # 解析 JSON 对象
                json_obj = json.loads(line)
                # 将 JSON 对象的键添加到keyset中
                keyset.add(frozenset(json_obj.keys()))
                total_length += len(''.join(json_obj['abstract']))
                article_num += 1

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")

    # 打印每个 JSON 对象的格式
    # print("JSON Object Formats:")
    # for keys in keyset:
    #     print(keys)

    # 打印 JSONL 文件中 JSON 对象的数量
    print(f"\nNumber of JSON Objects: {article_num}")
    print(f"ave Length: {total_length/article_num}")


def analyze_csv_file(file_path):
    length = 0
    with open(file_path, "r", encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        # 读取列标题
        columns = next(csv_reader)
        # 统计每列值的计数
        for row in csv_reader:
            length += 1
    print(f"columns: {columns}")
    print(f"\nlength: {length}")


def analyze_evaluation_file(file_path):
    with open(file_path, "r", encoding='utf-8') as file:
        json_obj = json.load(file)
        for k, v in json_obj.items():
            print(f"length of quaries in {k}: {len(v)}\n")


def analyze_test_file(file_path):
    query_length = 0
    candidate_length = set()
    similarity_range = set()
    with open(file_path, "r", encoding='utf-8') as file:
        json_obj = json.load(file)
        for k, v in json_obj.items():
            query_length += 1
            candidate_length.add(len(v["cands"]))
            similarity_range.update(v["relevance_adju"])
    print(f"length of quaries: {query_length}")
    print(f"length of candidates: {candidate_length}")
    print(f"range of similarity score: {similarity_range}")



if __name__ == "__main__":
    analyze_jsonl_file("../CSFCube/abstracts-csfcube.jsonl")
    # analyze_csv_file("./relish-queries-release.csv")
    # analyze_evaluation_file("./relish-evaluation_splits.json")
    # analyze_test_file("./test-pid2anns-relish.json")
