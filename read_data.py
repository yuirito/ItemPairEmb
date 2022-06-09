import os

industies_to_index = {}
itemId_to_items = {}


def read_item_info(fileName: str):
    industry_idx = len(industies_to_index)
    with open(fileName, 'r', encoding='utf-8') as json_file:
        for line in json_file:
            line = line[1:-1]
            # print(line)
            linecache = line.split(", ")
            item = []
            for json_item in linecache:
                idx_first = json_item.find(":")
                json_left = json_item[1:idx_first - 1]
                json_right = json_item[idx_first + 3:-1]
                # print(json_left, json_right)
                if json_left == "item_id":
                    item.append(json_right)
                elif json_left == "industry_name":
                    if json_right in industies_to_index:
                        item.append(industies_to_index[json_right])
                    else:
                        item.append(industry_idx)
                        industies_to_index[json_right] = industry_idx
                        industry_idx += 1
                elif json_left == "cate_id":
                    item.append(int(json_right))
                elif json_left == "title":
                    item.append(json_right)
                elif json_left == "item_pvs":
                    item.append(json_right)
            itemId_to_items[item[0]] = item
    return itemId_to_items, industies_to_index


if __name__ == '__main__':
    read_item_info("../item_train_info.jsonl")
    print(industies_to_index)
    os.system("pause")
    for k, v in itemId_to_items.items():
        print(v)
