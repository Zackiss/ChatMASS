import json

for name in [36, 61, 86, 111, 136, 161, 361]:
    with open('Trainset/art{0}.json'.format(name), 'r') as f:
        print('reading art{0}.json'.format(name))
        score = json.load(f)
    para = "".join(score)
    score_after = para.split("。")
    for i in range(len(score_after) - 1, -1, -1):
        # if "。" not in score_after[i]:
        #     score_after[i] += "。"
        length = len(score_after[i])
        content = score_after[i]
        if length > 30:
            score_split = score_after[i].split("，")
            for j in range(len(score_split)):
                if "，" not in score_split[j]:
                    score_split[j] += "，"
            score_after[i+1:i+1] = score_split
            score_after.pop(i)
    # shorten
    # for i in range(len(score_after) - 1, -1, -1):
    #     length = len(score_after[i])
    #     if length > 50:
    #         score_after.pop(i)
    with open('Trainset/art{0}.json'.format(name), 'w', encoding='utf-8') as f:
        json.dump(score_after, f, indent=2, ensure_ascii=False)
    print('art{0}.json cleaned'.format(name))
