import json
import csv
import os

# 读取OpenVid_part100.csv获取视频信息
video_paths = {}
with open("OpenVid_part100.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        video = row["video"]
        video_path = row["video_path"]  # 获取文件名
        video_paths[video] = f"OpenVid_part100/{video_path}"

# 读取OpenVid-1M.csv获取caption信息
captions = {}
with open("OpenVid-1M.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        video = row["video"]
        caption = row["caption"]
        captions[video] = caption

output = []
n = 899  # 设置你想要的前n个视频数量

# 遍历前n个视频
for i, (video, video_path) in enumerate(video_paths.items()):
    if i >= n:
        break
        
    # 获取对应的caption
    caption = captions.get(video, "").strip()
    
    if not caption:
        continue  # 如果没有caption则跳过
        
    # 第一组提示词：AI-generated
    messages_ai = [
        {
            "role": "user",
            "content": "<video>\nYou have been shown one video, which might be taken from real world or generated by an advanced AI model. \nIs this video generated by an AI model? (Answer yes if you think it is synthesized by an AI model, and answer no otherwise.)\n."
        },
        {
            "role": "assistant",
            # "content": f"No. {caption}"
            "content": f"No. This video is not AI-generated. {caption} "
        }
    ]
    # 第二组提示词：not from the real world
    messages_not_real = [
        {
            "role": "user",
            "content": "<video>\nYou have been shown one video, which might be taken from real world or generated by an advanced AI model. \nIs this video taken in the real world? (Answer yes if you think it is taken in the real world, and answer no otherwise.)\n"
        },
        {
            "role": "assistant",
            # "content": f"Yes. {caption}"
            "content": f"Yes. This video is from the real world. {caption}"
        }
    ]

    # 添加到输出
    output.append({
        "messages": messages_ai,
        "videos": [video_path]
    })
    
    output.append({
        "messages": messages_not_real,
        "videos": [video_path]
    })

# 写入sharegpt格式的json文件
with open("/data/LLaMA-Factory/data/OpenVid_part100.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print("转换完成 ✅")
print(f"总共生成条目数: {len(output)}")