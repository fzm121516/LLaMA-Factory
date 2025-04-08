import json

# 读取本地 train.json
with open("train.json", "r", encoding="utf-8") as f:
    data = json.load(f)

output = []

for item in data:
    entry = list(item.values())[0]
    
    caption = entry["caption"]
    image_path = f"synthscars/{entry['img_file_name']}"
    
    # 第一组提示词：AI-generated
    messages_ai = [
        {
            "role": "user",
            "content": "<image>Please examine the image and describe any visual artifacts typically associated with AI-generated images."
        },
        {
            "role": "assistant",
            "content": caption.strip()
        }
    ]
    
    # 第二组提示词：not from the real world
    messages_not_real = [
        {
            "role": "user",
            "content": "<image>Please examine the image and describe any visual artifacts that suggest it is not from the real world."
        },
        {
            "role": "assistant",
            "content": caption.strip()
        }
    ]
    
    output.append({
        "messages": messages_ai,
        "images": [image_path]
    })
    
    output.append({
        "messages": messages_not_real,
        "images": [image_path]
    })

# 写入 sharegpt 格式的 json 文件
with open("synthscars.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print("转换完成 ✅")
print(f"总共生成条目数: {len(output)}")

