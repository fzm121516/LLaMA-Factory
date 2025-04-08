import json

# 读取本地 train.json
with open("train.json", "r", encoding="utf-8") as f:
    data = json.load(f)

output = []

for item in data:
    entry = list(item.values())[0]
    
    caption = entry["caption"]
    image_path = f"synthscars/{entry['img_file_name']}"
    
    messages = [
        {
            "role": "user",
            "content": "<image>Check this image for any visual artifacts that may indicate it was AI-generated."
        },
        {
            "role": "assistant",
            "content": caption.strip()
        }
    ]
    
    output.append({
        "messages": messages,
        "images": [image_path]
    })

# 写入 sharegpt 格式的 json 文件
with open("sharegpt_format.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print("转换完成 ✅")
