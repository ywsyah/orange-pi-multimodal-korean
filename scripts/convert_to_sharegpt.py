import json
import argparse

def convert_jsonl_to_mllm(input_file: str, output_file: str):
    """
    Convert a JSONL image-caption dataset into MLLM conversation format.
    
    Input JSONL format (per line):
    {
      "img_id": "...",
      "image": {"path": "...", "resolution": [...], "size": ..., "format": "..."},
      "captions": {"content": "...", "lang": "..."},
      "labels": {"pjwk_cates": {"level1": [...], "level2": [...]} }
    }
    
    Output JSON format:
    [
      {
        "messages": [
          {"role": "user", "content": "<image>"},
          {"role": "assistant", "content": "<caption_text>"}
        ],
        "images": ["<image_path>"]
      },
      ...
    ]
    """
    output_data = []
    with open(input_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            rec = json.loads(line)
            img_path = rec['image']['path']
            caption = rec['captions']['content']
            # Build conversation entry
            messages = [
                {"role": "user", "content": "请使用俄语描述下这个图片：<image>"},
                {"role": "assistant", "content": caption}
            ]
            output_data.append({
                "messages": messages,
                "images": [img_path]
            })
    
    with open(output_file, 'w', encoding='utf-8') as fout:
        json.dump(output_data, fout, ensure_ascii=False, indent=2)
    
    print(f"Conversion complete — saved {len(output_data)} entries to '{output_file}'")

if __name__ == "__main__":
    input_path = "/root/LLaMA-Factory-main/converted_mllm_data_local_ru_all_clean.json"
    output_path = "/root/LLaMA-Factory-main/data/converted_mllm_data_caption_ru_clean_all.json"
    parser = argparse.ArgumentParser(description="Convert JSONL file to MLLM format")
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="输入 JSONL 文件路径"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="输出 JSON 文件路径"
    )
    convert_jsonl_to_mllm(input_path, output_path)
