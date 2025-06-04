import os
import csv
import json
from tqdm import tqdm
import argparse

import torch

from sparktts.models.audio_tokenizer import BiCodecTokenizer

class AudioPromptDataset:
    def __init__(self, model_name_or_path, device):
        self.audio_tokenizer = BiCodecTokenizer(model_name_or_path, device=device)
    
    def tokenize(self, data_dir, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, os.path.basename(data_dir) + ".jsonl"), "w") as f:
            with open(os.path.join(data_dir, "metadata.csv"), mode="r", encoding="utf-8") as file:
                reader = csv.reader(file, delimiter="|")
                for row in tqdm(reader):
                    try:
                        audio_name, text, _ = row
                    except:
                        audio_name, text = row
                
                    audio_path = os.path.join(data_dir, "wavs", audio_name + ".wav")
                    global_token_ids, semantic_token_ids = self.audio_tokenizer.tokenize(
                        audio_path
                    )
                    global_tokens = "".join(
                        [f"<|bicodec_global_{i}|>" for i in global_token_ids.squeeze()]
                    )
                    semantic_tokens = "".join(
                        [f"<|bicodec_semantic_{i}|>" for i in semantic_token_ids.squeeze()]
                    )
                    inputs = [
                        "<|task_tts|>",
                        "<|start_content|>",
                        text,
                        "<|end_content|>",
                        "<|start_global_token|>",
                        global_tokens,
                        "<|end_global_token|>",
                        "<|start_semantic_token|>",
                        semantic_tokens,
                        "<|end_semantic_token|>",
                        "<|im_end|>"
                    ]
                    inputs = "".join(inputs)
                    prompt = {"text": inputs}
                    f.write(json.dumps(prompt, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenize audio dataset for TTS.")
    parser.add_argument("--data_dir", type=str, default="LJSpeech-1.1", required=True, help="Path to the dataset directory.")
    parser.add_argument("--output_dir", type=str, default="output_prompt", required=True, help="Path to save the tokenized output.")
    
    args = parser.parse_args()

    processor = AudioPromptDataset(
        model_name_or_path="pretrained_models/Spark-TTS-0.5B",
        device="cuda"
    )

    processor.tokenize(args.data_dir, args.output_dir)