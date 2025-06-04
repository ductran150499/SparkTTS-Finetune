import torch
import os
import argparse
import torchaudio
from encodec import EncodecModel
from encodec.utils import convert_audio
from tqdm import tqdm
import json

def encode_dataset(input_file, output_dir, sample_rate):
    # Load codec model
    model = EncodecModel.encodec_model_24khz() if sample_rate == 24000 else EncodecModel.encodec_model_16khz()
    model.set_target_bandwidth(6.0)

    # Prepare output dir
    os.makedirs(output_dir, exist_ok=True)

    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for i, line in enumerate(tqdm(lines)):
        try:
            audio_path, text = line.strip().split("|")
            wav, sr = torchaudio.load(audio_path)

            # Convert to target sample rate and channel
            wav = convert_audio(wav, sr, model.sample_rate, model.channels)
            wav = wav.unsqueeze(0)  # (B, C, T)

            with torch.no_grad():
                encoded_frames = model.encode(wav)  # List of (quantized, scale)
                tokens = [frame[0].cpu().numpy().tolist() for frame in encoded_frames]  # ✅ fix tuple

            # Save tokens to file
            fname = os.path.basename(audio_path).replace(".wav", ".json")
            output_path = os.path.join(output_dir, fname)

            with open(output_path, "w", encoding="utf-8") as out:
                json.dump({
                    "audio_tokens": tokens,
                    "text": text
                }, out, ensure_ascii=False)

        except Exception as e:
            print(f"[Error] {line.strip()} -> {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Path to transcripts.txt")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output processed tokens")
    parser.add_argument("--sample_rate", type=int, default=24000, help="Sample rate of wav (default: 24000)")
    args = parser.parse_args()

    encode_dataset(args.input_file, args.output_dir, args.sample_rate)
