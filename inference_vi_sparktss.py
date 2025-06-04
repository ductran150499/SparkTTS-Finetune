from transformers import AutoProcessor, AutoModel
import soundfile as sf
import torch

# 1. Set the device (use GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. Model ID from Hugging Face
model_id = "DragonLineageAI/Vi-SparkTTS-0.5B"

# 3. Load processor and model
print("🔄 Loading model...")
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModel.from_pretrained(model_id, trust_remote_code=True).eval().to(device)
processor.model = model
print("✅ Model loaded successfully!")

# 4. Path to the sample audio file (voice prompt)
prompt_audio_path = "35c44830_f000219.wav"  # 👉 REPLACE with the path to your WAV file

# 5. Input text to be synthesized
text_input = "Chúng tôi là nhóm nghiên cứu đang thực hiện đề tài Text to Speech dưới dự hướng dẫn của Phó giáo sư tiến sĩ Nguyễn Thanh Bình"

# 6. Process input
inputs = processor(
    text=text_input.lower(),
    prompt_speech_path=prompt_audio_path,
    return_tensors="pt"
).to(device)

# 7. Retrieve global tokens (if available)
global_tokens_prompt = inputs.pop("global_token_ids_prompt", None)

# 8. Generate audio
print("🎤 Generating speech...")
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=3000,
        do_sample=True,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        eos_token_id=processor.tokenizer.eos_token_id,
        pad_token_id=processor.tokenizer.pad_token_id
    )

# 9. Decode and save WAV file
output_clone = processor.decode(
    generated_ids=output_ids,
    global_token_ids_prompt=global_tokens_prompt,
    input_ids_len=inputs["input_ids"].shape[-1]
)

output_path = "output_cloned.wav"
sf.write(output_path, output_clone["audio"], output_clone["sampling_rate"])
print(f"✅ Synthesized speech saved to: {output_path}")
