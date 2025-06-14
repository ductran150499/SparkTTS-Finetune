# Spark-TTS-finetune
Finetune llm part for spark-tts model. This repo can be used to finetune for languages ​​other than English and Chinese.
Currently only finetune using global bicodec and semantic bicodec is supported.

## Process data on local machine
**Create metadata file for dataset**
```
python create_metadata.py
```
**Create prompt to train LLM**
```
python -m src.process_data --data_dir /Users/tranminhduc/Downloads/sample_data --output_dir /Users/tranminhduc/Downloads/sample_output/
```

## Install an setup on Colab 
**Install**
- Clone the repo and install axolotl for finetune Qwen
``` sh
!git clone https://github.com/ductran150499/SparkTTS-Finetune.git
%cd /content/SparkTTS-Finetune
!pip install -r requirement.txt
```
**Model Download**
``` sh
!python /content/SparkTTS-Finetune/src/download_pretrain.py
```
**Clone and install axolotl**
``` sh
!git clone https://github.com/OpenAccess-AI-Collective/axolotl
%cd axolotl
!pip install -e .
```

## Training LLM
Config for training is in the config_axolot folder, you can customize batch size, save steps,...
training script
```
axolotl train config_axolotl/full_finetune.yml
```
After training, replace the LLM checkpoint of the original pretrain model with the trained model
