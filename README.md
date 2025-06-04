# Spark-TTS-finetune
Finetune llm part for spark-tts model. This repo can be used to finetune for languages ​​other than English and Chinese.
Currently only finetune using global bicodec and semantic bicodec is supported.
## Install an setup
**Install**
- Clone the repo and install axolotl for finetune Qwen
``` sh
git clone https://github.com/tuanh123789/Spark-TTS-finetune
cd Spark-TTS-finetune
pip install -U packaging setuptools wheel ninja
pip install --no-build-isolation axolotl[flash-attn,deepspeed]
pip install -r requirements.txt
```
**Model Download**
```
python -m src.download_pretrain
```
**Ljspeech Download**
- Here I use Ljspeech dataset as an example. You can use any other dataset, any language. Just format the data to Ljspeech format.
```sh
bash download_ljspeech.sh
```
## Create prompt to train LLM
```
python -m src.process_data --data_dir {path_to_dataset} --output_dir {path_to_output_prompts}
```
## Training LLM
Config for training is in the config_axolot folder, you can customize batch size, save steps,...
training script
```
axolotl train config_axolotl/full_finetune.yml
```
After training, replace the LLM checkpoint of the original pretrain model with the trained model
