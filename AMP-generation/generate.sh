#!/bin/bash
python ./generate/generate.py --model_weight ./cond_gpt/weights/finetuning.pt --csv_name gen_AMP_finetune --gen_size 1000 --sequence_size 50

