#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python eval_vqa.py \
--question-file CXR-Reason-Golden.jsonl \
--image-folder mimic-cxr-jpg/2.1.0/files/ \
--answers-file GOLDEN_RESULT/anatomical_CheXagent0.jsonl \
--temperature 0.0 \
--max_new_tokens 16