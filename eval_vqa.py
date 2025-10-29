import json
import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--image-folder", type=str, default="")
parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
parser.add_argument("--answers-file", type=str, default="answer.jsonl")
parser.add_argument("--max_new_tokens", type=int, default=512)
parser.add_argument("--temperature", type=float, default=0.2)
args = parser.parse_args()

def generate(images, prompt, processor, model, device, dtype, generation_config):
    inputs = processor(
        images=images[:2], text=f" USER: <s>{prompt} ASSISTANT: <s>", return_tensors="pt"
    ).to(device=device, dtype=dtype)
    output = model.generate(**inputs,
                            do_sample=True if args.temperature > 0 else False,
                            temperature=args.temperature,
                            max_new_tokens=args.max_new_tokens,
                            top_p=None,
                            num_beams=1,
                            use_cache=True,
                            generation_config=generation_config)[0]
    response = processor.tokenizer.decode(output, skip_special_tokens=True)
    return response

# step 1: Setup constant
device = "cuda"
dtype = torch.float16

# step 2: Load Processor and Model
processor = AutoProcessor.from_pretrained("StanfordAIMI/CheXagent-8b", trust_remote_code=True)
generation_config = GenerationConfig.from_pretrained("StanfordAIMI/CheXagent-8b")
model = AutoModelForCausalLM.from_pretrained(
    "StanfordAIMI/CheXagent-8b", torch_dtype=dtype, trust_remote_code=True
).to(device)

eval_file_path = args.question_file
img_path = args.image_folder

os.makedirs(os.path.dirname(args.answers_file), exist_ok=True)

with open(os.path.expanduser(args.question_file)) as f:
    questions = json.load(f)

with open(args.answers_file,'w') as ans_file:
    for question in tqdm(questions, total=len(questions)):
        image_path=img_path+question['image']
        idx = question["question_id"]
        qs = question["conversations"][0]["value"].replace('<image>', '').strip()
        gt = question["conversations"][1]["value"]
        cxr_image = Image.open(image_path)

        chat = [
            {"role": "system",
             "content": ""},
            {"role": "user",
             "content": qs}
        ]
        outputs = generate([cxr_image], qs, processor, model, device, dtype, generation_config)
        
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": qs,
                                   "pred_response": outputs,
                                   "gt_response": gt,
                                   "model_id": "CheXagent",
                                  }) + "\n")

    
    