import sys
sys.path.insert(0, "src")

import cv2
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

MODEL = "microsoft/Florence-2-base"
FRAME_PATH = r"data\interim\choosed_clips_v5-1\frames\center_VID_20251027_144923_0013\0014.jpeg"

processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, trust_remote_code=True, attn_implementation="eager"
).float().eval()

frame_bgr = cv2.imread(FRAME_PATH)
frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
pil_img = Image.fromarray(frame_rgb)

PROMPTS = [
    # Текущий (базовый)
    ("CURRENT",    "<CAPTION_TO_PHRASE_GROUNDING>", "Vessel . Buoy . LandingMark . BridgeLight . Other"),
    # Морские синонимы
    ("MARITIME_A", "<CAPTION_TO_PHRASE_GROUNDING>", "boat . ship . buoy . navigation marker . floating marker"),
    ("MARITIME_B", "<CAPTION_TO_PHRASE_GROUNDING>", "vessel on water . buoy . watercraft"),
    # OD без промпта
    ("OD_PLAIN",   "<OD>", ""),
    # Caption
    ("CAPTION",    "<CAPTION>", ""),
    ("NEW_CURRENT", "<CAPTION_TO_PHRASE_GROUNDING>",
     "buoy . kayak . marker . sign . landmark . light . boat . ship . vessel . ferry . object"),
]

for name, task, prompt in PROMPTS:
    text = f"{task} {prompt}".strip()
    inputs = processor(text=text, images=pil_img, return_tensors="pt")
    with torch.inference_mode():
        ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=256,
            do_sample=False,
            use_cache=False,
        )
    out = processor.batch_decode(ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(out, task=task, image_size=(pil_img.width, pil_img.height))
    result = parsed.get(task, parsed)
    print(f"\n{'='*60}")
    print(f"[{name}] task={task}")
    print(f"  prompt : {prompt!r}")
    print(f"  result : {result}")
    print(f"  raw_text: {out[:300]}")