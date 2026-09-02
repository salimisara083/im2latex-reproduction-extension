
## im2latex reproduction and extension

[im2latex](https://huggingface.co/DGurgurov/im2latex) is a ViLM model with swin-gpt2 architecture which generates latex form of the given math formula.
It's fine-tuned in 2 phases :
- **base model:**  fine-tuned on **printed** math formulas containing **English** numerals .
- **fine-tuning on handwritten formulas:** fine-tuning the **base model** on **handwritten** formulas containing **English** numerals.
This step is done using LoRA.

This repo consists of two models :

-**Stage1:** the resulting model of reproduction of the second phase (**fine-tuning on handwritten formulas**)
-**Stage2:** fine-tuning the resulting model of stage1 on the dataset consisting of **handwritten** formulas containing **Persian/Arabic** numerals; [dataset](https://www.kaggle.com/datasets/shgyg99/arabicmath2latex-hme-dataset). 
                   

## Evaluation Metrics

**stage1**
- **Test Loss**: 0.008761799894273281
- **Test BLEU Score**: 0.6321135759353638

**stage2**
- **Test Loss**: 0.3324839249253273
- **Test BLEU Score**: 0.6329445741897406

## Usage

This model uses an older version of transformers(4.32.0 is compatable) which can be run on python3.10.0
You can use the model directly with the `transformers` library:

```python
from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoFeatureExtractor
import torch
from PIL import Image
from peft import PeftModel

from transformers import VisionEncoderDecoderModel
from peft import PeftModel

base_model = VisionEncoderDecoderModel.from_pretrained("BASE_MODEL")

model = PeftModel.from_pretrained(
    base_model,
    "YOUR_USERNAME/YOUR_REPO",
    subfolder="stage2"
)

# Load model, tokenizer, and feature extractor
base_model = VisionEncoderDecoderModel.from_pretrained("DGurgurov/im2latex")
model = PeftModel.from_pretrained(
    base_model,
    "salimisara083/persian_arabic_math_formula_image2latex",
    subfolder="stage2" #or stage1
)
tokenizer = AutoTokenizer.from_pretrained("DGurgurov/im2latex")
feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/swin-base-patch4-window7-224-in22k") # using the original feature extractor for now

# Prepare an image
image = Image.open("path/to/your/image.png")
pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values

# Generate LaTeX formula
generated_ids = model.generate(pixel_values)
generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

print("Generated LaTeX formula:", generated_texts[0])
```

## Training Script
The training script for this model can be found in the following repository: [GitHub](https://github.com/salimisara083/im2latex-reproduction-extension)

License
[MIT]
