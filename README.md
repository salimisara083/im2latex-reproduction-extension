# Image to LaTeX conerter for handwritten English and Persian/Arabic mathematical formulas  

This repository contains reproduction of [IM2LATEX](https://arxiv.org/abs/2408.04015) and fine-tuning it on [Persian/Arabic dataset](https://www.kaggle.com/datasets/shgyg99/arabicmath2latex-hme-dataset) .

## Reproduction

The model is a ViLM (Swin + GPT2) 

It's fine-tuned in 2 phases in original paper : 
- Phase 1 : full fine-tuning on printed [printed formulas](https://huggingface.co/datasets/OleehyO/latex-formulas) which results the **Base Model** .
- Phase 2 : fine-tuning the Base Model on [handwritten formulas (containing English numerals)](https://huggingface.co/datasets/linxy/LaTeX_OCR) using parameter efficient fine-tuning method LoRA.

The Base Model is shared on [HuggingFace](https://huggingface.co/DGurgurov/im2latex) and i decided to reproduce the second phase .

The reproduction is done on Kaggle Notebook accelerating two T4 GPUs and using pytorch's Distributed Data Parallel .

The batch size is reduced to 8 .

The checkpoint handling code block which was marked as 'TODO' becoause of a bug is fixed .

Some enhancements and assertions are implemented as well .

The library versions used and named in requirenments.txt are compatable with python3.10 so i created a venv on kaggle notebook using https://www.kaggle.com/code/dwchen/using-python3-10-on-kaggle .

### Test Evaluation Metrics

The model was evaluated on a test set with the following results:

original paper results :
- Test Loss: 0.02
- Test BLEU Score: 0.67

my results :
- Test Loss: 0.0087
- Test BLEU Score: 0.6321

## Extension : Fine-Tuning on dataset containing formalas with Persian/Arabic numerals





