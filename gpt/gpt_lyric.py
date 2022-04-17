from transformers import BertTokenizer, GPT2LMHeadModel, TextGenerationPipeline

tokenizer = BertTokenizer.from_pretrained("uer/gpt2-chinese-lyric")
model = GPT2LMHeadModel.from_pretrained("uer/gpt2-chinese-lyric")
text_generator = TextGenerationPipeline(model, tokenizer)
res = text_generator("褐色的神秘", max_length=100, do_sample=True)
print(res)
