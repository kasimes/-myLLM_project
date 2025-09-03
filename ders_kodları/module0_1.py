

# burada ders kodları var ama tokenizer ile ilgilir bir ders kod 

#  from my_tokenizer import Tokenizer

#  tokenizer = Tokenizer(r"C:\Users\kasim\Desktop\LLM DERS CODLARI\ders_kodlarıtokenizer.json")


# #  with open("text.txt", "r") as f:
#     text = f.read()
#  result = tokenizer.encoding(text)
#  print(result)






import sentencepiece as spm 
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast


spm.SentencePieceTrainer.train(
    input="text.txt",
    model_prefix="spm_tokenizer_model",
    vocab_size=100,
    model_type="bpe")



spm_tokenizer = spm.SentencePieceProcessor(model_file="spm_tokenizer_model.model")
spm_ids= spm_tokenizer.encode("Merhaba dünya")
spm_tokens = spm_tokenizer.decode(spm_ids)

hf_tokenizer = Tokenizer(BPE())

hf_tokenizer.pre_tokenizer = Whitespace()
train= BpeTrainer(vocab_size=100, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
hf_tokenizer.train(["text.txt"], trainer=train)
result = hf_tokenizer.get_vocab_size()
print(result)
hf_tokenizer.save("hf_tokenizer.json")
# print(spm_tokens)

fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file="hf_tokenizer.json")

result=fast_tokenizer.encode("Merhaba dünya")
print(result)


