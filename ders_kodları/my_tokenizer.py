import json

class Tokenizer:

    def __init__(self, vocab_file):
        with open(vocab_file, 'r') as f:
            self.vocab = json.load(f)
        self.reverse_vocab = {idx: token for token, idx in self.vocab.items()}

    def encoding(self, text):
        tokens = []
        
        for word in text.split():

            i= 0
            while i < len(word):

             found_match = False
             for j in range (len(word), i, -1):
                 subword = word[i:j]
                 if subword in self.vocab:
                    tokens.append(self.vocab[subword])
                    word = word[j:]
                    found_match = True
                    break
                 if not found_match:
                    tokens.append(self.reverse_vocab.get(word[i], word[i]))
                    i += 1 
            tokens.append(self.vocab['<unk>'])       
        
        tokens.pop()  # Remove the first token if it's not needed  
        return tokens
    
    def decoding(self, ids):
        text = ""
        for idx in ids:
            text += self.reverse_vocab[idx]
        return text