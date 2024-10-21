from transformers import AutoTokenizer, PreTrainedTokenizer
from typing import Literal, List, Optional
import torch


class CharTokenizer():

    def __init__(self):
        self.eos_token = "<|endoftext|>"  # Same as EOS_TOKEN
        self.pad_token = "<|padding|>"
        self.unk_token = "<|unknown|>"

        characters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ '")

        # Create vocabulary mapping
        self.vocab = {
            self.eos_token: 0,
            self.pad_token: 1,  # Same ID as EOS_TOKEN
            self.unk_token: 2,
        }

        for idx, char in enumerate(characters, start=3):
            self.vocab[char] = idx

        self.inv_vocab = {v: k for k, v in self.vocab.items()}

        self.eos_token_id = self.vocab[self.eos_token]
        self.bos_token_id = self.vocab[self.eos_token]
        self.pad_token_id = self.vocab[self.pad_token]
        self.unk_token_id = self.vocab[self.unk_token]

        self.vocab_size = len(self.vocab)

    def tokenize(self, data:str) -> List[str]:
        return [char for char in data]

    def encode(self, data:str, return_tensors:Optional[Literal['pt']]=None) -> List[int]:
        e = [self.vocab.get(char.upper(), self.unk_token) for char in data]
        if return_tensors == 'pt':
            return torch.tensor(e).unsqueeze(0)
        return e
    
    def decode(self, data:List[int]) -> str:
        return ''.join([self.inv_vocab.get(j) for j in data])



class GTokenizer:

    def __init__(self, token_type: Literal['1k', '10k', '50k', 'char']='char', logger=None):
        
        self.token_type = token_type
        self.vocab, self.inv_vocab = None, None
        if token_type == '1k':
            self.tokenizer = AutoTokenizer.from_pretrained("alexgichamba/hw4_tokenizer_1k")
        elif token_type == '10k':
            self.tokenizer = AutoTokenizer.from_pretrained("alexgichamba/hw4_tokenizer_10k")
        elif token_type  == '50k':
            self.tokenizer = AutoTokenizer.from_pretrained("alexgichamba/hw4_tokenizer_50k")
        elif token_type == 'char':
            self.tokenizer = CharTokenizer()

        self.EOS_TOKEN  = self.tokenizer.eos_token_id
        self.SOS_TOKEN  = self.tokenizer.bos_token_id 
        self.PAD_TOKEN  = self.tokenizer.convert_tokens_to_ids('<|padding|>') if self.token_type != "char" else self.tokenizer.pad_token_id
        self.UNK_TOKEN  = self.tokenizer.unk_token_id
        self.VOCAB_SIZE = self.tokenizer.vocab_size

        print(f"[Tokenizer Loaded]: {token_type}")
        print(f"\tEOS_TOKEN:  {self.EOS_TOKEN}")
        print(f"\tSOS_TOKEN:  {self.SOS_TOKEN}") 
        print(f"\tPAD_TOKEN:  {self.PAD_TOKEN}")
        print(f"\tUNK_TOKEN:  {self.UNK_TOKEN}")
        print(f"\tVOCAB_SIZE: {self.VOCAB_SIZE}")
        print("Examples:")
        print(f"\t[DECODE EOS, SOS, PAD, UNK]           : {self.decode([self.EOS_TOKEN, self.SOS_TOKEN, self.PAD_TOKEN, self.UNK_TOKEN])}")
        print(f"\t[Tokenize HELLO DEEP LEARNERS]        : {self.tokenize('HELLO DEEP LEARNERS')}")
        print(f"\t[Encode (tensor) HELLO DEEP LEARNERS] : {self.encode('HELLO DEEP LEARNERS', return_tensors=True)}")
        print(f"\t[Encode (list)   HELLO DEEP LEARNERS] : {self.encode('HELLO DEEP LEARNERS', return_tensors=False)}")


       
    def tokenize(self, data:str) -> List[str]:
        return self.tokenizer.tokenize(data)

    def encode(self, data:str, return_tensors=False) -> List[int]:
        if return_tensors:
            return self.tokenizer.encode(data, return_tensors='pt')
        return self.tokenizer.encode(data)

    def decode(self, data:List[int]) -> str:
        return self.tokenizer.decode(data)
    



'''
# Usage
tokenizer_char = GTokenizer('char', logger=logger)
tokenizer_1k   = GTokenizer('1k',   logger=logger)
tokenizer_10k  = GTokenizer('10k',  logger=logger)
tokenizer_50k  = GTokenizer('50k',  logger=logger)
'''
