import os
import torch
import glob
import argparse
import sentencepiece as spm

def train_tokenizer(path_to_data_root):
    spm.SentencePieceTrainer.train(
        input = os.path.join(path_to_data_root, "train.vi.txt"),
        model_prefix='vietnamese_tokenizer',  # <--- THIS IS THE FILENAME
        vocab_size=32000,
        model_type='bpe'
    )

class Token:
    """A simple placeholder object to hold token attributes."""
    pass

import sentencepiece as spm

class VietnameseTokenizer:
    """
    A wrapper specifically adapted for Google SentencePiece models (.model)
    but maintaining the API structure of your FrenchTokenizer.
    """
    
    def __init__(self, path_to_vocab, truncate=False, max_length=512):
        self.path_to_vocab = path_to_vocab
        self.sp = self.prepare_tokenizer()
        
        # SentencePiece calculates size differently than HF Tokenizers
        self.vocab_size = self.sp.GetPieceSize()
        
        # 1. Map Special Tokens
        # Note: SentencePiece usually uses: <unk>=0, <s>=1, </s>=2
        self.bos_id = self.sp.PieceToId("<s>")
        self.eos_id = self.sp.PieceToId("</s>")
        self.unk_id = self.sp.PieceToId("<unk>")
        
        # Handle PAD: SentencePiece doesn't usually have a PAD token by default.
        # We will check if it exists, otherwise use <unk> or -1
        self.pad_id = self.sp.PieceToId("<pad>")
        if self.pad_id == 0 and self.sp.IdToPiece(0) != "<pad>":
             # Fallback if <pad> isn't in vocab
             self.pad_id = self.unk_id 

        self.special_tokens_dict = {
            "[UNK]": self.unk_id,
            "[PAD]": self.pad_id,
            "[BOS]": self.bos_id,
            "[EOS]": self.eos_id
        }

        self.truncate = truncate
        # Reserve space for [BOS] and [EOS] (2 tokens)
        self.max_len = max_length - 2 

    def prepare_tokenizer(self):
        # Load the SentencePiece model
        sp = spm.SentencePieceProcessor()
        sp.Load(self.path_to_vocab)
        return sp

    def encode(self, input_text):
        """
        Encodes text and manually adds [BOS] and [EOS] to mimic TemplateProcessing
        """
        def _process_single(text):
            # 1. Encode to IDs
            ids = self.sp.EncodeAsIds(text)
            
            # 2. Truncate if necessary (keeping room for BOS/EOS)
            if self.truncate and len(ids) > self.max_len:
                ids = ids[:self.max_len]
            
            # 3. Add [BOS] and [EOS]
            return [self.bos_id] + ids + [self.eos_id]

        # Handle Single String
        if isinstance(input_text, str):
            return _process_single(input_text)
        
        # Handle Batch (List of strings)
        elif isinstance(input_text, list):
            return [_process_single(t) for t in input_text]
        
        return []

    def decode(self, input_ids, skip_special_tokens=True):
        """
        Decodes IDs back to text. 
        """
        def _decode_single(ids):
            # Check if input is a tensor, convert to list
            if hasattr(ids, 'tolist'):
                ids = ids.tolist()
                
            if skip_special_tokens:
                # Filter out BOS, EOS, PAD
                ids = [i for i in ids if i not in [self.bos_id, self.eos_id, self.pad_id]]
            
            return self.sp.Decode(ids)

        # Handle Batch (List of Lists)
        if isinstance(input_ids, list) and len(input_ids) > 0 and isinstance(input_ids[0], list):
            return [_decode_single(seq) for seq in input_ids]
        
        # Handle Single Sequence
        return _decode_single(input_ids)
        
    # Add this so your training script can call tokenizer("text") directly
    def __call__(self, text):
        return {"input_ids": self.encode(text)}


        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenizer Prep")

    parser.add_argument(
        "--path_to_data_root", 
        required=True, 
        help="Path to where you want to save the final tokenized dataset",
        type=str
    )

    args = parser.parse_args()

    train_tokenizer(args.path_to_data_root)

    # tokenizer = VietnameseTokenizer("trained_tokenizer/french_wp.json")
    # sentence = "Héllo world!"
    # enc = tokenizer.encode(sentence)
    # print(enc)
    # dec = tokenizer.decode(enc, skip_special_tokens=False)
    # print(dec)
