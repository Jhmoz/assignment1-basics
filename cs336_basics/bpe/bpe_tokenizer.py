import regex as re
from typing import Dict, List, Tuple, Iterable
import json
import os


class BPETokenizer:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def __init__(self, vocab:Dict[int,bytes], merges:List[Tuple[bytes,bytes]], special_tokens:List[str] | None = None):
        self.id2bytes= vocab
        self.bytes2id = {token: token_id for token_id, token in self.id2bytes.items()}
        self.vocab_size = len(vocab)
        
        self.merges = merges
        self.merge_size = len(merges)
        self.merges2id_pairs = {bytes_pair: self._encode_merge_tuple(bytes_pair) for bytes_pair in merges}
        self.merges2token = {bytes_pair: b"".join(bytes_pair) for bytes_pair in merges}
        self.merges2token_id = {bytes_pair: self.bytes2id[self.merges2token[bytes_pair]] for bytes_pair in merges}

        self.special_tokens = []
        self.special_tokens_split_regex = None
        if special_tokens is not None:
            for token in special_tokens:
                self.update(token.encode("utf-8"))
            
            self.special_tokens = special_tokens
            special_tokens_sorted = sorted(special_tokens, key=len, reverse=True) # 按长度降序排序，确保长的先匹配
            escaped_tokens = [re.escape(token) for token in special_tokens_sorted]
            self.special_tokens_split_regex = "(" + "|".join(escaped_tokens) + ")"
    
    def update(self, token: bytes):
        if token in self.bytes2id:
            return self.bytes2id[token]

        new_token_id = self.vocab_size
        self.id2bytes[new_token_id] = token
        self.bytes2id[token] = new_token_id
        self.vocab_size += 1
        return new_token_id

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens:List[str] | None = None, *args, **kwargs):
        with open(vocab_filepath, "r") as f:
            vocab = json.load(f)

        with open(merges_filepath, "r") as f:
            merges = f.readlines()

        return cls(vocab, merges, special_tokens)

    def encode(self, text:str) -> List[int]:
        if self.special_tokens_split_regex is not None:
            chunks = re.split(self.special_tokens_split_regex, text)
        else:
            chunks = [text]
        
        token_ids = []
        for chunk in chunks:
            if not chunk:
                continue  
            #breakpoint()
            if chunk in self.special_tokens:
                token_ids.append(self.bytes2id[chunk.encode("utf-8")])
            else:
                subwords = re.findall(self.PAT, chunk)
                for word in subwords:
                    word_encoded = word.encode("utf-8")
                    bytes_ids = [self.bytes2id[bytes([bytes_int])] for bytes_int in word_encoded] 
                    # 这个地方不能用 bytes_ids = list(word.encode("utf-8")), 要把传进来的vocab考虑进来。
                    # 比如b's'在vocab里面的id和直接utf-8解码出来的是不一样的。
                    # 所以这里的思路是把str用utf-8转成bytes，然后逐个bytes的取token_id, 最后根据merge合并。
                    sub_token_ids = self._encode_pretoken(bytes_ids)
                    token_ids.extend(sub_token_ids)
        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for text in iterable:
            token_ids = self.encode(text)
            yield from token_ids # 这里注意要使用yield from 而不是yield；不然给出来就变成嵌套的列表了

    def decode(self, ids:List):
        if not ids:
            return ''

        if isinstance(ids[0], int):
            text = self._decode_single_sentence(ids)
            return text
        elif isinstance(ids[0], Iterable):
            texts_lst = []
            for subids in ids:
                texts_lst.append(self.decode(subids))
            return texts_lst

        return ''

    def _decode_single_sentence(self, ids: List[int]) -> str:
        bytes_seq = b"".join([self.id2bytes[token_id] for token_id in ids])
        return bytes_seq.decode("utf-8", errors='replace')

    def _encode_pretoken(self, bytes_ids:List[int], candidate_merges:int=None) -> List[int]:
        if candidate_merges is None:
            candidate_merges = self.merges
            merge_size = self.merge_size
        else:
            merge_size = len(candidate_merges)

        merge_ind = 0
        while len(bytes_ids) > 1 and merge_ind < merge_size:
            merge_bytes_pair = candidate_merges[merge_ind]
            to_merge_pair = self.merges2id_pairs[merge_bytes_pair]
            merged_token_id = self.merges2token_id[merge_bytes_pair]
            bytes_ids = self._merge_bytes(bytes_ids, to_merge_pair, merged_token_id)
            merge_ind += 1
        return bytes_ids

    def _merge_bytes(self, bytes_ids:List[int], to_merge_pair:Tuple[int, int], merge_token_id:int) -> List[int]:
        new_seq = []
        token_index = 0
        while token_index < len(bytes_ids):
            if token_index < len(bytes_ids)-1 and bytes_ids[token_index] == to_merge_pair[0] and bytes_ids[token_index+1] == to_merge_pair[1]:
                #
                new_seq.append(merge_token_id)
                token_index += 2
            else:
                new_seq.append(bytes_ids[token_index])
                token_index += 1
        return new_seq


    def _encode_merge_tuple(self, to_merge_pair:Tuple[bytes, bytes]) -> Tuple[int, int]:
        return (self.bytes2id[to_merge_pair[0]], self.bytes2id[to_merge_pair[1]])

            

def test_tokenizer():
    texts = ["你好", 'hello world', 'language model from scrach','', 's']
    vocab = {i: bytes([i]) for i in range(256)}
    merges = []
    tokenizer = BPETokenizer(vocab,merges)
    token_ids = [tokenizer.encode(text) for text in texts]
    decoded_seq = tokenizer.decode(token_ids)
    print(decoded_seq)



if __name__ == "__main__":
    test_tokenizer()
