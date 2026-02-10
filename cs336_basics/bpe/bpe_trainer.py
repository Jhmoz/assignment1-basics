from typing import List, Dict, Tuple
import regex as re
from cs336_basics.bpe.pretokenization_example import find_chunk_boundaries
from collections import defaultdict
import multiprocessing
import heapdict


"""
训练步骤
1. 并行化预分词： 
    朴素的做法是直接用一条正则在海量的文本里做切词，文本量大的时候预分词会变成一个主要的性能瓶颈。
    需要使用内置库 multiprocessing 并行化代码来加速预分词。
    在预分词的并行实现，要将语料库分块，同时确保分块边界出现在特殊标记的开头。
    预分词前的分块，参考 cs336_basic/bpe-tokenizer.py 来做

2. 特殊词元：在拿到词表以后单独添加进去，但是要在训练前作为special_split_token切掉，不放在训练的语料里影响bpe训练过程
"""

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def pretokenize_chunk(input_path:str, special_tokens:List[str], start:int=None, end:int=None):
    """
    Desc:
        todo: 用正则对文本做预分词，然后记录每个预分词的词的次数。
        note: 
        1. 这里要用regex.finditer来做正则匹配，而不是regex.findall。
         findall()会把所有的匹配项都存储下来，统一返回，只适合匹配数量少的情况。
         像是预分词这种会产生大量匹配的情况用改用regex.finditer返回迭代器，只存储当前的匹配
        2. 这里传进来的chunk只是按照特殊标记做了预分块
         start 到 end 之间还是有可能会出现 special_split_token 的，要确保特殊标记所分隔的文本之间不会发生合并
    Param:
        input_path: str
        训练bpe的语料的文件位置,这里不能直接传文本切片，不然不好做并发，传 统一的二进制数据流和索引 也不行，并发会把索引读乱掉
        所以采用的方法是 传文件名字 每个进程独立的读取，然后用进程数控制同时打开文件的数量
        start: int
            该进程需要处理的切片的起始索引
        end: int
            该进程需要处理的切片的结束索引
        subword2count: collections.defaultdict
            用来记录预分词子词的频数
        special_tokens: List[str]
            特殊token，这里要按照特殊标记把chunk分开， 免得特殊标记干扰预分词结果，而且也不会把两个不属于一个词的但是被特殊token连在一起的单词且在一起。
    """
    subword2count = defaultdict(int)
    with open(input_path, "rb") as f:
        if start is not None or end is not None:
            f.seek(start)
            chunk = f.read(end-start).decode("utf-8", errors="ignore")
        else:
            chunk = f.read().decode("utf-8", errors="ignore")
    
    escaped_tokens = [re.escape(token) for token in special_tokens] #对special token当中的|等符号转义
    docs = re.split("|".join(escaped_tokens), chunk)
    for text in docs:
        match_iter = re.finditer(PAT, text)
        for match_obj in match_iter:
            subword = match_obj.group()
            subword_bytes_seq = tuple([i for i in list(subword.encode("utf-8"))]) # 本质上是一个List[int]
            subword2count[subword_bytes_seq] += 1
    return subword2count

def pretokenize_chunk_wrapper(args):
    input_path, special_tokens, start, end = args
    return pretokenize_chunk(input_path, special_tokens, start, end)


def get_neighberbytes2count(subword2count:Dict[Tuple[int], int]):
    byte2counts = defaultdict(int)
    for subword_bytes_seq, count in subword2count.items():
        # 这个地方要把相邻两个字节对的出现频次做计数，在每个预分词的子词范围内
        if len(subword_bytes_seq)<2:
            continue
        for i, j in zip(subword_bytes_seq[:-1],subword_bytes_seq[1:]):
            byte2counts[(i, j)] += count * 1 
    return byte2counts

def merge_bytes(subword2count, tokenizer, byte2counts, merges):
    top_freq_pair = max(byte2counts.items(), key=lambda x: (x[1], tokenizer.return_tuple(x[0])))
    new_token_id = tokenizer.update(top_freq_pair[0])
    merges.append(
        (tokenizer.id2bytes[top_freq_pair[0][0]], tokenizer.id2bytes[top_freq_pair[0][1]])
    )
    byte2counts, subword2count = update_byte2count(top_freq_pair, new_token_id, subword2count, byte2counts)
    return byte2counts, subword2count


def merge_bytes_with_heap(subword2count, tokenizer, byte2counts, merges, tokens2freq_heap):
    top_freq_pair = tokens2freq_heap.pop()
    new_token_id = tokenizer.update(top_freq_pair[0])
    merges.append(
        (tokenizer.id2bytes[top_freq_pair[0][0]], tokenizer.id2bytes[top_freq_pair[0][1]])
    )
    tokens2freq_heap, subword2count = update_byte2count_with_heap(top_freq_pair, new_token_id, subword2count, byte2counts, tokens2freq_heap)
    return tokens2freq_heap, subword2count

def update_byte2count(to_merge_pair, token_id, subword2count, byte2counts):
    bytes_pair, bytes_pair_count = to_merge_pair[0], to_merge_pair[1]
    
    new_subword2count = defaultdict(int)
    for subword_bytes_seq, seq_count in subword2count.items():
        if seq_count <= 0:
            continue
        if bytes_pair[0] not in subword_bytes_seq or bytes_pair[1] not in subword_bytes_seq:
            new_subword2count[subword_bytes_seq] = subword2count[subword_bytes_seq]
            continue

        new_seq = []
        is_change = False
        # 更新子串的序列 原来的两个token合并成一个新的
        first_token_idx = 0
        while first_token_idx < len(subword_bytes_seq):
            if first_token_idx < len(subword_bytes_seq)-1 and subword_bytes_seq[first_token_idx] == bytes_pair[0] and subword_bytes_seq[first_token_idx + 1] == bytes_pair[1]:
                is_change = True
                new_seq.append(token_id)
                first_token_idx += 2 # 把后面那个也跳过去
            else:
                new_seq.append(subword_bytes_seq[first_token_idx])
                first_token_idx += 1
        
        new_seq = tuple(new_seq)
        if not is_change:
            # assert new_seq == subword_bytes_seq
            new_subword2count[subword_bytes_seq] = seq_count
        else:
            # former_id_seq = " ".join([str(num) for num in new_seq]).replace(f"{token_id}", f"{bytes_pair[0]} {bytes_pair[1]}")
            # later_id_seq = " ".join([str(num) for num in subword_bytes_seq])
            # assert former_id_seq==later_id_seq
            new_subword2count[new_seq] = seq_count # 替换subword2count当中对于某个单词/预分词片段的编码和计数
            """
            增量更新 相邻词元计数
            例如 a b c d c d e -> a b cd cd e
            计数要做下面的改变：bc、de、dc的频次-整个序列的频次，cdcd 加2*整个序列的频次
            操作起来分下面三组情况： b cd ; cd cd  ; cd e
            去掉所有 c d 的相邻词元组计数
            下面的直接变更+max的方法也是可以的
            """
            byte2counts[bytes_pair] = 0
            for former_token_id, later_token_id in zip(new_seq[:-1], new_seq[1:]):
                if former_token_id == token_id and later_token_id == token_id: # cd cd
                    byte2counts[(former_token_id, later_token_id)] += seq_count
                    # 中间两个的相邻词元组计数也要减掉 seq_count 
                    byte2counts[(bytes_pair[1], bytes_pair[0])] -= seq_count
                elif former_token_id == token_id: # cd e
                    byte2counts[(former_token_id, later_token_id)] += seq_count
                    byte2counts[(bytes_pair[1], later_token_id)] -= seq_count
                elif later_token_id == token_id: # b cd
                    byte2counts[(former_token_id, later_token_id)] += seq_count
                    byte2counts[(former_token_id, bytes_pair[0])] -= seq_count 
            
    return byte2counts, new_subword2count


def update_byte2count_with_heap(to_merge_pair, token_id, subword2count, byte2counts, tokens2freq_heap):
    bytes_pair, bytes_pair_count = to_merge_pair[0], to_merge_pair[1]
    
    new_subword2count = defaultdict(int)
    for subword_bytes_seq, seq_count in subword2count.items():
        if seq_count <= 0:
            continue
        if bytes_pair[0] not in subword_bytes_seq or bytes_pair[1] not in subword_bytes_seq:
            new_subword2count[subword_bytes_seq] = subword2count[subword_bytes_seq]
            continue

        new_seq = []
        is_change = False
        # 更新子串的序列 原来的两个token合并成一个新的
        first_token_idx = 0
        while first_token_idx < len(subword_bytes_seq):
            if first_token_idx < len(subword_bytes_seq)-1 and subword_bytes_seq[first_token_idx] == bytes_pair[0] and subword_bytes_seq[first_token_idx + 1] == bytes_pair[1]:
                is_change = True
                new_seq.append(token_id)
                first_token_idx += 2 # 把后面那个也跳过去
            else:
                new_seq.append(subword_bytes_seq[first_token_idx])
                first_token_idx += 1
        
        new_seq = tuple(new_seq)
        if not is_change:
            new_subword2count[subword_bytes_seq] = seq_count
        else:
            new_subword2count[new_seq] = seq_count # 替换subword2count当中对于某个单词/预分词片段的编码和计数

            """用堆来维护相邻词元组计数，优化速度""" 
            tokens2freq_heap.update(bytes_pair, 0)
            for former_token_id, later_token_id in zip(new_seq[:-1], new_seq[1:]):
                if former_token_id == token_id and later_token_id == token_id: # cd cd
                    # 中间两个的相邻词元组计数也要减掉 seq_count 
                    tokens2freq_heap.update_increment((former_token_id, later_token_id), seq_count)
                    tokens2freq_heap.update_increment((bytes_pair[1], bytes_pair[0]), -seq_count)
                elif former_token_id == token_id: # cd e
                    tokens2freq_heap.update_increment((former_token_id, later_token_id), seq_count)
                    tokens2freq_heap.update_increment((bytes_pair[1], later_token_id), -seq_count)
                elif later_token_id == token_id: # b cd
                    tokens2freq_heap.update_increment((former_token_id, later_token_id), seq_count)
                    tokens2freq_heap.update_increment((former_token_id, bytes_pair[0]), -seq_count) 
            
    return tokens2freq_heap, new_subword2count


def train_bpe(input_path:str, vocab_size:int, special_tokens:List[str], num_processes:int=4):
    """
    special_tokens:  
        要添加到词汇表的字符串列表，例如<|endoftext|>、<cls>。
        在用正则做预分词之前，应该从语料库（或块，如果使用并行实现）中删除所有特殊标记，确保特殊标记所分隔的文本之间不会发生合并，
        且在特殊标记上进行拆分，能够让这些特殊标记不影响BPE训练。
    """    
    # 这里切出对应数量的粗粒度的chunk就可以了
    all_boundaries = [] # 至少要有 num_processes+1 个元素
    with open(input_path, "rb") as f:
        for special_split_token in special_tokens:
            boundaries = find_chunk_boundaries(f, num_processes, special_split_token.encode()) 
            # 读的是bytes对象，所以这里special_split_token要做编码
            all_boundaries.extend(boundaries)
            if len(all_boundaries) >= num_processes + 1:
                all_boundaries = sorted(set(all_boundaries))
                break
    # print("all_boundaries:", all_boundaries)
            
    # 用多进程执行正则
    boundary_pairs = list(zip(all_boundaries[:-1], all_boundaries[1:]))
    to_do = [(input_path, special_tokens, b[0], b[1]) for b in boundary_pairs]
    with multiprocessing.Pool(processes = num_processes) as pool:
        results = pool.map(pretokenize_chunk_wrapper, to_do)

    # 合并预分词计数结果
    subword2count = defaultdict(int)
    for count_dict in results:
        for word, count in count_dict.items():
            subword2count[word] += count
    
    merges = []
    tokenizer = BPEtokenizer(special_tokens)
    neighberByte2count = get_neighberbytes2count(subword2count)
    tokens2freq_heap = heapManager(tokenizer, neighberByte2count)

    """
    增量更新的时候下面用堆和加减byte2count都是可以的，但是,好像时间上并不会差很多，可能也是堆没有写的很高效的原因？
    """
    # while tokenizer.vocab_size < vocab_size and len(neighberByte2count)>0:
    #     neighberByte2count, subword2count = merge_bytes(subword2count, tokenizer, neighberByte2count, merges)
    
    while tokenizer.vocab_size < vocab_size and len(tokens2freq_heap.heap)>0:
        tokens2freq_heap, subword2count = merge_bytes_with_heap(subword2count, tokenizer, neighberByte2count, merges, tokens2freq_heap)
    return tokenizer.id2bytes, merges



class BPEtokenizer:
    def __init__(self, special_tokens:List[str]):
        self.bytes2id = {bytes([i]):i for i in range(256)}
        for token in special_tokens:
            self.bytes2id[token.encode("utf-8")] = len(self.bytes2id)
        self.id2bytes = {token_id: token for token, token_id in self.bytes2id.items()}
        self.vocab_size = len(self.bytes2id)

    def return_tuple(self, token_ids):
        res = ()
        for token_id in token_ids:
            res = res + (self.id2bytes[token_id],)
        return res

    def decode(self, token_ids):
        if isinstance(token_ids, int):
            return self.id2bytes[token_id]
        return b"".join([self.id2bytes[token_id] for token_id in token_ids])

    def update(self, bytes_pair:Tuple[int]):
        if bytes_pair not in self.bytes2id:
            bytes_str = self.decode(bytes_pair)
            token_id = len(self.bytes2id)

            self.bytes2id[bytes_str] = token_id
            self.id2bytes[token_id] = bytes_str
            self.vocab_size += 1
        
        return token_id


class BytesPairPriority:
    """包装字节序列，实现反转的字典序比较"""
    def __init__(self, bytes_obj):
        self.bytes_obj = bytes_obj

    def __lt__(self, other):
        """小于比较：字节序列字典序大的，在这个比较中"更小" """
        # 正常字典序：b'AB' < b'BA' (因为 65 < 66)
        # 但我们想要：b'BA' 在堆的优先级更高，所以在比较中应该 b'BA' < b'AB'
        return self.bytes_obj > other.bytes_obj

    def __eq__(self, other):
        return self.bytes_obj == other.bytes_obj



class heapManager:
    def __init__(self, tokenizer, byte2count):
        self.heap = heapdict.heapdict()
        self.tokenizer = tokenizer
        for token_pair, freq in byte2count.items():
            self.update(token_pair, freq)
    
    def _make_priority(self, token_pair, freq):
        """优先级元组"""
        bytes_pair = self.tokenizer.return_tuple(token_pair)
        priority_reversed_bytes_pair = (BytesPairPriority(bytes_pair[0]), BytesPairPriority(bytes_pair[1]))
        return (-freq, priority_reversed_bytes_pair)
    
    def update(self, token_pair, new_freq):
        priority = self._make_priority(token_pair, new_freq)
        self.heap[token_pair] = priority

    def update_increment(self, token_pair, delta):
        """增量更新单个token_pair的频率"""
        old_priority = self.heap.get(token_pair, (0,))
        old_freq = -old_priority[0]
        new_freq = old_freq + delta
        
        self.update(token_pair, new_freq)

    def pop(self):
        """弹出堆顶元素"""
        try:
            token_pair, priority = self.heap.popitem()
            freq = -priority[0]
            return (token_pair, freq)
        except KeyError:
            return None

    
def test_byte_compare():
    a = BytesPairPriority("AB")
    b = BytesPairPriority("BA")
    c = BytesPairPriority("B")
    print(b<a)
    print(b<c)

def test_heapdict():
    test = {
        (1,2): 1,
        (1,3): 1,
        (1,4): 1,
        (2,4): 2,
        (3,4): 3,
        (3,3): 3,
        (2,3): 3
    }
    tokenizer = BPEtokenizer([])
    heap = heapManager(tokenizer, test)
    while heap.heap:
        print(heap.pop())

def test_train_bpe_offline():
    import time
    input_path = "/mnt/lp-dw/zhang_yingying/test/cs336/assignment/assignment1/assignment1-basics/tests/fixtures/corpus.en"
    start_time = time.time()
    _, _ = train_bpe(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )
    end_time = time.time()
    print(f"cost: {end_time-start_time:.2f}s")

if __name__=="__main__":
    # test_byte_compare()
    # test_heapdict()
    test_train_bpe_offline()
