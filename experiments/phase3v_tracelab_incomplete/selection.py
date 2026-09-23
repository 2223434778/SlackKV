import math
def size_bucket(n): return 'small' if n<3072 else ('medium' if n<5632 else 'large')
def pause_bucket(n): return 'short' if n<500 else ('medium' if n<1500 else 'long')
def service_bucket(n):
    return next(x for x in (1024,2048,4096,6144,8192) if x>=n)
def ceil5(x): return int(math.ceil(x/5.0)*5)
def required_blocks(tokens,block_size): return int(math.ceil(tokens/block_size))
def h2d(total_bytes,minimum_bytes): return total_bytes,max(0,total_bytes-minimum_bytes)
