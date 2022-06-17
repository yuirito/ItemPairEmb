from typing import List
import numpy
def dist(x,y):
    return numpy.sqrt(numpy.sum((x - y) ** 2))
   
def compute(item_emb_1: List[float], item_emb_2: List[float]) -> float:
    a = numpy.array(item_emb_1[256:])
    b = numpy.array(item_emb_2[256:])
    s = 1 / (1 + dist(a,b))
    return s
