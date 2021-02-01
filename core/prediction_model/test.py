import heapq
import random
from multiprocessing import Manager
def addQuene(heap,i):
    heap.append(i)
    _siftdown(heap, 0, len(heap) - 1)
    return i
def main(quene):
    from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor,wait
    with ProcessPoolExecutor(max_workers=8) as threadPool:
        arr=[]
        for _ in range(100):
            arr.append(random.randint(1,100))
        futureArr=[]
        for a in arr:
            futureArr.append(threadPool.submit(addQuene,quene,a))
        wait(futureArr,return_when="ALL_COMPLETED")

def _siftdown(heap, startpos, pos):
    newitem = heap[pos]
    # Follow the path to the root, moving parents down until finding a place
    # newitem fits.
    while pos > startpos:
        parentpos = (pos - 1) >> 1
        parent = heap[parentpos]
        if newitem < parent:
            heap[pos] = parent
            pos = parentpos
            continue
        break
    heap[pos] = newitem
def heappop(heap):
    """Pop the smallest item off the heap, maintaining the heap invariant."""
    lastelt = heap.pop()    # raises appropriate IndexError if heap is empty
    if heap:
        returnitem = heap[0]
        heap[0] = lastelt
        _siftup(heap, 0)
        return returnitem
    return lastelt
def _siftup(heap, pos):
    endpos = len(heap)
    startpos = pos
    newitem = heap[pos]
    # Bubble up the smaller child until hitting a leaf.
    childpos = 2*pos + 1    # leftmost child position
    while childpos < endpos:
        # Set childpos to index of smaller child.
        rightpos = childpos + 1
        if rightpos < endpos and not heap[childpos] < heap[rightpos]:
            childpos = rightpos
        # Move the smaller child up.
        heap[pos] = heap[childpos]
        pos = childpos
        childpos = 2*pos + 1
    # The leaf at pos is empty now.  Put newitem there, and bubble it up
    # to its final resting place (by sifting its parents down).
    heap[pos] = newitem
    _siftdown(heap, startpos, pos)


if __name__ =='__main__':
    manager=Manager()
    quene=manager.list()
    print(type(quene))
    main(quene)
    print(quene)
    while True:
        try:
            print(heappop(quene))
        except IndexError:
            break