import collections

class Stack:
    def __init__(self, items=None, max_length=None):
        self._stack = collections.deque(maxlen=max_length)
        
        if items is not None:
            for item in items:
                self._stack.append(item)


    def __str__(self):
        return str(self._stack)

    def __getitem__(self, indices):
        return self._stack.__getitem__(indices)

    def __setitem__(self, key, value):
        self._stack.__setitem__(key, value)

    def __iter__(self):
        return self._stack.__iter__()    
    
    def __len__(self):
        return self._stack.__len__()

    def pop(self):
        return self._stack.pop()

    def push(self, item):
        self._stack.append(item)

if __name__ == '__main__':
    pass

