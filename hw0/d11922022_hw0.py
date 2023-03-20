# %%
from collections import defaultdict

"""
1. 島號從 1 開始，越大越危險。
2. 起點在 1 號島。
3. 下個島沒草，就不會過去。
4. 行動不會跳步，只會走一步。
5. 只要還有路，要盡可能地走。
6. 只要經過島，島上的草就被吃光
7. 若要走到下個島，只能去最安全，且仍有草的島 (不重複拜訪)
8. 每個島一開始都長滿草 (可拜訪)
"""


class stack:
    """
    A stack push from self.stack[-1] (append)
    """

    def __init__(self):
        self.stack = []

    def push(self, e):
        self.stack.insert(0, e)
        return self

    def pop(self):
        e = self.stack[0]
        del self.stack[0]
        return e

    def __not__(self):
        return not self.stack

    def __contains__(self, e):
        return e in self.stack

    def __str__(self):
        return f"[head] {self.stack} [tail]"

    def not_empty(self):
        return bool(self.stack)

    def top(self):
        return float("-inf") if not self.stack else self.stack[0]

    def reverse(self):
        return [self.stack[i] for i in range(len(self.stack) - 1, -1, -1)]


class Solution:
    def __init__(self):
        self._parse_stdin()

    def _parse_stdin(self):
        lines = []
        try:
            while x := input():
                lines.append(x)
        except:
            pass

        n, m = lines[0].split(" ")
        self.n, self.m = int(n), int(m)

        self.graph = defaultdict(list)
        for line in lines[1:]:
            if line:
                a, b = line.split(" ")
                a, b = int(a), int(b)
                self.graph[a].append(b)
                self.graph[b].append(a)
        else:
            for k in self.graph.keys():
                self.graph[k].sort()

    def succs(self, node):
        s = self.graph[node]
        s.reverse()
        return s

    # Feel free to define your own member function
    def __str__(self):
        return f"n:{self.n}, m:{self.m}, graph:{self.graph}"

    def solve(self):
        curr = 1
        explored = stack()
        fringe = stack().push(curr)

        while fringe.not_empty():
            curr = fringe.pop()

            succs = [node for node in self.succs(curr) if node > curr]
            for i, succ in enumerate(succs, 1):
                if succ not in explored and succ > curr and i == len(succs):
                    fringe.push(succ)
            else:
                explored.push(curr)

        ans = " ".join([str(a) for a in explored.reverse()])
        print(f"{ans}")
