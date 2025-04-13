#!/usr/bin/env python
# coding: utf-8

# # Assignment 05

# ## 과제 3:heap.py 를 이용한 생일 힙 정렬

# In[1]:


from datetime import datetime
import pandas as pd

df=pd.read_csv(r"C:\Users\user\OneDrive\바탕 화면\희서\성신여대\3학년\자료구조\birthday.csv")
# CSV 파일 읽기

df['birthday'] = pd.to_datetime(df['생년월일8자리(예.20040101)'], format='%Y%m%d', errors='coerce')
df = df.dropna(subset=['birthday'])  # 날짜 오류 제거

df.head()
# 힙 코드 정의 (heap.py 기반 수정 버전)
class Heap:
    def __init__(self, args):
        if len(args) != 0:
            self.__A = args[0]
        else:
            self.__A = []

    def insert(self, x):
        self.__A.append(x)
        self.__percolateUp(len(self.__A) - 1)

    def __percolateUp(self, i: int):
        parent = (i - 1) // 2
        if i > 0 and self.__A[i][1] > self.__A[parent][1]:
            self.__A[i], self.__A[parent] = self.__A[parent], self.__A[i]
            self.__percolateUp(parent)

    def delteMax(self):
        if not self.isEmpty():
            max = self.__A[0]
            self.__A[0] = self.__A.pop()
            self.__percolateDown(0)
            return max
        else:
            return None

    def __percolateDown(self, i: int):
        child = 2 * i + 1
        right = 2 * i + 2
        if child <= len(self.__A) - 1:
            if right <= len(self.__A) - 1 and self.__A[child][1] < self.__A[right][1]:
                child = right
            if self.__A[i][1] < self.__A[child][1]:
                self.__A[i], self.__A[child] = self.__A[child], self.__A[i]
                self.__percolateDown(child)

    def isEmpty(self) -> bool:
        return len(self.__A) == 0

    def size(self) -> int:
        return len(self.__A)

# 힙에 생일 삽입
heap = Heap([[]])
for _, row in df.iterrows():
    heap.insert((row['이름'], row['birthday']))

# 생일이 느린 순서로 10명 출력
for _ in range(min(10, heap.size())):
    name, birth = heap.delteMax()
    print(name, birth.strftime('%Y-%m-%d'))


# ## 과제 4: circularDoublyLinkedList.py 를 이용한 조원 필터링

# In[2]:


class ListNode:
    def __init__(self, item, next=None):
        self.item = item
        self.next = next

class CircularLinkedList:
    def __init__(self):
        self.__tail = ListNode("dummy", None)
        self.__tail.next = self.__tail
        self.__numItems = 0

    def append(self, newItem) -> None:
        newNode = ListNode(newItem, self.__tail.next)
        self.__tail.next = newNode
        self.__tail = newNode
        self.__numItems += 1

    def __iter__(self):
        return CircularLinkedListIterator(self)

    def getNode(self, i: int) -> ListNode:
        curr = self.__tail.next
        for index in range(i + 1):
            curr = curr.next
        return curr

class CircularLinkedListIterator:
    def __init__(self, alist):
        self.__head = alist.getNode(-1)
        self.iterPosition = self.__head.next

    def __next__(self):
        if self.iterPosition == self.__head:
            raise StopIteration
        else:
            item = self.iterPosition.item
            self.iterPosition = self.iterPosition.next
            return item

    def __iter__(self):
        return self

# 같은 조 친구들 이름
group_names = {"이희서", "이채연", "이지후", "김지우", "신수민", "강윤서", "윤여빈", "이예린", "김나영", "김명신"}

# 리스트에 추가
cdll = CircularLinkedList()
for _, row in df.iterrows():
    if row['이름'] in group_names:
        cdll.append((row['이름'], row['birthday'].strftime('%Y-%m-%d')))

# 출력
for item in cdll:
    print(item[0], item[1])


# ## 과제 6: 교재 8장 우선 순위 큐 연습 문제

# ### 연습 문제 01
# 가능합니다. 최대 힙은 부모 >= 자식만 만족하면 되므로, 루트보다 더 깊은 노드가 루트보다 큰 값을 가져도 힙 조건은 만족할 수 없습니다.

# ### 연습 문제 02
# A[0]은 항상 가장 큰 값입니다. A[n-1]은 항상 가장 작은 값은 아닙니다. 리프 노드 중 하나일 뿐입니다. 

# ### 연습 문제 03
# 리프 노드는 스며내리기 필요 없습니다. 리프 노드 수는 floor(n/2)부터 n-1까지로 그 수는 약 n/2개 입니다.

# ### 연습 문제 04
#  Θ(log n)  -> 힙의 높이 만큼 비교 및 교환이 발생하기 때문입니다.

# ### 연습 문제 05
# 마지막 원소를 삭제하는 건 매우 간단한 작업입니다. 힙은 일반적으로 배열로 구현되고, 마지막 원소는 배열의 가장 끝 인덱스 A[n-1]입니다. 배열에서 마지막 원소를 제거하는 것은 시간 복잡도 O(1)의 작업입니다.

# ### 연습 문제 06
# 위에서부터 스며올리기 방식으로도 힙을 만들 수는 있습니다.
# 하지만 아래에서부터 스며내리기 방식이 더 효율적입니다. 
# 점근적 시간 복잡도 기준으로: 위에서부터:O(nlogn)이고 아래서부터 O(n)이므로 아래에서부터가 더 효율적입니다. 

# ### 연습 문제 07
# 스며올리기를 사용해야합니다. 

# In[ ]:


def sift_up(heap, i):
    while i > 0:
        parent = (i - 1) // 2
        if heap[i] > heap[parent]:
            heap[i], heap[parent] = heap[parent], heap[i]
            i = parent
        else:
            break


# ## LeetCode 703.Kth Largest Element in Stream 
