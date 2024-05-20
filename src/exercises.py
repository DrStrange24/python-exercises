import math
import time
from typing import Any, Counter, List, NamedTuple, Optional, Tuple

class ListNode:
    def __init__(self, val=0, next=None):
        self.val:Any = val
        self.next:ListNode = next

    @staticmethod
    def _static_init_list(l:List):
        if len(l) == 0: return None
        head = ListNode(l[0])
        cur_head = head
        for v in l[1:]:
            cur_head.next = ListNode(v)
            cur_head = cur_head.next
        return head

    @staticmethod
    def _static_print_list(head:'ListNode'):
        while head:
            print(head.val)
            head = head.next
    
    @staticmethod
    def _static_get_list(head:'ListNode'):
        list = []
        while head:
            list.append(head.val)
            head = head.next
        return list

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
    
    @staticmethod
    def init_tree(l:list):
        root = TreeNode(l[0])
        q = [root]
        i = 1
        while i < len(l):
            cur_root = q.pop(0)
            if l[i]:
                cur_root.left = TreeNode(l[i])
                q.append(cur_root.left)
            i+=1
            if i < len(l) and l[i]:
                cur_root.right = TreeNode(l[i])
                q.append(cur_root.right)
            i+=1
        return root
    
    @staticmethod
    def get_list(root:'TreeNode'):
        if root is None:
            return []
        res = []
        q = [root]
        while q:
            node = q.pop(0)
            if node:
                res.append(node.val)
                q.append(node.left)
                q.append(node.right)
            else: res.append(None)
        while res and res[-1] is None: res.pop()
        return res

def track_time_execution(func:callable):
    start_time = time.time()
    func()
    end_time = time.time()
    execution_time_seconds = end_time - start_time
    execution_time_ms = execution_time_seconds * 1000
    execution_time_ms = round(execution_time_ms, 2)
    print("Execution time:", execution_time_ms, "ms")

class TestCase(NamedTuple):
    result: Any
    expected: Any

class main_exercises():
    def __init__(s):
        s.compare_linkedlist_test()
        # pass

    def test1(s,result,expected):
        # print(result,expected)
        print('pass' if result==expected else 'failed')
    def test2(s,test_cases:List[TestCase],show_details:bool = False):
        RED = '\033[91m'
        GREEN = '\033[92m'
        END = '\033[0m'
        passed_count = 0
        def successed_message(message:str) -> str: return f'{GREEN}{message}{END}'
        def failed_message(message:str) -> str: return f'{RED}{message}{END}'
        for i in range(len(test_cases)): 
            if test_cases[i].result == test_cases[i].expected:
                test_result = successed_message('Passed')
                passed_count+=1
            else: test_result = failed_message('Failed')
            message = f'Case {i+1}: {test_result}'
            if show_details: message += f' Result: {test_cases[i].result} Expected: {test_cases[i].expected}'
            print(message)
        print()
        if passed_count==len(test_cases):
            test_result = successed_message('Passed')
            print(f'All test cases have {test_result}')
        else:
            test_result = failed_message(f'{len(test_cases)-passed_count} Failed')
            print(f'Passed:{passed_count}/{len(test_cases)} => {test_result}')
        print()
    
    def containsNearbyDuplicate(s, nums, k: int) -> bool:
        for i in range(len(nums)):
            if nums[i] in nums[i+1:]:
                j = nums[i+1:].index(nums[i])
                if abs(i-j) <= k:
                    return True
        return False

    def max_len_substring_test(s):
        s.test1(s.maximumLengthSubstring('bcbbbcba'),4)
        s.test1(s.maximumLengthSubstring('aaaa'),2)
        s.test1(s.maximumLengthSubstring('sdaxczfdfsrere'),14)
    def maximumLengthSubstring(self, s: str) -> int:
        l,r,ht,ml = 0,0,{},0
        while r<len(s) and l <= len(s)-ml:
            if s[r] not in ht: ht[s[r]] = 1
            elif ht[s[r]] < 2: ht[s[r]] += 1
            elif ht[s[r]] == 2:
                while ht[s[r]] == 2:
                    ht[s[l]]-=1
                    l+=1
                ht[s[r]] += 1
            r+=1
            if r - l > ml: ml = r - l
        return ml
    
    def is_happy_test(s):
        s.test1(s.isHappy(19),True)
        s.test1(s.isHappy(2),False)
        s.test1(s.isHappy(100),True)
        s.test1(s.isHappy(102),False)
        s.test1(s.isHappy(1),True)
    def isHappy(s, n: int) -> bool:
        str_num = str(n)
        num_set = {n}
        while True:
            tem_sum = 0
            for i in range(len(str_num)): tem_sum += int(str_num[i])**2
            if tem_sum == 1: return True
            else:
                str_num = str(tem_sum)
                if tem_sum in num_set: return False
                else: num_set.add(tem_sum)

    def double_it_test(s):
        s.test1(ListNode._static_get_list(s.doubleIt(ListNode._static_init_list([1,8,9]))),[3,7,8])
        s.test1(ListNode._static_get_list(s.doubleIt(ListNode._static_init_list([9,9,9]))),[1,9,9,8])
    def doubleIt(s, head: Optional[ListNode]) -> Optional[ListNode]:
        stack = []
        while head:
            stack.append(head.val)
            head = head.next
        cur_stack = stack.pop() * 2
        r = 0
        if cur_stack > 9: 
            cur_stack -= 10
            r = 1
        new_head = ListNode(cur_stack)
        while stack:
            cur_stack = stack.pop()
            cur_stack*=2
            if r == 1:
                cur_stack+=1
                r = 0
            if cur_stack > 9: 
                cur_stack -= 10
                r = 1
            cur_node = ListNode(cur_stack)
            cur_node.next = new_head
            new_head = cur_node
        if r == 1:
            cur_node = ListNode(1)
            cur_node.next = new_head
            new_head = cur_node
        return new_head

    def intersect_test(s):
        s.test1(s.intersect([1,2,2,1],[2,2]),[2,2])
        s.test1(s.intersect([4,9,5],[9,4,9,8,4]),[4,9] or [9,4])
    def intersect(s, nums1: List[int], nums2: List[int]) -> List[int]:
        cn1 = Counter(nums1)
        cn2 = Counter(nums2)
        ans = []
        for k in cn1:
            if k in cn2: ans.extend([k] * min(cn1[k],cn2[k]))
        return ans
    
    def reverse_only_letters_test(s):
        s.test2([
            TestCase(s.reverse_only_letters('ab-cd'),'dc-ba'),
            TestCase(s.reverse_only_letters('a-bC-dEf=ghlj!!'),'j-lh-gfE=dCba!!'),
        ])
    def reverse_only_letters(s,string:str):
        ls = list(string)
        l,r = 0,len(ls)-1
        while l < r:
            while not ls[l].isalpha():
                if l >= r: return ''.join(ls)
                l+=1
            while not ls[r].isalpha():
                if l >= r: return ''.join(ls)
                r-=1
            
            ls[l], ls[r] = ls[r], ls[l]
            l+=1
            r-=1
        return ''.join(ls)
    
    def max_depth_test(s):
        s.test1(s.maxDepth(TreeNode.init_tree([3,9,20,None,None,15,7])),3)
        s.test1(s.maxDepth(TreeNode.init_tree([1,None,2])),2)
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        d = 0
        q = [root]
        while q:
            d+=1
            cur_level = len(q)
            for _ in range(cur_level):
                cur_q = q.pop(0)
                if cur_q.left: q.append(cur_q.left)
                if cur_q.right: q.append(cur_q.right)
        return d
    
    def reverse_test(s):
        s.test2([
            TestCase(s.reverse(123),321),
            TestCase(s.reverse(-123),-321),
            TestCase(s.reverse(120),21),
            TestCase(s.reverse(0),0),
            TestCase(s.reverse(1),1),
            TestCase(s.reverse(-1),-1),
            TestCase(s.reverse(123456789),987654321),
            TestCase(s.reverse(900000),9),
            TestCase(s.reverse(1534236469),0),
        ])
        print()
        s.test2([
            TestCase(s.reverse_math(123),321),
            TestCase(s.reverse_math(-123),-321),
            TestCase(s.reverse_math(120),21),
            TestCase(s.reverse_math(0),0),
            TestCase(s.reverse_math(1),1),
            TestCase(s.reverse_math(-1),-1),
            TestCase(s.reverse_math(123456789),987654321),
            TestCase(s.reverse_math(900000),9),
            TestCase(s.reverse_math(1534236469),0),
        ])
    def reverse(self, x: int) -> int:
        if not (-2**31 <= x <= 2**31 - 1): return 0
        ans = int(str(x)[::-1].replace('-',''))
        if x < 0: ans *= -1
        if not (-2**31 <= ans <= 2**31 - 1): return 0
        return ans
    def reverse_math(self, x: int) -> int:
        cur_x = x
        if x == 0: return 0
        np10 = 10 ** math.floor(math.log10(abs(x)))
        if x < 0: cur_x *= -1
        start = 10
        end = np10 * 10
        div = np10
        s = 0
        while start <= end:
            num = cur_x % start
            cur_x -= num
            num/=start/10
            num *= div
            div/=10
            s +=num
            start *= 10
        if x < 0: s *= -1
        s = int(s)
        if not (-2**31 <= s <= 2**31 - 1): return 0
        return s
    
    def binary_search_test(s):
        s.test2([
            TestCase(s.binary_search([],1),None),
            TestCase(s.binary_search([1],1),0),
            TestCase(s.binary_search([1],2),None),
            TestCase(s.binary_search([1,2],1),0),
            TestCase(s.binary_search([1,2],2),1),
            TestCase(s.binary_search([1,2],3),None),
            TestCase(s.binary_search([1,2],-1),None),
            TestCase(s.binary_search([1,2,3],1),0),
            TestCase(s.binary_search([1,2,3],2),1),
            TestCase(s.binary_search([1,2,3],3),2),
            TestCase(s.binary_search([1,2,3],4),None),
            TestCase(s.binary_search([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 5),4),
            TestCase(s.binary_search([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 1),0),
            TestCase(s.binary_search([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 10),9),
            TestCase(s.binary_search([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0),None),
            TestCase(s.binary_search([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 11),None),
            TestCase(s.binary_search([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 12),None),
            TestCase(s.binary_search([1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10], 2),2 or 1),
            TestCase(s.binary_search([1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10], 7),7),
            TestCase(s.binary_search([5], 5),0),
            TestCase(s.binary_search([5], 10),None),
            TestCase(s.binary_search(range(1,1001), 500),499),
            TestCase(s.binary_search(range(1,1001), 1000),999),
            TestCase(s.binary_search(range(1,1001), 2000),None),
        ])
    def binary_search(s, data, target):
        left, right = 0, len(data) - 1
        while left <= right:
            mid = (left+right) // 2
            if target == data[mid]: return mid
            if target > data[mid]: left = mid + 1
            else: right = mid - 1
        return None
    
    def compare_linkedlist_test(s):
        def short_func(list1:List,list2:List): return s.compare_linkedlist(ListNode._static_init_list(list1),ListNode._static_init_list(list2))
        s.test2([
            TestCase(short_func([],[]),True),
            TestCase(short_func([1,2,3,4],[1,2,3,4]),True),
            TestCase(short_func([1,2,3,4],[1,2,3]),False),
            TestCase(short_func([1, 2, 3, 4, 5],[1, 2, 3, 4, 6]),False),
            TestCase(short_func([1, 2, 3, 4, 5],[5, 4, 3, 2, 1]),False),
            TestCase(short_func([1, 2, 3, 4, 5],[1, 2, 3]),False),
            TestCase(short_func([1, 2, 3, 4, 5],['a', 'b', 'c', 'd', 'e']),False),
            TestCase(short_func([1, 1, 2, 3, 3, 4, 4, 5],[1, 2, 2, 3, 4, 5]),False),
            TestCase(short_func([None, None, None],[None, None, None]),True),
            TestCase(short_func([1, 'a', None],['b', 2, None]),False),
            TestCase(short_func([1.0, 2.5, 3.7],[1, 2.5, 3.7]),False),
            TestCase(short_func([(1, 2), (3, 4)],[(1, 2), (3, 4)]),True),
            TestCase(short_func([{'a': 1}, {'b': 2}],[{'a': 1}, {'b': 2}]),True),
            TestCase(short_func([[1, 2], [3, 4]],[[1, 2], [3, 4]]),True),
            TestCase(short_func([[]],[[]]),True),
            TestCase(short_func([math.nan],[math.nan]),False),
        ])
    def compare_linkedlist(s,head1: Optional[ListNode],head2: Optional[ListNode]) -> bool:
        while head1 or head2:
            if (head1 is None 
                or head2 is None 
                or type(head1.val)!=type(head2.val) 
                or head1.val!=head2.val
                ): return False
            head1 = head1.next
            head2 = head2.next
        return True

    def reverse_linkedlist_test(s):
        def shortcut(l:List): return ListNode._static_get_list(s.reverse_linkedlist(ListNode._static_init_list(l)))
        s.test2([
            TestCase(shortcut([]),[]),
            TestCase(shortcut([0]),[0]),
            TestCase(shortcut([1]),[1]),
            TestCase(shortcut([5]),[5]),
            TestCase(shortcut([1,2,3]),[3,2,1]),
            TestCase(shortcut([1, 2, 3, 4, 5]),[5, 4, 3, 2, 1]),
            TestCase(shortcut([1, 1, 2, 2, 3, 3]),[3, 3, 2, 2, 1, 1]),
            TestCase(shortcut(['a', 2, 'b', 4.5, True]),[True, 4.5, 'b', 2, 'a']),
            TestCase(shortcut([1, None, 3, None, 5]),[5, None, 3, None, 1]),
            TestCase(shortcut([[1, 2], [3, 4], [5, 6]]),[[5, 6], [3, 4], [1, 2]]),
        ])
    def reverse_linkedlist(s,head: Optional[ListNode]) -> Optional[ListNode]:
        '''by chatgpt'''
        prev = None
        current = head
        while current:
            next_node = current.next
            current.next = prev
            prev = current
            current = next_node
        return prev
    
    def alt_subsequence_best_test(s) -> int:
        s.test2([
            TestCase(s.alt_subsequence_best([0, 1, 0, 1, 0]),5),
            TestCase(s.alt_subsequence_best([0]),1),
            TestCase(s.alt_subsequence_best([1]),1),
            TestCase(s.alt_subsequence_best([0, 0, 0, 0, 0]),1),
            TestCase(s.alt_subsequence_best([1, 1, 1, 1, 1]),1),
            TestCase(s.alt_subsequence_best([0, 1, 0, 1, 0, 1, 0]),7),
            TestCase(s.alt_subsequence_best([1, 0, 1, 0, 1, 0, 1]),7),
            TestCase(s.alt_subsequence_best([0, 0, 1, 1, 0, 0, 1, 1, 0]),2),
            TestCase(s.alt_subsequence_best([1, 1, 0, 0, 1, 1, 0, 0, 1]),2),
            TestCase(s.alt_subsequence_best([0, 1] * 50000),100000),
            TestCase(s.alt_subsequence_best([1, 0] * 50000),100000),
            TestCase(s.alt_subsequence_best([0, 1, 0, 1, 0, 1, 0, 0]),7),
            TestCase(s.alt_subsequence_best([1, 0, 1, 0, 1, 0, 1, 1]),7),
            TestCase(s.alt_subsequence_best([1,1,0,1,0]),4),
            TestCase(s.alt_subsequence_best([1,1,1,1,0,0,1,0,1,0]),5),
            TestCase(s.alt_subsequence_best([1,1,0,1,0,0,1,1,1,0]),4),
            TestCase(s.alt_subsequence_best([0, 0, 0, 0, 0, 0]),1),
            TestCase(s.alt_subsequence_best([1, 1, 1, 1, 1, 1]),1),
        ])
    def alt_subsequence_best(s,X:List[int]) -> int:
        c,m = 1,0
        for i in range(len(X)-1):
            if X[i] != X[i+1]: c+=1
            else:
                if c > m: m = c
                c = 1
        if c > m: m = c
        return m
    
    def squared_sum_test(s):
        s.test2([
            TestCase(s.squared_sum_1([1,-1,1,-1,1]),1),
            TestCase(s.squared_sum_1([1,2,3]),36),
            TestCase(s.squared_sum_1([0]),0),
            TestCase(s.squared_sum_1([]),0),
            TestCase(s.squared_sum_1([1, -2, 3, 4]),49),
            TestCase(s.squared_sum_1([1, 2, 3, 4]),100),
            TestCase(s.squared_sum_1([-1, -2, -3, -4]),1),
            TestCase(s.squared_sum_1([0, -1, 2, -3, 4]),16),
        ])
    def squared_sum(s,n:List[int]) -> int:
        l,s = 0, 0
        for r in range(1,len(n)):
            pass
        return s**2
    def squared_sum_1(s,n:List[int]) -> int:
        ms = mc = 0
        for num in n:
            ms = max(0,ms+num)
            mc = max(mc,ms**2)
        return mc
