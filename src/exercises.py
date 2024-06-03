import math
import time
from typing import Any, Counter, List, NamedTuple, Optional, Tuple

class ListNode:
    def __init__(self, val=None, next=None):
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
    def __init__(self, val=None, left=None, right=None):
        self.val:Any = val
        self.left:TreeNode = left
        self.right:TreeNode = right
    
    @staticmethod
    def init_tree(l:list):
        if len(l) == 0: return None
        root = TreeNode(l[0])
        q = [root]
        i = 1
        while i < len(l):
            cur_root = q.pop(0)
            if l[i] is not None:
                cur_root.left = TreeNode(l[i])
                q.append(cur_root.left)
            i+=1
            if i < len(l) and l[i] is not None:
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

def runtime(func:callable):
    start_time = time.time()
    func()
    end_time = time.time()
    execution_time_seconds = end_time - start_time
    execution_time_ms = execution_time_seconds * 1000
    execution_time_ms = round(execution_time_ms)
    print("Runtime:", execution_time_ms, "ms")
    print()

class TestCase(NamedTuple):
    result: Any
    expected: Any

class main_exercises():
    def __init__(s):
        runtime(s.max_sub_array_test)
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

        def test_result_pass(test_case:TestCase) -> bool:
            if isinstance(test_case.result,ListNode) and isinstance(test_case.expected,ListNode): return s.compare_linkedlist(test_case.result,test_case.expected)
            if isinstance(test_case.result,TreeNode) and isinstance(test_case.expected,TreeNode): return s.compare_tree(test_case.result,test_case.expected)
            return test_case.result == test_case.expected

        for i in range(len(test_cases)): 
            if test_result_pass(test_cases[i]):
                test_result = successed_message('Passed')
                passed_count+=1
            else: test_result = failed_message('Failed')
            message = f'Case {i+1}: {test_result}'
            if show_details:
                result, expected = test_cases[i].result, test_cases[i].expected
                if isinstance(result,ListNode) and isinstance(expected,ListNode): result, expected = ListNode._static_get_list(result), ListNode._static_get_list(expected)
                elif isinstance(result,TreeNode) and isinstance(expected,TreeNode): result, expected = TreeNode.get_list(result), TreeNode.get_list(expected)
                message += f'\n   Result:   {result}\n   Expected: {expected}'
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
    
    def search_data(s,algorithm: callable) -> List[TestCase]:
        return [
            TestCase(algorithm([],1),None),
            TestCase(algorithm([1],1),0),
            TestCase(algorithm([1],2),None),
            TestCase(algorithm([1,2],1),0),
            TestCase(algorithm([1,2],2),1),
            TestCase(algorithm([1,2],3),None),
            TestCase(algorithm([1,2],-1),None),
            TestCase(algorithm([1,2,3],1),0),
            TestCase(algorithm([1,2,3],2),1),
            TestCase(algorithm([1,2,3],3),2),
            TestCase(algorithm([1,2,3],4),None),
            TestCase(algorithm([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 5),4),
            TestCase(algorithm([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 1),0),
            TestCase(algorithm([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 10),9),
            TestCase(algorithm([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0),None),
            TestCase(algorithm([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 11),None),
            TestCase(algorithm([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 12),None),
            TestCase(algorithm([1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10], 7),7),
            TestCase(algorithm([5], 5),0),
            TestCase(algorithm([5], 10),None),
            TestCase(algorithm(range(1,1001), 500),499),
            TestCase(algorithm(range(1,1001), 1000),999),
            TestCase(algorithm(range(1,1001), 2000),None),
            TestCase(algorithm(range(1,100001), 1),0),
            TestCase(algorithm(range(1,100001), 50000),49999),
            TestCase(algorithm(range(1,100001), 100000),99999),
            TestCase(algorithm(range(1,10**7+1), 1),0),
            TestCase(algorithm(range(1,10**7+1), 10**7/2),10**7//2-1),
            TestCase(algorithm(range(1,10**7+1), 10**7),10**7-1),
        ]

    def binary_search_test(s):
        s.test2(s.search_data(s.binary_search))
    def binary_search(s, data:Optional[List[int]], target:int) -> Optional[int]:
        left, right = 0, len(data) - 1
        while left <= right:
            mid = (left+right) // 2
            if target == data[mid]: return mid
            if target > data[mid]: left = mid + 1
            else: right = mid - 1
        return None
    
    def linear_search_test(s):
        s.test2(s.search_data(s.linear_search))
    def linear_search(s, data:Optional[List[int]], target:int) -> Optional[int]:
        for i in range(len(data)):
            if data[i] == target: return i
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
            if (head1 is None or head2 is None 
                or type(head1.val)!=type(head2.val) 
                or head1.val!=head2.val
                ): return False
            head1 = head1.next
            head2 = head2.next
        return True

    def reverse_linkedlist_test(s):
        ctll = ListNode._static_init_list #convert to linked list
        tm = s.reverse_linkedlist #testing method
        s.test2([
            TestCase(tm(ctll([])),ctll([])),
            TestCase(tm(ctll([0])),ctll([0])),
            TestCase(tm(ctll([1])),ctll([1])),
            TestCase(tm(ctll([5])),ctll([5])),
            TestCase(tm(ctll([1,2,3])),ctll([3,2,1])),
            TestCase(tm(ctll([1, 2, 3, 4, 5])),ctll([5, 4, 3, 2, 1])),
            TestCase(tm(ctll([1, 1, 2, 2, 3, 3])),ctll([3, 3, 2, 2, 1, 1])),
            TestCase(tm(ctll(['a', 2, 'b', 4.5, True])),ctll([True, 4.5, 'b', 2, 'a'])),
            TestCase(tm(ctll([1, None, 3, None, 5])),ctll([5, None, 3, None, 1])),
            TestCase(tm(ctll([[1, 2], [3, 4], [5, 6]])),ctll([[5, 6], [3, 4], [1, 2]])),
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

    def compare_tree_test(s):
        ctt = TreeNode.init_tree #convert to tree
        tm = s.compare_tree #testing method
        s.test2([
            TestCase(tm(ctt([]),ctt([])),True),
            TestCase(tm(ctt([]),ctt([1])),False),
            TestCase(tm(ctt([1]),ctt([])),False),
            TestCase(tm(ctt([1]),ctt([1])),True),
            TestCase(tm(ctt([1,2,3]),ctt([1,2,3])),True),
            TestCase(tm(ctt([1,2,3]),ctt([1,3,2])),False),
            TestCase(tm(ctt([1,2,3,4,5,6,7]),ctt([1,2,3,4,5,6,7])),True),
            TestCase(tm(ctt([1,2,3,4,5,6,7]),ctt([1,2,5,3,4,6,7])),False),
            TestCase(tm(ctt([1, 2, 3, None, 4]),ctt([1, 2, 3, None, 4])),True),
            TestCase(tm(ctt([1, 2]),ctt([2, 1])),False),
            TestCase(tm(ctt([1, 2, 3]),ctt([1, 2, None, 3])),False),
            TestCase(tm(ctt([1, 2, 3, 4]),ctt([1, 2, 3, 4, 5])),False),
            TestCase(tm(ctt([1, 2, 3]),ctt([1, 2, 4])),False),
            TestCase(tm(ctt([1, None, 2, None, 3,None, 4]),ctt([1, None, 2, None, 3,None, 4])),True),
            TestCase(tm(ctt([1, 2, None, 3,None, 4]),ctt([1, 2, None, 3,None, 4])),True),
        ])
    def compare_tree(s,root1:Optional[TreeNode],root2:Optional[TreeNode])->bool:
        if root1 is None and root2 is None: return True
        if root1 is None or root2 is None: return False
        return root1.val==root2.val and s.compare_tree(root1.left,root2.left) and s.compare_tree(root1.right,root2.right)

    def invert_tree_test(s):
        cltt = TreeNode.init_tree #convert list to tree
        tm = s.invert_tree #testing method
        s.test2([
            TestCase(tm(cltt([])),cltt([])),
            TestCase(tm(cltt([1])),cltt([1])),
            TestCase(tm(cltt([1,2])),cltt([1,None,2])),
            TestCase(tm(cltt([1,2,3])),cltt([1,3,2])),
            TestCase(tm(cltt([1,2,3,4,5,6,7])),cltt([1,3,2,7,6,5,4])),
            TestCase(tm(cltt([4,2,7,1,3,6,9])),cltt([4,7,2,9,6,3,1])),
            TestCase(tm(cltt([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,None,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]))
                     ,cltt([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,None,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,0,0,0,0,0,0,0,0,0,0,0,0])),
        ])
    def invert_tree(s, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root: return None
        root.left, root.right = root.right, root.left 
        s.invert_tree(root.left)
        s.invert_tree(root.right)
        return root
    
    def island_perimeter_test(s):
        s.test2([
            TestCase(s.islandPerimeter([[0,1,0,0],[1,1,1,0],[0,1,0,0],[1,1,0,0]]),16),
            TestCase(s.islandPerimeter([[1]]),4),
            TestCase(s.islandPerimeter([[1,0]]),4),
            TestCase(s.islandPerimeter([[1,1],[1,1]]),8),
        ])
    def islandPerimeter(self, grid: List[List[int]]) -> int:
        '''by chatgpt'''
        r,c,p = len(grid),len(grid[0]), 0
        for i in range(r):
            for j in range(c):
                if grid[i][j] == 1:
                    p += 4
                    if j > 0 and grid[i][j - 1] == 1: p -= 2
                    if i > 0 and grid[i - 1][j] == 1: p -= 2
        return p

    def sort_data(s,algorithm: callable) -> List[TestCase]:
        return [
            TestCase(algorithm([8,7,6,5,4,3,2,1]),[1,2,3,4,5,6,7,8]),
            TestCase(algorithm([5, 3, 8, 1, 2]),[1, 2, 3, 5, 8]),
            TestCase(algorithm([1, 2, 3, 4, 5]),[1, 2, 3, 4, 5]),
            TestCase(algorithm([5, 4, 3, 2, 1]),[1, 2, 3, 4, 5]),
            TestCase(algorithm([7, 7, 7, 7, 7]),[7, 7, 7, 7, 7]),
            TestCase(algorithm([10]),[10]),
            TestCase(algorithm([10, 5]),[5, 10]),
            TestCase(algorithm([3, -2, -5, 0, 1]),[-5, -2, 0, 1, 3]),
            TestCase(algorithm([4, -1, 2, -6, 0, 7]),[-6, -1, 0, 2, 4, 7]),
            TestCase(algorithm([1000000, 500, 100, 100000, 2500]),[100, 500, 2500, 100000, 1000000]),
            TestCase(algorithm([]),[]),
            TestCase(algorithm([0, 0, 0, 0, 0]),[0, 0, 0, 0, 0]),
            TestCase(algorithm([1, 100, 2, 99, 3, 98]),[1, 2, 3, 98, 99, 100]),
            TestCase(algorithm([9, 4, 6, 2, 7, 5, 3, 8, 1]),[1, 2, 3, 4, 5, 6, 7, 8, 9]),
            TestCase(algorithm(list(range(100,0,-1))),list(range(1,101))),
            TestCase(algorithm([4, 2, 4, 3, 4, 1, 2]),[1, 2, 2, 3, 4, 4, 4]),
            TestCase(algorithm(list(range(10**3*5,0,-1))),list(range(1,10**3*5+1))),
        ]
    
    def selection_sort_test(s):
        s.test2(s.sort_data(s.selection_sort))
    def selection_sort(s,l:List[int])->List[int]:
        for i in range(len(l)):
            for j in range(i+1,len(l)):
                if l[j] < l[i]: l[i], l[j] = l[j], l[i]
        return l
    
    def merge_sort_test(s):
        s.test2(s.sort_data(s.merge_sort))
    def merge_sort(s,l:List[int])->List[int]:
        def m(l:List[int],r:List[int])->List[int]: #merge(left,right)
            nn = [] #new nums
            li = ri = 0 # left and right index
            while li<len(l) and ri<len(r):
                if l[li] <= r[ri]:
                    nn.append(l[li])
                    li+=1
                else:
                    nn.append(r[ri])
                    ri+=1
            while li<len(l):
                nn.append(l[li])
                li+=1
            while ri<len(r):
                nn.append(r[ri])
                ri+=1
            return nn
        def d(n: List[int]) -> List[int]: #divide(nums)
            if len(n) <= 1: return n
            mid = len(n)//2
            return m(d(n[:mid]),d(n[mid:]))
        return d(l)
    
    def linked_list_delete_duplicates_test(s):
        ctll = ListNode._static_init_list #convert to linked list
        tm = s.linked_list_delete_duplicates #testing method
        s.test2([
            TestCase(tm(ctll([1,2,3,3,4,4,5])),ctll([1,2,5])),
            TestCase(tm(ctll([1,1,1,2,3])),ctll([2,3])),
            TestCase(tm(ctll([])),ctll([])),
            TestCase(tm(ctll([2])),ctll([2])),
            TestCase(tm(ctll([2,2,3])),ctll([3])),
            TestCase(tm(ctll([3,5,5])),ctll([3])),
        ])
    def linked_list_delete_duplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
        '''by chatgpt'''
        if not head:
            return None

        # Dummy node to handle edge cases easily
        new_head = ListNode(0)
        new_head.next = head
        prev = new_head  # The last node before the sequence of duplicates

        while head:
            # If we are at the start of a sequence of duplicates
            if head.next and head.val == head.next.val:
                # Move head to the end of the sequence of duplicates
                while head.next and head.val == head.next.val:
                    head = head.next
                # Skip all duplicates
                prev.next = head.next
            else:
                # No duplicates detected, move prev
                prev = prev.next
            
            # Move head forward
            head = head.next

        return new_head.next

    def max_sub_array_test(s):
        tm = s.max_sub_array #testing method
        s.test2([
            TestCase(tm([-2,1,-3,4,-1,2,1,-5,4]),6),
            TestCase(tm([1]),1),
            TestCase(tm([5,4,-1,7,8]),23),
            TestCase(tm([-32,-54,-36,62,20,76,-1,-86,-13,38,-58,-77,17,38,-17,43,32,-88,-19,-70,95,0,-64,75,15,-87,-26,69,-95,-65,-18,-28,-43,22,-88,54,-25,-13,67,61,-74,-91,60,42,24,-80,-15,-44,-91,42,-38,-96,-58,-3,55,33,-13,-71,2,-9,-60,60,39,-26,-41,50,-72,33,-62,94,-28,-37,79,-68,81,3,-71,-57,35,-63,61,96,-83,-97,-29,48,35,57,76,-86,-52,92,50,86,-34,85,14,-30,18,51,-36,66,90,-79,75,48,0,-96,67,-64,-83,28,-91,-90,30,-44,57,-58,-87,10,-68,-63,-21,81,-76,45,66,14,-85,-39,-58,-44,-95,-68,-47,79,56,52,59,23,64,75,-49,50,61,57,-94,-5,98,95,81,-70,-68,-40,87,-68,-95,30,45,96,90,86,-71,94,94,-19,50,27,-90,9,-50,51,-39,-23,-22,-78,-66,-17,-7,-68,-22,-26,-62,-13,34,-75,18,38,54,-36,11,22,-73,39,-7,98,96,-56,25,83,52,75,34,-86,-48,-88,-88,-14,-29,5,-6,26,78,9,-87,12,10,30,-72,-58,70,15,63,97,-68,-67,95,-72,-24,20,-89,-94,-28,21,-81,1,32,-93,63,80,11,-43,6,-33,42,18,78,-47,-75,82,-6,95,-25,-66,69,6,-57,41,10,19,-62,21,1,10,-81,19,-89,28,2,73,8,-86,-93,-86,-20,49,8,-65,78,32,94,-51,27,-31,-41,-27,51,1,-86,-39,96,-48,58,-3,38,77,92,25,5,-5,-25,89,-15,-42,79,41,83,-13,52,61,-81,23,86,23,68,-55,72,19,23,86,80,19,-85,38,93,29,-8,85,-46,73,-43,5,62,41,62,41,-41,23,-72,-88,-39,-76,34,-52,23,-20,-31,-5,98,91,-19,78,-12,-28,-6,-19,-99,85,-34,-69,59,0,12,-2,-82,-25,-60,-23,74,-56,-35,-65,-33,75,-18,89,-45,51,-38,-46,19,42,-91,-93,91,-21,-13,91,-35,30,99,-99,-70,11,-2,-53,62,14,0,36,58,64,48,-98,40,-70,68,71,57,-70,-75,-23,48,-89,-17,39,-11,70,8,30,-23,-16,7,6,94,82,29,34,-4,-70,-53,-69,70,94,-67,-13,-98,77,-41,58,-93,-40,-88,31,-30,-5,-29,36,-58,55,-34,18,-84,73,-99,86,32,29,20,-72,35,67,-64,6,38,-55,92,39,-78,-72,-2,-95,-12,9,35,34,80,82,-30,-78,14,13,16,29,-37,16,16,94,-54,-87,98,57,56,-66,-37,-5,-22,-44,-66,-24,-17,8,-20,47,94,92,-18,51,74,28,50,-11,-59,-34,94,-20,59,10,-26,-95,23,-27,61,-21,-17,-98,50,38,-66,84,-86,-7,-31,-6,-59,-60,-14,22,91,-63,-73,41,2,-32,83,-3,47,42,83,98,23,6,-52,-38,62,30,-37,12,-32,-4,-27,-18,88,19,52,-94,58,-85,4,26,-72,31,-56,30,75,-72,-73,-1,69,-90,-3,-30,-7,44,31,-68,-48,70,20,19,-57,93,77,-92,12,29,-86,-53,20,17,73,48,-75,-83,-22,76,-79,-19,-24,67,-33,49,-63,36,-29,44,67,22,14,-12,-59,56,-42,-81,40,46,24,53,92,-55,-52,42,92,-51,36,-53,-74,56,4,1,0,70,-73,36,7,-3,-43,-49,95,70,38,-63,3,95,-68,-56,41,32,73,11,76,-79,-47,45,-53,65,68,-28,-1,-28,49,98,-80,75,12,26,-50,68,76,-55,16,-8,-42,-81,-36,-34,-61,-94,98,-87,-7,51,-90,22,-26,-44,-12,-58,4,63,-9,-47,61,10,-94,-50,-87,-68,95,65,-24,11,-43,11,6,-2,76,45,3,74,34,95,26,43,-6,76,76,81,94,-20,44,-15,10,-17,71,-8,9,83,23,4,26,75,-85,59,-37,-2,-43,-60,-57,36,90,53,8,-7,-27,-97,-31,-51,83,-36,6,5,25,92,64,-3,-39,-27,-43,60,77,82,12,68,19,75,-34,75,-85,-15,-11,-95,-62,96,-2,11,98,66,36,59,-93,-81,-59,32,87,-95,-71,-52,-23,-38,-92,-69,-78,20,99,40,-5,-58,-8,-14,27,80,-10,41,77,64,-71,52,8,42,11,14,60,28,-77,48,32,-72,72,86,-10,80,93,11,-23,69,-72,48,-88,19,-89,15,-23,-23,-67,-46,-58,-38,82,26,-96,-29,-83,40,98,-60,-12,31,-33,-62,-6,33,94,-13,-79,-29,-43,-52,95,-55,44,-94,36,-79,-18,68,-49,0,-70,-67,-74,-90,3,-80,50,-21,-41,-85,86,2,-48,-20,-64,-54,43,-44,-7,76,-19,-35,-79,-75,-53,33,-78,6,2,-28,-94,8,-19,-91,18,61,-72,-55,-60,-37,-41,-74,64,-12,-18,76,10,-75,-67,80,-98,-10,-55,98,14,-31,32,35,74,-89,83,56,18,-36,64,-87,-75,68,-43,-59,-69,-7,-57,95,57,24,71,-33,49,80,-53,27,-30,-31,2,60,14,-89,-28,-12,-79,-45,-56,39,-4,-91,28,-3,76,85,38,84,-45,80,79,-88,5,27,-22,-69,-15,2,34,31,47,-64,-8,-39,-53,40,48,-56,95,68,82,-32,76,-12,64,30,-27,-21,14,4,-40,-30,15,41,73,92,48,-42,-29,-18,66,-82,-6,-65,-22,17,74,-97,61,46,71,20,-86,24,-92,55,-69,-43,-66,21,60,-7,67,-25,-89,41,-55,58,75,15,-83,18,10,11,52,64,-72,27,67,65,-50,15,-14,-59,-61,-7,95,-53,50,-71,67,10,-1,35,85,-14,-47,-70,-78,-95,21,-62,23,-69,25,-25,95,30,78,62,-5,-94,-46,80,-54,-8,-26,-59,15,-99,-54,-17,10,21,94,-28,-92,-53,1,-71,-49,99,-34,50,-70,68,-74,-75,-2,80,63,92,85,-83,73,-69,-14,-76,-52,-99,-76,-29,-40,33,91,-46,-94,98,-23,-17,-96,5,-18,-53,-45,-11,-85,-43,13,12,-62,-46,-19,99,-53,-46,-92,9,-46,-45,10,-23,2,46,87,-6,77,69,-31,-46,-71,-27,35,-12,19,90,76,10,-76,-34,-78,-62,19,2,-62,43,33,-55,-48,86,-24,38,-59,55,-15,-72,-74,-38,73,-6,-8,-98,43,-72,-11,-61,94,-35,48,94,31,46,-67,-73,-74,-53,46,58,-32,98,21,43,36,38,-25,97,65,0,-64,38,94,5,17,14,9,-94,53,3,-52,-22,-26,55,-75,6,58,-73,52,5,84,-80,3,-18,40,-84,-79,-85,-86,62,15,-74,-22,-14,30,94,99,-84,76,29,-5,-76,84,69,55,-91,-25,14,-88,-95,-81,72,-76,21,-46,40,36,-49,32,26,-86,-53,29,-32,9,-64,-61,-15,-48,-85,-85,46,-84,98,15,70,83,67,-16,94,71,2,66,71,-77,96,12,36,-52,-56,-61,-62,-33,44,-18,52,80,20,-63,-68,34,-72,54,49,-97,-30,20,62,-86,80,33,84,-17,-24,55,-18,-27,-56,94,-3,-36,-66,11,7,77,92,-41,34,12,-27,65,-77,-23,-3,72,-20,43,-31,-81,56,-51,52,40,8,-72,-28,90,-23,-7,84,50,33,-82,-62,-59,72,-93,-24,6,18,48,-51,18,-74,45,-10,-18,-12,35,0,-79,83,-71,37,-31,-66,-91,-41,-13,-22,-57,-62,-88,-62,-48,-71,-90,-41,4,-8,53,-70,40,-29,-68,-38,37,13,26,-28,89,46,-67,-81,-17,1,-48,-31,36,38,46,78,52,34,-84,81,62,-98,-60,-56,-29,69,-27,-89,-82,4,-50,54,93,75,-96,82,22,-64,-99,-95,13,-71,-50,-51,67,96,-96,-4,30,96,-23,-30,74,-7,13,-55,-37,85,-67,-20,-34,-18,-66,-41,-43,-63,-82,55,-51,-81,37,38,-53,86,63,90,59,67,-13,-33,63,40,36,37,-66,-51,82,-4,-89,91,52,-24,-27,85,11,6,99,28,-61,-75,23,75,40,47,38,3,37,-2,-30,-99,64,-67,41,-23,47,74,2,6,-53,12,-3,98,64,69,60,-48,-48,36,79,-10,61,-20,-58,-22,-73,80,80,63,-45,-49,41,95,-40,-18,-28,-93,-67,73,12,-21,-38,-91,-46,-97,54,-86,-46,-93,50,-67,-27,-12,-11,15,65,15,-28,22,-44,3,72,96,-2,-90,-45,46,92,-36,-4,-18,19,57,-33,72,59,-78,62,-87,-95,-11,-78,-46,76,11,68,18,-74,-83,-82,-42,19,89,-46,-83,75,-15,62,68,48,-42,-50,-33,91,16,-84,50,14,78,39,-5,-33,38,49,-79,-51,-6,15,-49,11,32,9,-70,-1,39,-53,-26,-75,85,-81,49,-79,-55,-83,88,38,-91,39,-48,63,-45,24,-69,92,-27,-72,-82,66,42,-54,-23,-48,31,7,50,-30,30,-99,-29,-8,-80,-79,-11,-36,-86,-22,78,-78,93,-69,85,-52,54,92,-82,-96,-80,-88,69,38,-43,-76,89,87,7,16,-65,-63,17,5,5,13,25,94,53,15,-28,32,37,-58,39,-1,-34,-30,-9,-17,-27,-13,-28,-81,-98,28,-58,68,-7,48,-16,27,61,-99,-91,-56,-10,34,-62,-56,-51,-14,52,62,27,-9,-39,92,60,28,-47,-90,91,24,-72,93,-71,-53,38,21,94,-78,-75,33,99,-66,76,-11,67,90,9,92,-24,-39,32,79,51,69,71,-12,-25,-99,-3,-34,24,25,-64,53,48,73,-49,19,72,-24,-48,71,9,-95,-63,-47,94,46,-78,-30,-93,53,-74,35,0,74,-77,74,74,20,-83,-24,-78,-70,-94,46,79,-44,-35,51,8,17,22,93,21,59,-77,16,-18,-55,62,-12,74,88,99,-26,62,-77,-75,13,19,-58,88,17,-29,93,-37,-50,-50,27,78,57,-56,0,28,42,-64,50,-65,17,94,73,82,46,61,81,96,0,80,98,-86,-1,-60,78,92,-13,72,-45,13,98,82,-9,32,-97,68,-40,21,-96,-89,55,-2,81,29,-43,4,-33,-85,1,44,95,-1,34,-29,15,12,-36,-98,-39,18,-9,-41,-23,82,91,-43,50,-72,77,30,-62,10,-95,-80,-84,-39,23,-41,-47,-99,-97,-75,-24,13,-5,90,-74,35,68,-37,-69,-40,22,-16,-58,-10,40,68,17,-5,-2,32,-95,-20,51,-80,40,-49,-45,69,-71,-42,93,4,47,-34,-5,49,-99,-37,-87,-92,-1,11,-8,17,-99,-68,-15,-6,26,60,-74,7,-60,76,26,56,-95,81,26,-67,-84,19,13,39,61,-92,-11,62,-52,1,69,23,-88,-39,-59,-88,-30,25,-18,95,-15,8,3,24,-39,-94,-42,-35,63,83,-26,55,-21,87,-5,-59,71,83,79,95,-39,48,19,-28,86,59,-40,-67,-16,41,-95,46,-74,-16,-53,-13,-33,4,51,-70,-13,-97,62,43,66,-66,83,37,93,39,10,-46,-36,-71,2,-50,87,38,-41,-52,-44,-36,70,-18,47,-82,68,-86,21,-4,-79,85,-2,59,28,-59,92,-12,78,62,-73,-35,-84,-10,-7,17,17,-43,-68,75,4,-13,-84,-25,68,40,-31,36,30,89,-67,50,51,7,-14,-44,48,-45,43,3,-83,46,67,-68,-86,-40,25,7,-7,56,-41,73,-79,-25,-52,88,-85,-7,25,44,-17,34,-28,-89,-82,34,-33,65,89,-90,-55,-94,33,-88,-85,46,-52,16,-47,40,-50,11,14,-30,62,38,-42,53,-68,59,-25,-9,-30,23,1,-12,-43,67,-70,-54,54,-25,-71,-13,-14,-57,-90,11,58,-61,-72,-92,49,41,-23,-11,-43,12,-58,-12,-52,16,55,18,-61,-44,-94,-27,-99,-65,18,54,-14,23,-82,49,65,-73,-40,1,-35,-13,-15,90,-94,39,-21,-38,-49,-2,26,-2,-86,-42,92,-70,13,74,2,-10,86,96,-56,-28,20,-39,-79,-38,-36,-20,62,28,44,-52,-4,49,63,74,87,-86,-51,-10,-12,-60,47,-19,-31,-63,-68,47,27,-82,43,47,-10,-60,-91,10,2,-28,67,-59,77,-12,-35,-27,-86,-72,-76,-99,-82,-27,-33,6,12,90,63,57,-73,94,4,30,-87,-76,-22,-21,-36,-37,88,42,11,32,82,-35,-80,-52,37,-67,-25,38,-90,-7,87,-47,75,-1,-57,15,-67,-53,9,36,53,-2,36,-69,76,99,-30,42,18,58,-26,1,-77,-29,48,37,-20,99,-25,-10,-8,-38,42,-56,37,62,58,69,-91,-32,-18,61,42,94,69,-4,94,-84,-62,-11,73,-12,89,73,35,14,10,-84,14,61,-18,82,99,-99,-73,36,62,84,-18,47,-71,40,9,71,34,-45,-56,5,-30,-42,94,-79,45,60,93,-42,-25,-20,-27,88,-59,54,70,-60,54,96,52,-6,58,-89,-59,86,27,26,34,-39,80,77,43,-72,-65,37,-75,57,-3,-82,-85,71,-3,-36,36,14,94,-16,30,48,-20,-18,-81,-85,-31,-40,78,95,85,12,-67,43,-33,75,47,-22,-11,48,-88,86,65,-74,34,-61,-34,-52,29,-40,31,-41,85,87,40,-96,-97,85,-60,80,80,-97,-31,-87,45,35,65,-31,90,-46,-83,-98,-83,58,-95,28,73,-30,75,2,7,-17,60,-8,-30,-23,-28,-51,-38,12,-94,-81,14,-25,-92,-64,86,72,4,53,-73,-3,55,-57,-68,59,70,-95,-94,22,-17,-11,-95,-57,80,-48,-4,-47,0,57,64,6,52,-45,57,59,-10,43,32,70,-3,35,-33,-71,77,-2,64,25,-21,46,-76,61,-64,-94,80,92,-43,-24,-55,56,-90,85,-61,61,-83,95,20,6,16,-71,-24,89,63,19,94,18,93,58,-80,-28,4,-57,-90,16,-52,89,9,4,-58,30,-62,-49,-7,-24,-88,9,-51,9,-84,41,37,-32,30,1,87,24,-81,80,-41,-62,-70,39,-43,-61,-44,-18,-94,41,85,-53,-29,99,-25,63,-47,-14,-27,1,94,-35,-58,9,32,48,-90,19,-51,-95,-23,-16,-81,-17,-77,-48,-2,-45,33,3,95,-4,27,-33,-5,-98,-93,47,-36,-44,-52,-64,20,-34,44,-70,90,30,25,-61,11,-98,-1,6,-16,22,57,82,76,-9,-38,49,63,65,92,-42,43,98,-94,-92,-45,-70,-57,51,-27,63,80,-37,-7,6,-21,80,-16,77,-14,45,-24,-80,27,-70,-13,65,78,-50,8,47,8,-49,45,-10,-42,76,19,-23,5,68,-83,-15,-91,-14,67,-13,-34,-48,-59,51,96,-6,48,0,23,-65,-58,-22,-15,49,-75,92,-99,46,-41,-65,0,77,-11,-95,23,5,-11,31,91,-66,-82,-43,-15,-65,85,-42,28,33,57,-72,44,98,-18,-71,48,-17,97,25,-70,-44,59,6,-89,-52,10,33,53,75,-36,44,-91,-42,77,-30,-8,39,27,97,-51,-16,-98,93,-40,-16,-78,84,66,-80,9,72,-48,-32,-45,-38,-7,64,94,-77,17,-42,43,25,-8,20,72,84,-63,-1,58,-15,60,59,-22,-80,-80,0,80,63,-4,-11,35,47,33,90,-90,-97,31,80,-75,-52,-84,67,50,7,-35,-1,-32,1,97,-74,85,34,62,-59,30,81,40,10,44,-87,98,-43,36,9,-76,45,11,54,3,36,79,18,-20,6,-98,44,-95,-30,-78,78,-28,84,13,-66,-75,-57,92,41,30,36,53,5,69,-11,-86,92,-88,2,24,-9,-85,-20,-91,-6,85,-90,-84,66,-44,-86,44,28,97,-66,61,98,-23,53,39,83,-33,-31,-12,-87,35,2,82,46,80,6,-63,94,62,-78,-12,24,-92,-20,-10,-36,-6,-89,68,-32,-55,29,-33,-2,-40,82,-19,3,-48,68,15,86,-53,-3,9,-73,79,22,-79,41,43,-14,-35,27,42,31,67,-64,42,35,-19,86,-58,47,-16,1,29,64,80,57,9,-5,43,56,69,-71,82,48,50,-96,-33,-30,-34,8,97,-92,-83,41,-79,58,-23,1,21,18,-75,81,95,30,22,75,-12,-69,-53,-92,86,-84,-63,46,41,86,26,84,-66,91,-31,-69,98,-15,48,95,-80,-75,-4,40,42,-3,-2,14,-72,96,-11,15,-72,36,-1,90,28,35,36,-54,-1,-61,-93,32,-70,-24,39,-94,60,-13,-99,57,-11,-26,-26,-92,70,72,22,97,68,-89,-11,72,23,88,63,-71,0,-24,-26,75,15,-42,-16,21,33,-77,26,-30,-13,4,-73,75,77,1,82,47,-50,4,22,-82,-8,-89,67,91,75,30,-80,-47,82,-29,-72,-26,28,12,95,-62,11,-2,7,97,-98,11,49,55,88,9,3,-62,89,-75,-68,-19,-88,98,-27,87,5,-31,-60,-35,39,-56,38,-56,55,10,-19,-33,85,88,-59,86,-24,90,19,-36,-24,-78,-22,-35,-77,9,23,-66,-15,72,97,66,-59,-86,-69,80,-42,46,-76,89,-67,-18,-67,-82,-53,-26,81,22,-59,0,62,16,21,-60,57,20,25,-20,30,86,-48,5,53,-7,18,60,-50,52,6,-50,19,39,31,51,33,77,-75,14,-24,-58,-86,-62,34,11,-23,-8,-69,79,71,38,65,-99,-57,18,-7,37,-21,19,-33,61,45,85,-23,-23,36,-89,-69,38,24,7,79,38,21,14,-74,75,-94,-66,54,53,71,19,53,-86,-85,-77,27,-30,18,93,-92,-36,78,-15,17,-8,-5,47,-70,95,-69,85,10,52,-24,36,27,-42,69,-42,-88,40,53,41,30,44,64,-66,-86,-18,-72,98,22,82,-17,-61,-49,54,63,-43,49,93,-57,-40,22,-5,72,-74,53,41,-40,64,-42,-86,5,64,57,-54,74,48,5,78,46,-73,38,-94,-34,88,59,-71,-54,-15,98,-35,21,-2,59,-7,0,12,-89,60,-47,-55,73,35,85,7,80,60,-68,-15,38,54,-11,-47,36,54,-58,-28,59,-36,57,34,28,-45,-68,-36,25,32,53,12,-8,-94,34,-58,17,19,25,-2,-44,34,59,71,-12,-52,-76,-75,78,-58,-4,37,5,-70,-28,-90,-15,3,-26,86,-88,-96,-2,-20,85,-68,21,3,28,-76,-99,60,57,36,31,-78,61,32,-54,39,73,19,-23,-44,25,25,-35,86,-95,15,72,-83,-5,-29,-27,-19,-21,94,83,83,94,60,44,51,-4,-48,72,-43,60,95,73,34,-9,-50,66,-84,51,-92,-97,55,0,51,48,94,98,-78,-25,-45,92,-42,-62,86,94,81,14,-32,-66,-14,1,93,-42,-26,4,48,0,-53,41,-49,-45,20,-93,54,-29,31,48,-53,52,-99,1,44,-65,38,7,-70,-3,-79,-3,7,-16,97,0,41,-28,81,-33,71,-72,-92,99,-41,4,82,-87,74,13,38,97,-57,38,-2,63,49,-86,-29,78,10,-32,-48,-83,51,-50,-7,-30,-2,-49,13,-54,-44,-3,-78,14,-99,-96,4,51,17,-81,49,59,33,-76,22,-18,-63,69,-39,23,-62,-11,16,65,15,86,-88,-87,-63,24,-64,-31,-79,-43,60,-79,-39,-36,49,54,-41,-2,-10,91,21,-88,-50,35,58,-13,-65,95,76,51,37,-9,37,49,-19,50,-50,-7,-80,-29,49,79,-32,86,-80,-83,17,-45,-9,-93,-54,89,95,-5,1,53,82,35,25,35,-37,-38,-74,76,87,6,27,-62,98,23,-15,25,2,52,-12,-2,46,-94,52,-86,88,74,-97,83,-54,3,13,-95,-84,-85,39,-22,-46,65,-45,-59,-52,81,-45,-76,-19,39,25,59,-31,-87,-43,91,94,85,-95,83,59,84,43,81,-13,-66,86,-21,-52,25,57,-22,-33,-88,19,-8,69,50,15,-50,-33,-83,-91,35,-71,-58,3,-76,-73,-16,83,62,-32,3,43,32,36,29,-89,61,-68,67,-61,-25,55,34,66,24,-15,-42,50,51,-26,35,-37,-20,76,42,-20,-97,-74,-37,41,-6,42,-15,-97,-21,90,89,-83,98,57,-45,-26,-11,89,-83,90,-49,-48,40,78,2,-24,-59,81,-48,82,-62,-68,84,-23,49,-45,-80,-89,-43,-25,78,-53,-9,76,80,45,-73,-31,11,-79,-64,61,-51,76,39,50,-71,56,9,-43,15,46,-35,0,-76,-85,54,-57,-98,88,-6,79,-88,84,56,91,6,59,-63,-82,79,-27,55,-72,-74,-28,78,-46,-95,-36,-13,20,-90,52,96,-90,43,27,-71,44,15,22,24,26,6,-43,-6,-87,-84,31,-93,95,80,38,99,82,-90,54,-87,-85,-82,-1,11,-95,-72,7,-86,70,-66,-57,-8,26,64,-84,-48,-29,-51,23,-40,-36,-69,-57,-64,-89,80,-64,92,67,89,81,58,-92,-20,-31,88,8,52,-98,-45,85,-56,46,11,-91,38,40,-44,86,-37,-8,-72,-7,11,-37,-20,-32,-2,-51,-65,-35,-70,69,-51,9,-62,-63,93,89,-62,-74,52,-18,71,-60,-33,-13,79,-1,-50,-81,67,-23,11,-22,40,67,45,-85,-84,57,-21,-55,26,27,31,41,63,24,30,-98,26,-41,-40,-25,-1,3,61,-22,-98,87,-3,45,-35,84,-77,81,52,44,95,67,1,-25,89,-95,78,20,45,19,-79,-48,-3,47,11,56,-78,-90,-64,59,-36,-63,46,37,58,87,-1,80,68,-49,1,64,94,79,-84,83,83,70,-20,28,-11,-22,79,85,-98,-33,18,99,-47,53,58,17,66,-19,-69,24,-31,29,-19,-86,79,82,54,51,61,46,11,44,17,68,-51,82,-54,-95,68,-53,-28,-37,-77,24,-83,80,-82,59,-62,-51,-16,-17,54,64,-3,11,23,-49,-38,-39,74,49,81,-9,17,-70,-50,62,-65,94,85,-17,-42,-92,83,-49,64,78,-90,-97,-96,-7,-38,57,-65,58,-32,34,86,6,95,37,-44,76,-95,72,-93,-46,35,-82,-51,-3,-23,-17,5,-63,33,-31,-85,19,-52,94,89,-13,52,-76,-78,96,34,84,3,29,21,58,-17,25,7,-34,79,-58,-40,-95,16,36,-36,-79,49,96,-34,64,15,-9,58,4,77,-13,4,98,-16,-84,60,86,44,-19,21,-73,-93,-72,-8,62,-53,51,66,62,-36,-70,59,-86,-97,-98,54,-6,-31,-11,75,-54,53,-44,-78,-63,-29,81,99,92,-61,-3,95,-55,24,-35,-93,47,92,49,-14,-43,55,-55,-53,34,-76,77,-94,-8,-57,80,38,95,-87,36,-91,-16,-6,84,-47,32,-19,-52,54,82,12,37,29,5,-13,-84,38,-81,-63,84,53,-40,38,35,52,80,15,-33,53,27,79,61,87,-50,46,-60,58,-96,-35,12,85,76,-50,-8,-19,13,-16,95,8,20,79,-62,57,-6,72,-14,-25,-13,29,-72,91,-15,65,55,34,-12,72,-8,-8,-63,-95,-23,-87,-69,45,-29,20,-71,-34,-72,26,-77,-34,-17,16,-62,45,67,2,-26,71,-30,58,37,-75,-31,-98,96,60,-7,-90,41,47,-77,48,-8,-30,-32,21,12,-4,-76,34,61,6,-72,75,-72,71,-23,-22,43,46,-64,-43,48,-19,-42,44,-82,51,54,59,-25,53,-16,66,99,52,-36,11,-52,-13,-77,85,69,-50,60,96,-79,-85,-26,40,37,-14,96,-15,66,-68,-70,84,59,60,20,-66,-10,-96,76,-11,32,-59,99,56,27,-78,41,-3,-52,-21,-30,-54,-30,21,85,-93,-93,82,-8,50,90,-2,11,49,-42,-92,82,24,87,58,-87,20,75,88,-24,3,87,94,76,-65,-50,-77,-20,19,43,-34,25,27,-75,-83,-23,-85,14,64,40,48,71,-1,-28,58,-66,-16,-45,-90,-50,31,-11,-63,2,-58,-29,51,-36,51,-30,-92,93,94,11,94,88,87,-91,2,28,48,-49,-1,-53,-1,-66,-43,60,-34,-34,-90,96,54,-54,-25,72,-7,-74,-63,-79,-5,-79,-86,-34,31,-92,-46,-5,92,33,-77,17,60,-2,40,-41,8,97,-81,73,-37,-72,46,94,-50,-79,66,43,-77,79,63,-6,76,54,60,-16,61,90,78,54,23,77,-52,-17,-48,88,19,60,85,37,10,-75,42,-66,-81,68,-46,62,11,-46,41,-48,47,-5,-94,-16,78,43,-25,34,-26,74,11,22,-42,40,10,53,-23,-5,-10,86,19,-91,-80,15,77,-49,-23,-35,4,94,16,-72,-11,-2,12,-32,-58,86,78,15,60,-34,-86,94,-94,23,47,-18,94,-85,-54,-9,-77,-58,-94,99,-31,-41,-58,49,-70,57,77,19,-44,-11,63,-3,-25,18,88,-88,83,-97,83,89,2,7,-52,-3,21,-30,64,21,-11,46,-3,57,81,38,-16,-89,-4,-39,6,28,49,-54,-98,-99,63,89,88,47,68,48,36,-52,-44,60,44,76,-69,-15,-26,19,-69,71,-47,-88,9,36,98,-95,73,5,9,22,50,10,-1,-9,-23,-35,-62,-55,-87,50,-8,-32,10,-87,-78,17,-26,-5,-64,4,-34,-34,-8,75,78,-32,56,52,72,-57,-26,-1,52,50,89,-71,-85,-96,-49,-73,53,19,-28,-36,32,-7,80,82,64,93,-36,-69,-64,55,82,-86,99,15,65,71,57,-83,-52,10,43,-63,15,57,-59,65,60,70,-15,-68,-66,93,1,91,75,-34,61,-61,-27,73,71,-68,86,70,46,29,19,-19,45,66,67,-12,79,-18,-78,96,-75,58,67,-14,67,77,-21,-32,45,-69,10,-17,-53,59,-44,-82,90,-57,87,14,48,-17,71,-30,-51,38,34,-94,96,55,-98,-79,-10,68,-17,-43,23,38,2,-32,69,88,51,-84,48,83,-90,15,-74,73,-71,50,-43,99,-80,5,14,53,-13,11,-15,-11,8,75,-66,91,-68,56,29,10,-98,-25,-1,29,-33,-53,12,-24,38,15,-50,67,-35,6,43,-38,87,-42,-85,-25,45,-23,63,-70,-48,73,-79,-40,7,27,70,8,78,-31,-86,-54,-8,25,21,-70,-83,70,73,-18,-24,-6,20,-36,28,34,15,-50,-89,-45,-21,-61,-72,-23,-1,34,-96,-54,-81,58,14,-91,-96,82,34,25,88,27,-28,39,85,25,-67,5,88,37,-60,-20,86,-73,34,-35,-34,61,41,41,-27,21,86,-32,-20,-23,-23,-40,-41,87,61,-76,14,-66,62,0,58,71,81,23,8,-79,-97,-6,-75,36,36,-10,-26,-23,-92,23,75,93,91,-45,-30,-55,-9,5,-91,-47,5,-1,-14,-32,75,-79,-84,57,43,1,54,22,71,78,-65,7,-55,8,61,-48,-91,36,-55,76,67,-9,-79,-42,95,-71,10,1,27,72,45,-20,69,37,-63,12,-62,90,-89,-90,69,-55,-7,-86,30,-46,-57,38,89,-13,14,56,55,11,91,27,16,-22,-72,-80,-49,49,-1,-80,-13,35,9,1,26,19,10,-28,41,-96,-38,-52,-66,4,85,-76,68,-24,56,23,63,24,-50,-21,-97,-45,-2,-47,80,74,-51,-56,9,34,45,-88,-69,-68,-17,71,-88,45,-81,-54,-74,80,-54,93,-67,78,-7,95,3,20,-49,5,74,25,34,31,-1,82,75,85,17,-3,96,24,28,-43,-5,17,78,-10,62,-96,-53,-92,74,-21,85,-33,-49,-12,86,-22,69,37,79,3,-31,-21,-14,-79,-59,79,17,-86,3,-55,-30,74,38,-75,-36,0,-71,-13,-16,-97,65,-53,-54,92,11,-67,-30,-19,69,-51,83,-85,4,-54,-65,44,-98,51,35,-19,73,81,-68,11,-93,-28,88,-88,58,-28,90,23,-5,-64,-8,-93,-32,61,-13,-85,87,-53,-71,-9,-30,-59,-87,-29,91,24,29,41,5,60,-70,-12,-67,-82,-1,90,-34,-11,-9,61,2,82,67,69,44,-69,60,8,-23,66,98,46,-93,87,94,97,11,23,16,93,-40,45,-19,-8,62,57,-40,-72,-54,50,65,24,32,32,-29,-47,39,-69,60,-83,96,36,39,-97,0,10,77,-88,-90,-7,81,-30,-62,-38,38,76,-81,97,80,-58,47,46,65,-43,-45,-64,9,94,-56,-53,87,-60,59,3,-80,-41,-86,96,-53,99,-34,-72,-31,-20,-33,84,-67,-38,81,-87,3,6,58,-54,62,-86,-42,-52,84,-22,-6,71,-5,-70,-26,14,-11,-36,-13,35,-36,29,40,9,-91,-93,-7,-59,-32,-48,-70,48,57,88,70,19,78,-95,43,39,-17,-85,10,77,-56,60,-32,32,-75,54,-55,-35,-17,84,50,-32,67,-79,-15,-88,-28,14,59,-71,-21,6,24,-43,11,-56,95,70,-42,-18,47,2,-57,-85,-66,-56,45,-45,8,-95,-61,-65,-28,82,-45,56,94,26,-30,53,31,49,-63,-68,82,24,-24,55,-6,33,-86,-82,-65,-44,-68,-54,-24,53,-23,60,57,-7,95,-71,-25,-50,61,-31,53,-92,99,-16,56,-64,-7,16,59,68,-52,-70,-22,-39,-53,-10,-6,-45,12,69,-92,88,7,41,-19,2,46,56,28,7,-98,-42,14,77,19,-52,-87,-88,40,48,56,87,54,-66,-75,-99,23,18,-45,-88,-35,-61,-99,71,79,-42,-50,25,90,54,-91,92,-87,23,69,31,47,59,42,-13,-16,-2,-26,-61,8,-2,-84,-91,93,-52,-80,-65,-14,96,-94,-57,-45,-68,67,-55,85,76,-86,74,-24,60,5,-77,-4,-76,9,56,-2,59,71,-93,-66,87,91,-73,-88,-89,-62,-3,-92,-57,39,-61,-49,-16,-40,-86,-63,-49,87,12,-89,-30,11,-17,-7,-80,-84,91,55,87,74,88,51,65,92,39,75,-70,-64,59,-50,52,97,-23,-64,-65,-10,-28,-15,-22,60,-28,47,-29,31,-83,-33,46,-92,-78,10,58,86,-62,23,55,76,-24,62,89,-65,-88,41,-90,-35,53,-56,55,-98,5,-90,61,76,33,-91,7,49,-25,-70,34,-27,40,-8,-41,54,-84,91,31,-9,53,20,2,41,-62,-12,-94,90,31,60,91,-64,46,29,88,-21,37,-5,-71,-88,2,39,-16,-58,-69,19,95,-77,10,3,13,-37,99,92,-96,-63,-20,86,-73,88,23,-82,-99,-30,23,89,48,-40,-39,53,-29,62,-31,31,80,0,-49,53,22,37,-44,-87,77,-44,81,57,-8,-39,43,-4,-74,-33,-10,-73,13,-86,-84,-62,-27,-47,91,21,92,-40,-48,72,36,-21,25,-64,93,58,-75,70,90,-94,27,-41,-57,-52,54,-31,-9,-55,-5,-96,-42,-13,-59,7,39,-91,-72,8,-54,-43,-19,81,12,82,93,5,40,-82,-48,7,99,79,66,-57,-96,20,-12,-6,41,82,-3,75,-54,37,-17,61,46,-13,-30,91,20,50,49,-67,9,43,37,-73,37,66,33,37,45,-1,79,25,96,-33,-81,14,-74,-8,-10,-52,-71,49,-91,51,-87,-22,19,33,-95,-54,65,14,88,-21,17,25,-55,-50,-61,66,26,94,-9,-1,-62,-14,89,-37,77,-21,10,-93,-95,-4,34,94,73,-69,27,55,-24,-8,-54,-36,47,62,65,69,88,-95,35,-9,98,-96,66,-63,-11,56,-24,-33,-88,63,-50,-83,-41,60,-89,-91,90,14,63,-34,82,-91,6,29,47,-28,-2,36,-47,-89,-96,-72,-86,70,-59,2,26,17,45,37,-43,71,30,15,32,17,23,22,-69,-13,65,-87,-28,-29,19,-80,42,93,32,71,-96,35,99,-83,5,-60,-4,-69,33,-82,45,66,89,-48,-19,-2,69,81,96,76,-55,61,66,16,-67,-15,12,51,-45,-56,23,-41,56,22,-48,-39,38,24,-31,-51,42,-86,-85,8,-35,-27,82,-89,-69,78,87,-25,40,30,67,72,91,-43,23,-54,0,-77,-19,-44,21,33,-6,-63,-66,62,-15,52,52,-24,-40,93,25,-58,81,55,97,45,-93,37,-25,74,-14,-34,-69,85,-12,7,-91,69,40,29,-21,-89,65,12,72,-73,-58,1,-20,1,-29,-95,20,-48,60,17,96,66,-69,-52,17,16,13,24,-22,78,-91,-14,24,48,92,-97,58,57,-9,7,61,32,84,-59,-89,-45,-55,7,7,81,-99,-20,25,-69,28,-58,23,-82,43,-99,95,51,-36,19,-24,-44,-2,-89,89,66,-6,50,-25,-22,90,-38,9,-87,68,92,93,45,-27,18,52,77,37,-25,94,-43,-47,67,-92,-7,-37,-40,25,38,-30,14,4,63,-35,55,-82,-68,-82,26,-56,85,-4,-85,8,68,9,-40,45,-77,12,16,-21,41,-17,-37,-66,46,22,58,-16,91,-51,64,31,-10,19,25,22,36,-71,42,-1,24,56,-93,-8,-57,-34,-86,-35,77,29,44,18,89,-93,28,35,6,-37,95,-26,12,59,82,-98,-22,7,23,-85,35,-57,-10,-41,75,-4,27,-82,38,17,-17,16,-76,-96,-89,-87,86,16,47,-31,78,19,43,-10,-22,25,-8,55,-68,91,46,44,-66,-87,-97,86,8,7,-96,-76,-76,-37,16,-53,-57,26,-64,-93,-81,82,-25,97,1,-82,-13,55,42,55,-12,51,47,-66,-28,57,-76,-26,43,31,57,24,54,58,63,-53,81,-93,73,-82,-87,91,76,-13,65,-45,-18,-47,-90,-99,-92,-3,51,31,8,-77,-11,31,-26,-68,-61,-92,32,70,65,-4,-83,24,2,66,41,-9,35,17,77,0,71,-64,52,81,-63,37,-45,64,-32,62,64,56,-30,-62,65,9,-78,-26,-44,64,69,-28,-12,-52,-84,-71,38,50,22,-7,-50,-6,28,79,-48,64,16,-17,6,60,-54,-30,-6,-8,7,-64,77,5,9,-67,-31,-44,-18,56,3,-3,61,-58,-53,-16,34,-4,54,-61,-48,-94,80,44,-35,-14,5,10,55,75,79,-61,-89,56,43,-4,89,89,51,-29,45,-45,44,-16,-27,90,67,-93,-37,-2,-78,15,-20,-98,-41,44,-13,40,-68,-58,15,-89,-20,2,43,-99,75,32,89,26,79,11,57,23,94,-70,-10,38,-87,53,-64,-65,-32,-8,-64,3,-63,23,-79,-32,41,35,-45,98,-85,97,98,89,7,64,-8,-37,-25,-50,86,-53,-21,-24,61,-8,-94,74,3,-50,-34,38,53,-21,-62,73,46,56,-15,77,54,-24,52,29,-35,-41,-7,34,21,44,83,-16,90,-61,37,29,7,42,3,-90,92,68,48,22,-52,62,-5,70,18,56,25,-28,-67,-23,77,-26,35,-30,-92,-67,91,67,17,81,-93,-69,10,13,72,89,23,41,58,-52,-37,82,-90,34,29,5,-9,-46,53,23,7,31,96,-81,77,81,-48,68,48,45,27,31,75,-86,-55,47,-96,44,-11,38,68,28,20,54,-37,-74,-41,53,-43,-87,-47,63,20,-74,82,-3,-93,-89,-34,55,55,69,-14,30,82,7,-46,85,-72,19,23,95,-52,-80,50,10,-54,-91,-37,2,97,15,-58,17,40,-99,-85,24,-89,-43,-44,65,-74,18,-28,-92,-74,26,70,-70,-78,-7,-74,-31,13,75,78,35,60,-82,-86,58,9,-45,-25,-73,-44,-34,50,-57,-1,-94,85,1,0,-66,-14,2,36,-44,32,58,48,57,-73,-62,9,82,72,-31,76,85,-73,-14,-82,-21,12,-50,21,-61,-31,97,-79,-46,-2,20,-12,83,-77,-76,-61,54,58,64,11,85,78,96,44,28,-35,20,-10,-31,-94,7,24,-6,33,-55,9,-97,-58,29,-67,17,-50,-3,0,-51,-2,16,-97,-44,-43,-10,-59,-88,-14,84,39,27,-18,-70,95,-36,13,19,58,-76,41,-33,-97,59,96,-65,-24,-77,32,53,-29,-70,-31,-50,-15,2,39,3,13,-98,-36,-70,28,45,-41,1,-90,48,-80,44,48,37,87,50,-3,60,85,49,82,-6,2,-70,0,47,-21,61,-51,94,-36,39,95,5,68,1,-73,-96,-98,-64,28,97,56,-23,12,44,-73,-91,4,88,34,-36,-41,-63,-7,58,60,49,-3,86,43,37,25,16,-58,-30,-83,-31,-27,94,-19,-99,91,-62,-46,-20,58,-42,-12,-38,-54,99,25,5,-87,95,-60,72,44,36,58,-36,-26,-17,79,-8,29,72,37,-98,66,-81,79,35,-67,-67,-85,90,-10,-20,-71,13,-21,31,94,-9,26,-89,-60,-53,24,98,10,74,-42,-10,-34,87,-38,-96,65,-94,98,21,40,-69,54,54,97,-79,-89,-97,10,-11,33,-19,56,-64,-8,96,-41,-84,94,69,66,-71,-64,-68,-84,-26,-88,58,-21,86,-21,-81,-83,-90,50,90,-92,60,93,17,-73,3,98,82,16,-33,55,74,-41,26,43,-75,55,78,-67,47,-70,-55,5,8,-69,61,4,-75,70,-46,91,77,90,61,-28,-83,-35,-53,76,80,13,31,31,71,-65,-25,95,89,-70,-95,13,-41,-51,95,44,-43,56,-52,-42,3,1,-51,-42,-32,87,29,-15,-48,75,-62,-91,-35,68,39,-87,79,90,84,68,97,88,-41,-67,-85,-46,76,-52,86,-76,5,-11,1,30,46,69,17,-48,-69,45,-96,-32,30,-32,-87,-30,-20,92,-63,-36,37,-66,-47,95,-56,-33,-51,-80,-9,-88,-79,95,1,21,2,-76,-33,96,-24,-3,41,-21,41,-28,47,54,18,-96,46,54,-55,-17,65,-3,-44,-91,40,-19,-95,-69,-7,-75,-97,93,23,4,-83,66,0,-31,-59,-81,48,81,-33,-28,35,84,-48,-42,15,-4,18,-20,69,73,-35,-14,-46,-30,-7,46,-29,-5,16,70,76,10,37,-47,78,77,-29,3,35,14,51,-29,-2,80,5,-86,-24,-77,-30,-54,72,35,8,-73,81,-98,49,-48,-4,-34,99,-51,52,13,2,7,-33,49,-89,-98,-36,38,-28,38,18,54,28,71,76,98,-83,49,33,1,52,-9,2,1,19,75,43,18,-76,-28,-69,2,79,-3,-48,66,-24,-8,4,47,-93,99,1,35,70,-45,33,63,80,-57,-36,32,33,66,-90,-48,41,30,47,41,-98,77,-56,57,-48,-28,23,27,40,-95,-49,46,-96,29,-19,50,60,90,13,40,-67,76,-51,42,19,-65,94,-63,64,18,78,43,-5,-78,-23,23,-30,-99,-73,-90,81,-45,32,61,60,-86,11,20,-96,-76,-63,-86,76,-15,55,-27,19,26,9,-39,-56,-13,80,16,-15,-43,-84,54,-66,-57,40,-85,-26,-27,52,-66,-37,-37,30,-57,-37,66,-44,-83,-49,88,-11,46,14,97,-93,35,-38,86,-49,23,-56,-57,-45,-46,-15,94,-31,35,-56,-79,68,6,-39,-2,49,23,-36,81,39,-9,69,-95,-62,-39,-97,-79,95,-59,-92,-77,63,-72,-35,-82,-19,27,89,26,-38,-67,24,-92,39,84,-18,-12,-16,46,-54,-99,-86,-84,-95,-49,-47,83,-51,-75,-99,-44,-53,64,-40,-12,58,17,91,47,-56,-69,80,67,-62,19,28,19,83,-88,-58,-71,88,-67,-79,93,-40,-27,53,8,-3,54,40,20,-5,99,85,-47,17,76,-99,37,83,80,-19,-79,-24,85,17,-64,73,35,-36,62,67,-38,32,-72,34,-15,35,-91,-84,-24,5,10,-48,-10,63,45,-56,63,-18,-73,20,39,-75,72,24,41,-92,-2,-47,-52,-63,-79,9,68,-75,-79,-70,36,-94,-54,-88,10,55,-59,77,-5,85,-79,-42,-55,-53,54,60,-29,26,-38,88,10,36,-81,35,72,-84,-56,17,-60,-59,46,75,-77,68,64,33,1,81,10,95,66,7,-69,87,53,61,25,-98,-12,86,66,74,22,84,9,-29,99,30,87,-84,47,10,68,70,-21,32,3,79,-10,89,51,-44,-27,81,-79,-97,43,45,-96,-92,8,47,81,-70,8,-32,0,-15,97,-36,0,21,73,-32,-9,-71,76,70,-92,-34,36,59,-1,-91,17,-4,-88,-40,-82,91,-56,25,38,-97,31,-76,-30,-92,8,-56,-29,84,64,-56,-70,-67,-27,-94,-97,-43,-51,39,92,24,47,10,96,58,46,-86,27,67,15,-35,69,-54,-12,-61,-47,72,-41,23,57,99,43,86,-68,92,68,-65,-50,-83,50,-81,17,-3,5,13,-67,-71,3,59,95,-82,-99,64,-37,-34,79,15,-62,-62,-85,94,-86,58,57,-54,27,26,-43,53,19,-93,72,-63,3,-46,26,12,-18,-70,-52,-23,-53,-51,-59,-13,14,96,78,28,-89,93,0,-75,28,-43,-30,-45,82,26,85,-21,32,57,15,-88,-89,-82,1,-30,-76,48,46,-29,96,64,57,-13,37,12,-84,48,5,15,72,-90,48,-81,-58,31,21,-73,86,-70,83,1,-59,70,-5,41,-83,19,-33,63,66,-37,4,0,27,-59,-88,19,88,17,-89,37,3,58,55,44,-34,-24,70,-48,5,30,29,23,-22,-75,-59,-6,20,7,-43,-14,47,37,85,-26,78,96,92,43,-10,-97,-42,93,37,-87,37,-96,-34,8,-45,47,-85,84,-30,-8,85,11,86,5,94,-80,90,41,57,75,91,12,48,60,-68,-61,39,-11,-91,-23,-21,45,-20,44,-70,11,-9,-55,-28,38,13,-43,-74,-1,38,-80,-4,28,38,-70,-20,29,41,-71,-10,49,-56,-71,-84,-71,-18,-6,50,38,-62,80,-73,5,-98,-2,-80,14,32,-55,90,-30,-35,-14,98,-97,15,-45,8,-67,-40,74,81,-20,80,73,8,-38,-33,-41,77,-19,-61,-96,63,-83,1,82,7,-67,-72,-26,3,-31,59,-22,-52,-49,32,55,-17,92,30,-58,-28,10,-85,57,48,-41,-84,25,39,-69,28,2,-53,-93,-38,30,-61,-11,-95,18,57,40,95,-95,91,5,36,-49,-3,-57,-31,45,29,-39,2,-22,19,17,79,-65,24,8,14,70,14,75,-22,29,-59,-18,-52,97,-78,-80,78,89,24,-9,-59,97,34,-14,43,-37,46,-78,17,-58,16,-3,-23,-60,81,67,87,-28,42,-35,-99,82,23,-75,56,21,-78,11,11,45,78,28,43,12,-86,-37,52,37,-39,-54,78,53,42,-68,93,0,98,57,71,17,98,49,-24,-2,50,-91,-81,71,19,6,17,-25,-89,-63,86,-98,-24,15,38,13,61,-7,-33,-20,25,-40,80,0,93,51,-82,91,-23,-30,-11,-72,-21,-16,98,74,67,-8,-51,77,5,-87,78,80,27,-7,-6,-35,86,37,45,-12,96,2,87,66,-47,81,57,30,-48,-77,34,7,-17,32,-19,-50,-98,30,-72,6,19,-94,-36,-77,-1,34,-12,-38,-29,-90,49,-56,-12,36,-90,17,-5,43,-53,46,42,57,-47,-75,-33,10,-25,-32,-83,2,50,35,-16,-86,58,59,24,-77,-79,94,-91,69,-85,-4,6,-99,-87,77,43,-63,23,-15,70,52,10,-63,-38,61,4,-44,-37,-45,90,23,-55,25,-17,69,24,3,40,9,-27,55,81,55,55,71,32,-24,84,-68,37,54,60,-53,91,-78,84,-28,-46,-53,26,-55,70,48,46,-47,17,-52,32,-66,56,81,88,38,13,21,-14,-55,73,-30,-47,10,-99,-87,34,91,-88,18,-36,-35,42,66,-14,12,14,-67,41,-92,-20,-50,-58,-87,-68,-93,27,44,4,-87,66,-23,-41,-81,64,-40,-91,-2,-49,96,92,-9,-39,34,-65,47,-77,25,56,63,10,-87,90,51,-75,-78,34,29,65,-84,41,8,-8,-99,4,-67,-63,12,30,-36,8,0,32,45,-66,66,-31,-43,-32,-98,96,77,-86,86,5,-84,-92,-83,44,50,-68,-38,-65,0,-61,38,33,-25,-73,-60,39,11,39,-29,-43,73,-86,25,-93,-19,4,-97,-64,17,89,17,9,73,33,-47,0,64,15,35,-35,30,-50,74,-94,-23,-86,21,87,53,91,-56,3,81,46,9,-61,-73,-12,-26,43,-46,68,-47,-95,1,-18,4,-34,73,-84,7,4,-34,-42,-14,42,-29,-93,6,-99,74,27,-20,55,-50,65,70,-24,-46,-78,-81,7,89,-52,-12,-10,7,92,32,-20,84,-84,60,-50,-27,-54,68,21,-48,-25,21,-74,78,78,57,-72,43,28,3,96,49,-2,-19,38,46,68,-95,-47,37,37,9,22,52,-30,48,-97,15,-6,23,44,68,-79,46,46,98,4,50,42,-68,-47,38,57,50,95,-28,-27,64,-24,2,1,-87,-88,99,42,-42,48,44,49,-58,-56,-7,86,-36,-60,9,-37,20,-64,-95,28,-12,19,-15,-85,15,56,-12,-44,-68,66,33,-78,54,33,40,-88,-42,-39,61,-24,4,-69,-61,44,-53,24,-93,66,59,87,-6,-76,7,55,-62,98,-12,2,-69,20,-31,63,18,-77,-27,-65,-65,7,-5,72,59,75,79,74,19,-73,-2,26,92,-66,-10,-13,-43,73,-81,-28,48,7,73,78,-96,-81,19,21,-58,91,32,-47,-24,-73,-75,-65,-98,-96,-90,-3,-93,-16,99,99,-5,-10,-37,51,-37,-19,99,-88,87,50,66,90,68,-15,-12,-90,-46,96,-38,29,-77,-37,40,-99,-56,26,-25,-50,-89,-26,-74,5,63,-12,-67,-96,46,-90,91,10,59,57,-22,27,-80,65,37,-27,39,75,78,61,-61,18,39,-41,-55,-86,-15,-68,-13,11,13,-72,75,45,-92,21,54,75,-68,13,32,9,-59,28,-49,54,-99,89,29,55,28,44,-26,-33,79,-5,-43,-35,26,20,75,-61,47,50,60,31,-51,-8,6,-20,-95,-84,88,-78,43,-84,52,21,-94,-18,-24,33,-97,26,76,82,-79,-67,-53,23,52,98,38,-23,48,-1,84,96,90,-9,-47,71,6,-82,70,-74,-66,22,23,-84,80,99,48,83,25,1,65,23,10,-88,-77,-61,-90,61,-8,34,-40,-23,8,-73,67,60,74,-50,78,21,-48,88,-79,75,-96,-99,74,28,83,-24,-71,25,75,15,13,98,53,-1,59,-54,-66,95,22,18,98,-34,-45,72,15,-67,-29,43,97,-9,18,-99,-9,-8,28,-48,45,33,53,20,-52,-33,-5,-98,65,-46,-76,75,-74,45,-30,-76,87,25,73,2,34,-56,-78,32,34,40,9,-98,-91,-62,-47,53,47,-93,-49,95,-50,22,73,91,76,73,44,78,18,-86,-97,-18,38,75,60,-50,95,81,81,29,21,-10,7,30,4,60,-40,-48,43,-13,-76,69,-90,73,60,62,46,4,40,-59,18,19,22,-67,-6,-18,-18,-34,-60,-60,94,61,7,-21,-9,-89,-84,27,-61,58,-85,-61,4,-76,-88,-36,85,34,45,2,51,-60,-78,-50,-51,15,8,31,58,48,47,52,9,-46,-69,76,-35,-54,3,-20,80,-6,19,-39,17,7,2,-21,-81,-53,81,69,-37,-97,-80,11,-6,4,-81,-48,-48,66,81,37,96,88,-10,-62,-88,93,18,68,-13,-86,-70,81,20,31,-40,38,54,-59,-15,93,-80,-19,-19,14,84,0,65,-86,-57,23,50,-84,12,-59,-46,23,-89,71,-32,-26,84,-3,-45,-19,-95,-8,-80,35,9,4,28,29,61,-90,-80,45,-14,61,58,5,-15,-14,20,73,3,50,-27,13,21,40,63,-18,-62,95,63,-81,-13,59,54,72,-60,-41,-22,-99,-55,-3,22,7,-64,-42,88,20,43,9,-30,-54,-64,42,35,34,-18,-25,-84,-4,-30,55,15,33,-9,-54,6,7,4,60,-16,26,-42,-93,-67,-7,-36,-78,89,83,7,-41,-94,42,77,40,-24,-41,-85,-31,55,-38,-99,-53,94,-8,-8,-23,75,72,-62,-40,-2,-28,-34,8,41,6,29,7,-11,-64,42,-6,54,19,-89,-92,77,-98,-24,9,-60,-24,32,-65,44,-99,-12,20,73,2,56,-52,50,98,55,-9,4,-16,74,69,-4,16,-37,-72,-65,-50,-65,89,-72,10,-25,-32,62,-92,2,-16,84,66,80,57,-54,36,82,95,34,37,-36,-85,98,-62,83,-6,54,-77,-2,65,-28,32,54,-1,-81,-70,66,58,-86,-55,-58,97,88,-78,32,-66,34,14,-94,-55,-49,-31,-41,-51,-16,-58,20,-62,-36,-82,79,12,27,34,-89,45,40,54,3,30,98,21,4,63,19,-64,96,-47,26,79,-3,54,-52,32,79,31,73,-1,45,-86,-6,25,25,97,-64,13,42,52,-33,-77,-41,-58,43,62,-94,-37,75,78,-8,1,57,89,-45,82,-2,-65,-10,-28,10,-87,-38,80,14,87,77,26,0,-3,-22,43,-4,36,61,-61,-25,43,-22,-50,22,70,28,-44,36,59,-85,-66,70,-18,-18,-19,-6,44,-62,-92,8,92,10,84,-34,87,27,61,0,-12,76,51,-91,-45,78,7,-98,6,62,37,-35,54,47,-64,-64,-70,-7,6,50,7,-10,-42,-1,99,-58,64,63,45,2,-60,-90,-21,90,-82,33,68,24,11,-49,-36,-52,-7,94,71,5,-93,77,74,12,27,-19,-98,61,57,77,79,-2,40,1,-99,56,11,-21,46,5,88,-8,-93,-1,19,-30,23,-88,-36,-6,-7,-30,-51,-33,58,75,25,36,14,-18,13,-30,79,-70,71,56,85,-18,-65,31,63,99,99,69,-2,19,-84,-79,7,-44,-8,-99,25,40,-56,-40,-8,68,95,82,-50,8,-47,-93,37,-76,62,-1,-18,74,-92,-55,73,-93,91,-51,2,-93,-54,86,-37,37,63,64,-46,-93,-75,23,52,96,5,-98,5,57,84,-81,57,24,18,-61,-2,2,-39,-52,8,-48,-4,87,-65,42,73,-3,-21,36,-61,10,19,-37,-67,-29,-41,37,-50,40,-28,-66,36,-71,34,-69,-55,-68,32,5,-20,-83,33,75,-96,44,17,53,-81,73,-34,-43,-40,62,-4,-8,-67,31,-94,-41,-51,-23,-7,84,-17,-96,91,27,-64,0,-91,-8,16,18,-56,-3,-38,38,-73,-20,11,-7,36,47,-45,-91,-84,-36,40,-79,23,88,-25,92,49,34,95,40,-39,-69,-60,-31,-1,32,63,43,-94,-75,-19,-67,-18,68,2,-82,15,56,-73,30,96,43,27,19,31,78,11,-20,12,7,96,72,14,-87,-82,-86,44,80,33,27,82,-86,59,-36,81,-39,-42,-27,93,-39,-97,89,-96,6,9,34,-15,96,90,96,-20,86,46,71,98,40,84,20,-2,-6,-53,-20,7,-94,43,-12,-57,77,37,-64,-61,39,-98,-58,-78,-13,53,6,-16,43,-20,63,30,-97,34,28,42,-5,-75,-60,88,-29,-4,-28,-24,-84,36,-5,93,-27,30,-68,88,9,-27,-89,95,-97,-7,-21,46,72,-80,-24,74,30,80,-6,-74,-94,10,90,-47,-93,-60,6,98,75,-99,91,-51,8,-77,-86,-83,-27,-76,-88,-25,17,-9,97,-34,-13,-27,-82,-83,-69,11,-81,-87,97,-90,-35,80,-51,70,79,-99,48,70,-74,-44,70,-60,-28,-80,-37,-17,-29,56,50,-31,-77,-63,18,-83,-69,25,4,-51,37,1,-65,79,-18,-40,26,-39,-39,-26,-92,-14,29,77,25,77,73,65,-39,-55,21,87,-10,20,-98,-92,36,-68,10,-60,-20,-53,18,15,-97,99,51,29,-63,-88,-97,-79,-25,9,75,76,-14,48,41,23,69,-60,-12,-41,59,-11,-56,-27,-3,53,-11,76,76,-93,-32,79,82,96,-15,-81,-92,-36,-60,-41,72,-85,-64,-64,39,53,35,9,92,-77,44,29,-12,64,-98,84,17,-10,38,-29,-26,82,-73,-44,78,11,-25,63,74,-9,-78,24,-18,-43,59,97,-89,-29,-17,79,-6,-72,8,81,68,-14,66,-14,53,4,33,-73,-37,59,-41,-58,-30,33,-95,-79,-99,-74,44,-41,-40,-20,-43,-53,51,16,-74,-55,43,11,-97,12,73,45,74,3,-74,-16,6,-11,-57,65,30,12,-25,34,32,-47,36,-46,11,72,33,44,-81,-16,-39,-78,5,80,9,8,-31,-18,-47,-80,62,-21,3,-32,67,45,-67,-26,-66,-15,84,43,37,-3,96,-52,-30,6,69,-12,90,-70,-14,-28,86,94,79,-44,-46,-90,-48,-84,-35,55,-17,-91,-23,-7,-18,-12,77,-57,-69,14,39,-96,38,-91,9,7,-26,-24,13,59,48,0,31,4,32,84,13,83,99,54,38,-41,62,92,28,-79,-20,-18,-37,10,95,-97,-10,11,87,98,94,-39,-48,8,97,99,84,28,-20,16,12,93,-1,-12,47,-85,23,-13,-93,-48,7,-14,33,70,72,6,49,-38,93,36,-40,87,74,-12,-5,48,-13,-21,-24,67,71,64,37,-29,-71,-39,-15,51,47,-32,79,-68,53,-10,-98,25,95,50,86,65,-37,-77,52,-63,11,-75,61,74,-20,-63,18,-48,77,-45,22,82,-7,83,11,-83,-49,-10,-52,80,79,-51,5,51,75,-32,16,-61,-10,-54,-48,77,-30,90,52,49,-73,-30,-99,81,2,-1,40,71,-18,-49,87,9,41,-64,-34,-3,-39,70,-74,-86,-85,18,-71,-18,63,-43,-41,-67,-53,87,82,-49,-42,59,-68,36,-41,48,7,17,99,-6,-96,17,6,-31,-86,44,-83,-84,57,-69,34,85,12,-3,-58,48,7,-34,35,-34,16,92,-75,24,5,-17,-28,-88,77,-29,-17,80,87,65,-51,78,9,-35,93,-57,-4,4,-95,84,78,-53,32,-15,-11,-32,50,-95,-63,74,28,42,-65,99,30,-88,-52,89,91,-65,55,-83,89,-59,-18,60,-16,54,-36,-12,38,-58,11,-52,26,99,-8,-47,80,28,-95,8,-53,-61,8,54,-50,55,43,-82,66,-25,-66,56,-7,-8,16,53,-54,56,-82,61,98,6,85,1,5,-23,31,-14,81,35,70,-71,73,-45,59,99,86,-21,93,53,31,-96,-14,-76,-4,78,53,18,34,70,-44,9,-24,-59,-89,58,-6,41,-79,-24,-47,90,-19,-74,-54,-60,-98,-68,-5,94,61,-74,74,-53,-74,-30,-98,78,88,36,49,-56,-55,-75,-38,-68,82,55,49,-97,-69,-98,70,-12,-96,-84,27,-95,-76,-77,75,85,-75,50,-91,-49,96,-89,-71,84,46,54,-94,67,79,-56,99,38,98,25,18,-93,27,-12,-6,30,-20,98,12,3,97,-13,-35,-78,-86,-26,-51,-89,-39,-22,71,-93,-91,76,50,87,96,26,3,95,-48,-79,78,78,-15,-51,-15,64,-53,96,-56,-56,-39,9,-57,-26,59,-9,-39,-80,45,-67,-97,54,85,52,18,82,-44,97,54,7,94,9,-38,-44,57,46,-80,-95,20,-59,-75,57,49,-56,-92,-15,12,-31,4,-43,1,6,87,-37,-65,-17,-55,-10,-43,98,96,-48,7,-65,7,-58,58,4,-54,-22,44,-53,-65,-30,-9,-58,54,79,-13,-65,-87,87,17,-99,-49,51,-40,-28,-81,-7,-52,91,21,31,-73,-94,72,84,9,94,-61,-70,-81,-50,-24,-14,-32,-93,-35,55,-59,-23,19,57,-46,46,-15,13,-81,3,83,-57,-5,4,73,-2,9,23,-18,94,17,20,-99,35,46,-47,-79,-86,-41,-38,68,-24,39,-12,-90,92,-89,-28,-17,-94,74,-34,-52,-54,69,98,43,-44,21,-98,-50,-85,-2,-72,50,-56,-20,47,-42,16,-90,-97,91,25,-33,78,17,-22,-50,99,-40,-99,-58,8,-53,-88,6,66,-56,-96,45,-7,-82,-57,20,44,-36,77,-8,-2,-7,-22,0,-16,-97,44,-61,-4,98,-12,-4,-42,-11,37,65,12,25,-52,-45,68,-49,99,-61,-54,-80,35,89,-17,12,-19,57,81,35,-66,-34,14,77,-19,-89,75,69,-17,-89,-65,96,75,46,22,-99,-99,-33,-72,77,-18,-27,96,-82,-37,-44,6,20,-87,-12,-67,-77,30,46,-99,-89,56,52,56,-61,62,90,-64,15,36,34,-8,14,77,19,-9,-40,69,-13,-46,-68,19,36,28,8,-99,60,30,30,7,7,18,-60,-63,-26,-44,98,-35,-32,-10,77,-98,81,-9,55,-22,81,91,-53,-55,-78,-45,40,58,82,-52,58,20,55,-34,-73,-38,60,-57,-25,-66,-2,-49,-25,65,40,-47,43,-1,43,-1,76,2,-33,-77,23,-11,-23,-36,-76,-64,-12,58,55,-57,-76,58,-18,83,-98,56,94,-24,-93,-31,-58,24,21,61,-77,-58,-63,98,43,4,97,-34,69,-49,29,92,-37,-82,50,-81,-63,51,53,-5,11,31,-48,5,84,-65,51,-97,58,-28,63,57,13,-99,32,-67,-19,29,-1,-50,56,-95,-58,-80,98,68,-85,-87,19,44,7,-92,-24,58,12,-40,70,-60,-38,5,12,-98,-37,-98,-98,94,-65,-41,1,10,8,-43,-9,49,-47,-10,-5,-33,2,-86,-88,-91,97,-36,44,10,0,14,49,38,95,-62,39,-42,-83,18,-70,-49,76,30,37,-15,63,-72,-89,-83,-6,-18,-40,95,-4,-29,81,-7,11,2,79,-89,-84,6,25,87,20,-58,-54,36,59,-48,63,12,81,-23,73,21,5,61,37,98,-57,97,71,15,-55,29,85,33,-69,64,20,-77,-53,-54,-90,67,-13,-45,-20,-77,6,-79,34,-36,-3,-15,85,-98,-54,-1,77,-12,-27,25,-96,-5,-46,88,-72,-16,29,24,6,76,46,-8,20,-67,24,76,-68,-93,96,43,-29,-6,-72,-67,-28,-27,8,-74,60,81,50,40,-47,80,28,-20,-36,-42,80,46,-89,27,-84,7,36,39,-17,68,45,-43,11,92,50,38,-98,98,87,10,-98,47,-9,51,-12,43,9,92,99,72,-73,56,-4,-63,-17,-12,43,-80,-73,-96,64,71,-40,-48,41,86,66,42,84,53,-48,85,-23,19,14,41,-61,-77,34,-85,71,60,71,-56,74,-46,31,-6,-50,57,-3,90,6,56,-58,-53,19,84,88,4,37,-82,-34,91,36,56,32,75,78,-57,89,27,80,37,70,54,68,-21,-52,94,-87,22,85,18,-45,-96,41,74,88,7,-45,25,24,20,16,-40,-46,-74,-88,-68,-32,78,35,-52,15,6,78,-17,61,-96,-22,73,25,-60,-8,56,-57,-90,30,-69,16,61,33,-60,-41,-74,-23,12,-49,-12,20,-4,-57,-44,20,58,38,-1,-82,-1,2,-5,-28,-96,11,40,36,54,50,-33,61,-34,-95,-29,-17,-37,-3,36,74,24,-76,-5,-80,-33,27,17,-98,-35,-84,-81,63,-6,-9,-88,-26,-98,52,-90,32,-21,-47,-6,21,-42,41,80,20,-62,16,-28,39,17,43,35,83,-30,-48,61,34,-33,79,-26,37,-53,-38,-89,-74,90,-79,57,-31,-50,-72,67,7,69,47,-72,83,-36,-1,22,80,-81,-42,40,-12,9,-22,98,53,-43,-51,-10,-19,10,-22,6,1,-25,-60,46,24,-32,13,-68,-86,-62,-64,96,-22,11,-4,-65,29,53,51,-83,-61,-71,91,91,-38,-60,-41,42,49,36,-75,-73,87,-35,73,11,9,-37,19,22,-23,54,-5,54,-58,90,65,70,43,16,63,82,44,54,50,-93,93,-91,25,43,21,50,46,8,-85,19,96,23,59,15,-78,-64,-53,16,67,-12,7,32,-41,-73,-51,-78,85,69,53,13,75,46,21,1,-34,-80,-72,-87,4,-58,31,0,41,-33,-84,39,-20,-38,-45,-76,49,38,55,84,42,80,83,4,50,36,-83,2,-18,15,79,48,34,-93,60,-62,-75,-32,38,42,12,30,-19,91,68,36,91,-6,-49,23,-21,69,4,-38,-26,-69,74,-10,9,-44,81,-11,80,15,-28,17,30,-3,62,-55,-61,-26,51,20,-58,19,-67,-67,-87,-40,55,-9,-70,-64,29,-20,-57,3,46,52,36,28,-82,16,-80,-34,-89,26,62,72,70,78,-54,21,-25,-36,-60,-16,-4,-70,20,-72,20,49,-59,-73,-94,-17,6,52,-88,-58,-20,-71,58,-24,-5,45,-97,-66,17,72,88,39,-29,-60,-97,-13,23,-25,16,-56,2,-86,69,42,-60,75,-97,45,27,-86,64,83,-80,-1,-64,-9,-56,-62,1,60,87,-10,-1,-42,29,1,44,-70,75,-62,-50,-46,-49,-81,73,-33,70,75,88,-26,65,52,-66,61,-49,-30,-48,-29,7,53,-69,-6,-80,29,28,25,-93,-50,31,58,-13,80,-87,14,76,62,57,46,37,22,97,-21,-25,-69,-59,2,0,-8,73,84,-78,80,-45,-82,10,83,20,-7,-90,-72,-48,-27,-91,-59,63,61,-97,20,-16,16,-58,-19,71,-7,-11,-88,72,65,-19,45,49,-21,2,-95,-26,88,64,93,-18,50,97,10,-77,82,50,85,43,29,5,-95,-78,23,-15,92,-83,50,-95,65,16,-38,-13,-58,40,88,-54,-86,77,86,83,-41,36,81,45,58,63,-28,20,-16,-99,-98,-12,-78,25,49,90,18,-23,71,-17,92,-67,69,-66,73,34,-43,63,11,-80,24,46,56,5,-9,-9,-55,62,88,5,39,89,69,37,14,-4,28,-91,72,-1,-32,64,32,-63,74,-18,48,-69,-78,-41,-49,45,82,83,-73,-27,-49,-51,-87,-61,53,28,-71,0,-57,19,95,70,28,67,-54,95,8,54,-90,-18,-87,57,13,-65,-84,40,79,74,23,83,-75,50,-68,-63,89,61,41,-82,-39,83,36,-67,30,-59,99,76,-63,7,7,-77,89,20,79,-21,31,71,95,-13,46,-5,-30,70,-55,1,83,-66,-60,24,-72,0,-15,41,33,-85,81,9,67,94,93,-48,-83,59,48,72,-62,-21,44,-90,65,-10,4,-87,-63,48,-10,-80,59,29,20,63,-93,81,4,16,96,-37,-75,40,-43,-82,91,-49,53,16,-77,67,94,66,76,37,-67,-20,49,-54,-94,-61,-35,41,68,62,4,51,43,85,-33,16,-52,-32,-43,-19,-37,-75,31,-84,41,54,83,12,-3,59,-51,7,-83,74,52,98,90,94,39,58,56,-80,-14,-24,-95,52,92,-48,-80,-74,33,-18,-73,-59,-2,67,94,-42,-43,-31,-83,-18,52,-90,-43,4,84,-53,-25,0,81,-92,20,-33,-16,1,95,-47,30,-85,-22,-60,-3,-19,-20,71,-51,-48,-71,-18,20,-78,63,-51,-91,96,52,92,-57,-95,-30,-99,-88,66,-56,-5,-33,-61,24,96,-69,78,-64,4,-40,-7,-25,-15,44,-20,66,-59,1,6,-11,-14,3,-82,-21,22,-78,25,23,-66,-9,43,5,57,82,-70,31,89,-92,43,-7,-33,-87,67,28,56,-53,-6,96,24,-99,-38,-89,3,-20,-34,-98,1,67,24,34,-42,-55,-60,91,-96,-54,-1,92,-70,-80,-15,-26,-68,-71,-98,88,-48,-4,61,52,-27,-76,39,52,3,81,53,-96,48,-45,38,-94,-1,54,74,2,99,-27,-29,-71,-8,55,-21,24,60,-19,-11,-88,76,-50,40,-74,-27,-20,77,52,-39,30,56,-90,84,-29,91,-40,-98,65,38,77,-61,-91,82,7,40,-38,-69,0,42,19,-12,-5,46,-71,20,95,8,-3,48,68,-96,4,54,65,51,46,-75,52,11,63,29,49,71,11,-43,-11,49,-36,65,-9,60,-46,85,6,-41,-18,-22,66,55,-74,11,59,6,-34,24,-43,88,25,85,-1,-12,-9,-74,-63,79,-41,2,28,22,-33,96,-18,20,81,64,55,-36,-57,21,95,-55,-90,54,51,-48,-45,84,-60,80,70,38,-55,-62,40,-42,-83,98,59,22,21,-73,18,79,23,-24,21,-21,16,40,76,12,84,62,-34,35,-86,-2,-3,-70,-22,-56,68,-1,-19,-15,57,-25,-16,-7,96,-95,-4,-9,60,96,66,-42,-48,-17,97,-95,71,82,67,-86,-6,80,11,90,87,65,-66,55,-35,-8,-83,98,66,0,-32,39,-19,64,-70,-58,37,72,-1,88,-68,-3,92,-97,-44,36,93,49,17,4,17,-19,-30,27,-64,11,-81,-71,-14,61,29,53,0,-90,-6,6,-49,-92,-44,27,95,-36,-99,65,-33,-67,1,-40,58,94,40,-25,-25,86,79,-13,-26,97,16,59,36,-55,-10,-87,-69,60,-4,-18,68,51,85,40,-85,62,5,-42,94,-93,-6,52,77,-65,-72,-47,97,83,16,-28,80,-68,-92,-7,-47,-3,6,-16,-42,-98,42,-97,29,4,42,43,-34,-52,78,-40,-69,-28,-88,-92,-17,-84,-63,-19,98,30,52,56,61,36,-51,91,-66,54,-25,-32,32,93,-30,-39,-3,-88,81,62,-64,59,-78,-57,-69,-90,-49,14,-74,-36,71,-99,93,0,56,32,-64,82,23,-54,13,74,13,-54,67,59,6,-36,71,-13,-74,83,-54,-76,-73,-46,-67,-46,44,-65,17,15,-64,-12,91,68,96,-73,50,19,-27,-36,-30,62,9,-63,22,91,-23,-30,-44,2,-47,-22,2,-44,31,11,9,-25,46,-96,66,81,-32,57,49,-36,61,-23,82,-89,40,-48,72,-51,-35,-6,-83,41,40,-28,-80,70,50,21,-74,-19,-68,35,33,54,15,-24,12,82,-89,-38,-54,48,38,-95,58,-45,-67,7,3,-3,78,19,-85,18,68,-66,-12,18,31,14,75,62,-74,-15,-6,40,61,-94,-1,48,-33,21,-4,81,25,-69,-87,34,37,15,-69,-8,-88,-55,-89,79,-45,-25,-3,85,65,-51,24,90,34,-82,7,-28,-76,82,-80,-33,3,-8,-74,-72,-1,-62,38,-87,30,68,5,41,89,15,-3,-79,66,70,-94,32,-81,7,22,29,24,-93,-99,24,88,96,91,-31,-35,93,-27,63,-69,-88,76,60,56,-19,-22,23,72,51,43,38,21,26,-53,40,-67,69,-54,33,52,-53,-42,17,-57,-74,85,8,-81,-41,-52,-51,-30,-76,86,-97,-19,-59,25,-70,92,-31,45,13,71,91,-70,-96,37,-24,13,-11,-1,70,83,41,95,68,-74,90,-96,-49,-83,-50,-49,-97,-47,9,-57,77,38,-88,-77,-17,25,93,-25,31,73,-88,-93,-13,-22,-18,-66,-62,-99,-93,-94,-74,96,86,-47,-87,-87,4,91,-35,-87,-88,-80,27,-77,-81,-89,47,-87,-38,-44,62,-50,-61,-51,27,21,-40,64,21,-34,47,23,-38,10,76,-25,22,-43,-57,64,-31,-46,60,95,-23,78,82,-99,67,-56,-44,7,-6,-5,55,97,92,-85,39,13,-43,-37,-64,19,49,88,70,-52,-55,-87,88,-10,-33,48,-37,20,4,-78,20,-29,-34,53,54,36,-75,10,10,-83,1,-74,6,58,-35,18,-46,90,7,-99,-61,28,-87,-73,95,-44,75,-65,52,-44,-44,-27,3,-2,-97,-42,-89,-73,44,-2,-80,-55,23,2,79,65,-80,-90,55,3,-90,70,-91,99,97,-96,-45,49,-62,84,-95,-7,-43,7,68,-64,64,78,-61,8,-24,-65,30,76,36,-14,18,55,-4,-50,-64,-18,-79,-56,-19,-82,-53,13,66,-39,-3,47,30,30,-69,-2,65,95,-23,80,-20,29,-85,86,-18,50,-50,99,-18,45,-50,-6,-96,-30,37,-15,-36,60,97,-93,-79,-6,53,51,-99,83,-74,65,55,-21,22,-87,-91,36,98,-10,-37,-52,-33,21,69,16,-84,73,62,29,-42,-74,-10,-68,-68,-13,2,61,-62,2,44,-59,-56,-23,19,65,-11,4,-22,-36,94,40,12,-62,-62,-42,53,-47,-92,-8,82,-58,17,-51,-27,25,35,51,-14,50,-47,-93,90,-4,-17,-14,-62,48,90,15,-87,84,-68,1,98,-30,58,51,-1,66,-80,-19,84,13,6,-42,-62,-58,85,0,68,-61,6,35,11,89,-79,26,37,-89,-59,26,71,-27,-73,69,42,85,-3,-82,28,16,-1,-11,-71,81,46,-57,99,9,42,67,47,-51,2,-65,-85,23,60,28,10,-98,-45,-18,-49,-19,-49,69,-57,48,86,47,-59,-38,-63,-30,-56,59,12,-80,-32,-45,87,91,-20,-34,-73,-6,65,86,-78,-24,64,-47,34,91,34,84,37,53,9,24,-22,50,62,14,96,82,-50,8,-97,18,39,66,-14,95,-91,12,-11,74,-24,-12,-73,-83,40,60,-91,-49,21,45,81,-69,-54,-41,80,-91,49,53,67,98,-62,69,-7,77,35,79,72,21,-32,-61,-5,20,3,-2,37,-57,-42,22,-29,78,-56,-48,85,-33,-13,42,-48,-64,-5,-81,-89,33,88,4,-13,0,-40,-64,97,4,-49,-8,-75,53,89,38,96,-76,-63,-56,79,-43,-28,64,23,58,7,74,70,-98,70,80,11,35,-39,74,-88,-3,-89,-90,78,-39,1,79,-85,-32,17,87,-9,30,-92,-53,86,-21,-88,10,-86,18,-39,83,-4,-69,40,84,65,78,58,77,-25,68,86,-47,-93,-36,-68,-79,31,25,84,99,-45,91,-54,-58,-30,-43,28,-17,51,88,-57,-75,96,82,-91,61,60,66,38,-87,12,24,-58,18,88,-49,15,-4,75,-24,-28,-92,-33,18,-74,13,51,53,72,-20,18,91,4,14,73,-11,-24,11,-68,90,0,43,-8,41,-62,-43,91,29,52,44,-95,-75,51,-52,19,53,60,-30,6,9,49,-99,0,-70,15,-26,18,67,61,50,34,-39,-30,-74,-98,8,-18,70,-63,-88,14,-81,-87,-58,-34,31,94,-73,77,-23,35,4,-23,-64,33,68,-14,-71,12,-76,78,-53,-16,-52,-51,-38,32,31,-68,46,19,22,64,8,40,-70,-61,34,-67,92,87,45,-4,-35,80,-93,-90,-57,-65,22,-34,89,45,49,-63,93,87,46,1,96,-8,-80,-5,-44,27,-64,-38,42,46,-5,12,34,-60,84,98,96,90,84,-61,-98,6,4,90,-49,-70,-72,-55,93,50,22,89,-58,42,-15,-3,-54,20,35,65,-57,-70,-23,76,-54,-39,74,-58,-48,59,-43,-47,-58,37,-79,-30,66,-52,-9,37,97,-87,26,-84,54,87,-11,77,-16,-76,42,27,-46,-81,3,-24,-44,-45,18,-16,90,74,14,-91,-11,34,-22,-68,58,68,68,55,81,-28,47,12,-41,-64,-34,43,-41,-92,-53,-11,-97,26,65,-64,80,-40,19,47,-89,-67,56,-1,66,-89,31,1,79,-1,-67,37,70,79,-74,-93,-85,-31,26,-49,-47,72,39,-67,-2,-19,-32,-44,40,86,-97,51,96,35,-73,39,46,57,-59,2,-66,-27,-61,-96,-70,-59,86,-56,9,12,-6,62,-39,-89,94,59,-9,-61,91,-68,-98,70,59,97,6,85,14,-71,20,31,30,53,80,45,33,-90,62,20,-46,-28,-91,24,-89,-30,34,81,-94,-74,-80,96,-66,-79,43,92,95,-51,54,-14,-23,-26,-83,83,-73,73,28,37,83,-9,-66,-86,39,-58,37,26,11,-28,7,16,-26,3,-11,7,0,-67,-1,-28,-42,-70,57,12,4,50,-5,7,24,-99,-56,7,-32,-46,20,-93,-4,-66,32,-93,5,-84,0,78,-5,65,-38,94,97,37,-33,-67,67,-99,44,-29,51,-84,54,-25,92,75,58,-40,-71,-45,-34,24,-12,74,8,70,66,-92,25,-39,72,87,-44,47,24,-1,79,-9,-1,0,38,-73,-8,-7,-22,-15,-55,-64,44,-27,-33,87,74,55,61,-18,25,5,-34,-50,65,-62,-86,97,61,38,-4,17,-94,72,-83,44,98,9,-86,-46,93,-42,-11,37,7,56,24,-19,-12,-37,62,89,67,-72,-61,9,-57,-47,7,80,67,79,-2,72,28,14,16,4,99,6,57,69,63,-77,-16,-29,55,-92,51,-57,-53,-9,-68,90,-5,-52,0,14,-23,-93,-5,-79,62,69,92,-9,-40,-14,94,-40,91,51,-71,32,73,12,-97,6,-3,30,48,20,97,57,11,-7,81,-89,83,-42,93,-22,-22,55,-53,-53,-77,7,32,-83,43,1,44,48,-67,-5,-39,-88,-99,34,18,25,-46,16,-18,41,85,-37,51,68,-3,21,-76,51,-46,-30,98,-24,53,7,-30,-4,-92,90,-55,-83,-15,-18,-95,-38,92,23,87,-54,-84,-31,87,-99,9,15,45,82,-86,68,34,-33,-84,32,19,45,-84,65,-81,-99,56,-60,-83,-59,21,-78,3,13,21,-33,35,36,12,-1,13,-79,-85,-65,-96,-95,80,14,-29,95,-77,66,40,-62,32,35,-62,-12,74,-68,5,-28,-70,84,61,27,-49,-3,-60,40,72,52,37,63,63,17,67,43,-69,-62,38,-47,81,-44,-32,13,67,-18,77,42,13,-41,-86,19,-79,75,-54,-52,-51,-15,87,-79,-86,25,84,77,-58,51,20,-27,66,-64,2,24,-32,69,-63,-65,-48,90,76,41,-51,67,-39,-31,19,-17,17,67,68,4,64,81,-94,25,35,-52,-46,-44,-3,96,-32,75,20,35,45,56,69,73,23,23,-9,71,-10,51,-82,-91,-65,11,52,79,91,-7,37,97,-81,-27,-78,-28,5,18,-32,72,-7,87,7,-85,20,-47,64,-57,75,-44,90,-58,83,84,-73,94,-5,78,73,-13,-51,-12,-16,-33,60,5,15,-58,99,59,-86,-31,46,-3,-39,-57,26,-75,-38,-21,-20,-48,20,-36,-63,47,-42,8,2,-91,94,27,-4,-45,93,-67,36,-15,-26,35,-56,-13,81,66,60,-58,-91,-14,-34,-30,41,23,-1,-38,86,12,85,-79,-80,-36,-94,-85,91,1,-54,61,33,82,-53,-93,-6,-10,-30,-25,-43,-93,16,-58,-30,59,12,-89,-18,87,-28,-55,-24,-66,-58,-5,-2,-52,-14,-34,25,32,-73,-41,14,-27,-58,-92,40,-11,-40,96,-28,52,-62,41,11,26,52,69,-10,-99,90,65,-65,-67,-39,9,-20,23,51,81,-68,77,-60,45,-72,-41,30,44,24,89,40,95,41,54,37,-70,-42,-34,98,47,-34,89,13,-23,98,50,62,-22,-50,13,-41,-18,-32,-24,-96,-28,11,33,-84,-65,-1,32,-93,-82,-14,43,-53,43,-91,-78,-32,-48,87,57,28,85,7,-9,-60,33,-96,-24,-85,-52,51,-81,19,-61,28,11,-50,-95,-57,-44,-78,6,99,-55,-74,84,-34,-7,-64,-46,-50,41,-84,33,-68,-44,-33,-88,31,-19,-40,-41,-24,55,-3,-95,-33,23,-91,9,78,-93,91,54,-49,-82,16,-7,10,51,46,60,-8,62,-7,0,94,36,12,25,-82,48,83,-30,-20,-43,-49,46,79,-64,-67,34,-58,-99,-11,-8,-82,-95,62,28,32,-15,-35,-98,-76,34,-22,-82,-29,66,42,-35,-9,2,34,71,35,85,-6,-9,-79,3,-74,-60,4,-9,8,21,-28,-30,-51,80,55,-10,81,78,-75,-63,96,-28,-97,15,36,-6,-82,-53,-58,29,-91,35,-79,-93,-61,-54,-54,19,13,53,-60,84,-99,-34,64,55,55,23,10,56,59,6,-72,38,-78,40,-91,-84,87,-50,44,72,85,41,78,-99,86,-99,19,-24,31,35,-40,31,-99,-98,-37,33,24,72,-11,-40,-21,-6,74,-23,-66,83,-31,97,-67,89,70,94,31,25,94,-6,3,-10,70,-66,-74,-70,41,-97,-92,-96,35,8,75,-98,67,-69,-28,19,-16,-18,2,52,-20,11,42,-73,6,-27,28,-23,43,-69,67,13,-59,92,20,81,71,27,84,7,-64,37,-15,79,67,56,-2,-48,-62,76,80,-6,-12,-1,-3,70,48,-74,-53,-8,-67,90,-95,-26,59,1,-45,31,-71,-60,-85,-59,53,-1,96,20,-45,-5,48,-31,-52,6,63,12,-95,-40,-18,53,-38,6,-55,-5,-4,-74,68,-67,27,-77,63,32,-61,77,49,91,-24,45,-11,7,-83,-86,52,-35,19,15,76,24,52,-64,-46,90,41,-25,-15,-86,0,-47,-54,-96,-47,85,35,-32,62,-39,-40,-85,-93,25,-2,-77,38,-49,-36,-65,42,-82,35,-29,-47,88,-38,70,-37,46,60,39,76,83,43,-94,68,-45,-27,8,16,32,-1,22,-66,74,21,49,-98,85,83,43,79,18,91,-91,-17,-70,55,22,52,15,-39,28,-25,-96,10,-56,-64,83,28,51,-8,-96,-50,26,77,70,-25,78,32,-42,-1,-12,52,-10,-4,-88,-4,-49,33,-52,43,93,53,-82,97,-37,-62,-67,-54,65,60,38,46,-90,-59,23,56,15,2,-34,49,-99,53,-22,-33,-73,-11,62,-23,-78,86,-80,-85,39,-85,88,79,-48,97,-75,94,57,39,40,43,79,-60,-23,-29,41,42,96,18,-28,-26,61,-2,62,0,52,60,-13,48,75,-97,39,40,81,91,38,-93,62,-28,22,-21,-84,-21,18,-8,49,-63,-89,46,54,-41,96,16,34,36,-84,-14,-4,-21,-89,-52,-18,50,88,-37,18,3,-54,-43,74,68,35,89,-76,-70,-42,72,-34,-55,-5,-79,4,-8,13,-62,-95,5,0,0,61,10,48,-57,-63,13,-18,31,-84,28,87,89,-27,-1,56,-4,29,90,-55,94,-64,17,91,39,85,-19,53,89,63,-47,-11,24,39,-86,66,76,3,-75,-16,-5,29,-51,-38,-98,47,-82,-26,-47,84,-4,-52,-80,-87,-84,35,97,-26,-12,63,-63,17,-71,60,-43,-80,26,-90,22,27,-7,-82,-43,41,-21,-65,-11,-27,-14,18,-66,81,65,-46,70,58,65,44,-68,-69,7,-32,47,35,-72,-96,54,30,-87,-46,-43,-17,-29,-10,-76,26,25,-11,75,-89,-92,-91,68,49,-61,38,-16,-95,-18,15,-88,65,59,58,0,63,62,-68,-30,51,-15,4,-66,32,93,-65,-65,18,23,-90,5,7,94,-27,-66,-89,87,17,-85,-54,-90,-97,-89,68,60,-12,9,-1,-4,78,-49,80,-18,-39,-11,-47,-5,23,47,95,9,-47,2,-20,2,12,-10,-10,-70,-19,12,-84,-17,23,60,-56,87,-31,42,-17,-52,69,-60,6,-69,5,58,25,-95,6,20,-10,35,99,69,-63,11,-63,3,17,17,-84,32,77,15,-7,97,2,61,39,61,85,-14,77,-31,16,-18,-96,-82,-14,9,15,75,-79,-9,21,-42,-98,57,37,-4,-25,52,-72,-71,-33,-3,-74,45,-64,41,-17,97,-73,-40,-34,19,41,-30,37,-73,55,-71,78,53,19,0,-89,-3,-43,47,92,8,-23,97,36,43,-29,61,-34,6,3,48,4,6,85,46,-74,26,92,39,30,25,67,-15,-22,-14,84,-35,82,-81,88,75,26,-35,49,-60,85,-80,77,-49,3,-43,75,-93,63,60,29,65,-36,-1,4,70,-76,-29,-45,1,56]),9096),
        ])
    def max_sub_array(self, nums: List[int]) -> int:
        '''Kadane's Algorithm'''
        mc = ans = nums[0]
        for n in nums[1:]:
            mc = max(n,mc+n)
            ans = max(ans,mc)
        return ans