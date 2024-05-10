import math
import time
from typing import Counter, List, Optional

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

    @staticmethod
    def _static_init_list(l:List):
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

class main_exercises():
    def __init__(s):
        s.reverse_test()
        # pass

    def test(s,result,expected):
        # print(result,expected)
        print('pass' if result==expected else 'failed')

    def containsNearbyDuplicate(s, nums, k: int) -> bool:
        for i in range(len(nums)):
            if nums[i] in nums[i+1:]:
                j = nums[i+1:].index(nums[i])
                if abs(i-j) <= k:
                    return True
        return False

    def max_len_substring_test(s):
        s.test(s.maximumLengthSubstring('bcbbbcba'),4)
        s.test(s.maximumLengthSubstring('aaaa'),2)
        s.test(s.maximumLengthSubstring('sdaxczfdfsrere'),14)
    def maximumLengthSubstring(self, s: str) -> int:
        left = 0
        right = 0
        hash_table = {}
        max_len = 0
        while right<len(s) and left <= len(s)-max_len:
            if s[right] not in hash_table:
                hash_table[s[right]] = 1
                right+=1
            elif hash_table[s[right]] < 2:
                hash_table[s[right]] += 1
                right+=1
            elif hash_table[s[right]] == 2:
                while hash_table[s[right]] == 2:
                    hash_table[s[left]]-=1
                    left+=1
                hash_table[s[right]] += 1
                right+=1
            if right - left > max_len: max_len = right - left
        return max_len
    
    def is_happy_test(s):
        s.test(s.isHappy(19),True)
        s.test(s.isHappy(2),False)
        s.test(s.isHappy(100),True)
        s.test(s.isHappy(102),False)
        s.test(s.isHappy(1),True)
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
        s.test(ListNode._static_get_list(s.doubleIt(ListNode._static_init_list([1,8,9]))),[3,7,8])
        s.test(ListNode._static_get_list(s.doubleIt(ListNode._static_init_list([9,9,9]))),[1,9,9,8])
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
        s.test(s.intersect([1,2,2,1],[2,2]),[2,2])
        s.test(s.intersect([4,9,5],[9,4,9,8,4]),[4,9] or [9,4])
    def intersect(s, nums1: List[int], nums2: List[int]) -> List[int]:
        cn1 = Counter(nums1)
        cn2 = Counter(nums2)
        ans = []
        for k in cn1:
            if k in cn2: ans.extend([k] * min(cn1[k],cn2[k]))
        return ans
    
    def reverse_only_letters_test(s):
        s.test(s.reverse_only_letters('ab-cd'),'dc-ba')
        s.test(s.reverse_only_letters('a-bC-dEf=ghlj!!'),'j-lh-gfE=dCba!!')
    def reverse_only_letters(s,string:str):
        new_string = string
        l,r = 0,len(string)-1
        while l < r:
            if new_string[l].isalpha():
                while not new_string[r].isalpha(): r-=1
                if l < r:
                    temp_char = new_string[l]
                    new_string = new_string[:l] + new_string[r] + new_string[l+1:]
                    l+=1
                    new_string = new_string[:r] + temp_char + new_string[r+1:]
                    r-=1
                else: break
            else:
                l+=1
        return new_string
    
    def max_depth_test(s):
        s.test(s.maxDepth(TreeNode.init_tree([3,9,20,None,None,15,7])),3)
        s.test(s.maxDepth(TreeNode.init_tree([1,None,2])),2)
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
        s.test(s.reverse(123),321)
        s.test(s.reverse(-123),-321)
        s.test(s.reverse(120),21)
        s.test(s.reverse(0),0)
        s.test(s.reverse(1),1)
        s.test(s.reverse(-1),-1)
        s.test(s.reverse(123456789),987654321)
        s.test(s.reverse(900000),9)
        s.test(s.reverse(1534236469),0)
    def reverse(self, x: int) -> int:
        if not (-2**31 <= x <= 2**31 - 1): return 0
        ans = int(str(x)[::-1].replace('-',''))
        if x < 0: ans *= -1
        if not (-2**31 <= ans <= 2**31 - 1): return 0
        return ans