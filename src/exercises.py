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
    def _static_print_list(head):
        head:ListNode = head
        while head:
            print(head.val)
            head = head.next
    
    @staticmethod
    def _static_get_list(head):
        head:ListNode = head
        list = []
        while head:
            list.append(head.val)
            head = head.next
        return list

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
        track_time_execution(s.double_it_test)
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
    