from typing import Optional
from linked_list import LinkedList

class main_exercises():
    def __init__(s):
        s.double_it_test()
        # pass

    def test(s,result,expected):
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
    def isHappy(self, n: int) -> bool:
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
        # s.test(s.doubleIt(),[])
        ll1 = LinkedList()
        ll2 = LinkedList()
        ll1.insert_list([1,2,3])
        ll2.insert_list([1,2,3])
        print(ll1 == ll2)
    def doubleIt(self, head: Optional[LinkedList]) -> Optional[LinkedList]:
        return None
