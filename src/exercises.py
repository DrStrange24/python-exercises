class main_exercises():
    def __init__(s):
        
        pass

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