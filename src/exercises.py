class main_exercises():
    def __init__(self):
        self.test()
        

    def test(self):
        print('Hello world')

    def containsNearbyDuplicate(self, nums, k: int) -> bool:
        for i in range(len(nums)):
            if nums[i] in nums[i+1:]:
                j = nums[i+1:].index(nums[i])
                if abs(i-j) <= k:
                    return True
                
        return False