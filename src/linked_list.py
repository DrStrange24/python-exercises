from typing import Optional

class LinkedList:
    class Node:
        def __init__(s, val = None):
            s.val = val
            s.next: Optional[LinkedList.Node] = None

    head: Optional[Node]

    def __init__(s):
        s.head = None

    def insert_list(s,list:list):
        if not list:
            return
        
        if s._private_is_empty():
            s.head = s.Node(list[0])
            h = s.head
            list = list[1:]
        else:
            h = s.head
            while h.next:
                h = h.next

        for x in list:
            h.next = s.Node(x)
            h = h.next

    def print_list(s):
        s._private_check_for_empty_list()

        h = s.head
        while h:
            print(h.val)
            h = h.next

    def _private_is_empty(s):
        return s.head is None
    
    def _private_check_for_empty_list(s):
        if s._private_is_empty():
            print('no data')
            return
    
    def get_list(s):
        s._private_check_for_empty_list()

        arr = []
        cur_node = s.head
        while cur_node:
            arr.append(cur_node.val)
            cur_node = cur_node.next
        return arr

    def insert_head(s, val):
        if s._private_is_empty():
            s.head = s.Node(val)
        else:
            new_head = s.Node(val)
            new_head.next = s.head
            s.head = new_head

    def insert_tail(s, val):
        if s._private_is_empty():
            s.head = s.Node(val)
        else:
            new_tail = s.Node(val)
            cur_node = s.head
            while cur_node.next:
                cur_node = cur_node.next
            cur_node.next = new_tail
    
    def insert_index(s, val, index):
        if s._private_is_empty():
            s.head = s.Node(val)
            print('inserted to head since list is empty')
        else:
            new_node = s.Node(val)
            cur_node = s.head
            for _ in range(index-1):
                cur_node = cur_node.next
            new_node.next = cur_node.next
            cur_node.next = new_node

    def get_head(s):
        s._private_check_for_empty_list()
        return s.head

    def get_tail(s):
        s._private_check_for_empty_list()
        cur_node = s.head
        while cur_node.next:
            cur_node = cur_node.next
        return cur_node
    
    def get_index(s, index):
        s._private_check_for_empty_list()
        cur_node = s.head
        for _ in range(index):
            cur_node = cur_node.next
        return cur_node

    def update_head(s, val):
        s._private_check_for_empty_list()
        s.head.val = val

    def update_tail(s, val):
        s._private_check_for_empty_list()
        cur_node = s.head
        while cur_node.next:
            cur_node = cur_node.next
        cur_node.val = val
    
    def update_index(s, val, index):
        s._private_check_for_empty_list()
        cur_node = s.head
        for _ in range(index):
            cur_node = cur_node.next
        cur_node.val = val

    def delete_head(s):
        s._private_check_for_empty_list()
        s.head = s.head.next

    def delete_tail(s):
        s._private_check_for_empty_list()
        cur_node = s.head
        while cur_node.next.next:
            cur_node = cur_node.next
        cur_node.next = None
    
    def delete_index(s, index):
        s._private_check_for_empty_list()
        cur_node = s.head
        for _ in range(index-1):
            cur_node = cur_node.next
        cur_node.next = cur_node.next.next

def practice_linked_list():
    # my linked list
    mll = LinkedList() 

    arr1 = range(5)
    arr2 = range(3)

    mll.insert_list(arr1)
    mll.insert_list(arr2)

    print(mll.get_list())

    mll.insert_head(6)
    mll.insert_tail(7)
    mll.insert_index(8,2)

    print(mll.get_list())
    print(mll.get_head().val)
    print(mll.get_tail().val)
    print(mll.get_index(2).val)

    mll.update_head(9)
    mll.update_tail(10)
    mll.update_index(11,2)
    print(mll.get_list())

    mll.delete_head()
    mll.delete_tail()
    mll.delete_index(1)
    print(mll.get_list())