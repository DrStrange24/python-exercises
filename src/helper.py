def string_replace_index(string:str,char:str,index:int):
    last_string_starting_index = (index if index >= 0 else len(string) + index) + 1
    return string[:index] + char + string[last_string_starting_index:]
