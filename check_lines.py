with open('env_tensor.py', 'r') as f:
    lines = f.readlines()
    for i in range(835, 845):
        print(f'{i+1}: {repr(lines[i])}') 