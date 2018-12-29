def base36_decode(s):
    x = 0
    power = 1
    for c in s:
        if '0' <= c <= '9':
            digit = int(c)
        elif 'a' <= c <= 'z':
            digit = ord(c) - ord('a')
        else:
            raise RuntimeError(s)
        x += digit * power
        power *= 36
    return x

def str_encode(s):
    pass
