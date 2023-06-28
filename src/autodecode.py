import base64

# str = "..." # 何らかのbase64エンコードされた文字列を代入
str = 'rAEAAAAAAABEMyIRFAAAAAAADgAKAAQAAAAAAAAACQAOAAAAZAAAAAABXgAcABAADAAAAAAAAAAAAAAAAAAAAAAAGAAXAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACAAAAAAAAAAHAF4AAAAAAAABFAAAACQAAABEAAAAAAAAARAAAABg////ZP///2j///9G////AAIIJgEAAAAEAAAAFgAAAE5vdG9TZXJpZkpQLUV4dHJhTGlnaHQAAAEAAAAMAAAACAAMAAQACAAIAAAAjAAAADgAAAA0ABgAAAAAABQAAAAQAA8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACAAAAAQANAAAABgAAAAcAAAAAAAAASQAAAA0AAAABAAEAAQAAAAEAAYABAAAAAAACgAKAAcACAAJAAoAAAAAAAACCCYKAAgABQAGAAcACgAAAADq3c80AAAA5aCA5YWD44GV44KT44Go44GE44GG5Lq66ZaT44GoDeWRs+immuOBjOS8vOOBpuOBhOOCiwAAAAA='

decodedddd = base64.b64decode(str)
print('<hr>')
print('<h1>Original</h1>')
print(decodedddd.decode())
print('<hr>')
print(f'Original length:{str(len(decodedddd))}')
print('<hr>')
newDecoded = b''
for i in range(len(decodedddd)):
    value = bytes([decodedddd[i]]).hex()
    converted = bin(int(value, 16))[2:].zfill(8)
    #print(str(i) + ') ' + str(decodedddd[i]) + ' [' + converted + ']')
    #print('<hr>')
    if (converted != '00000000'):
        newDecoded += bytes([decodedddd[i]])
        
newDecoded = newDecoded.decode()
print('<h1>Binary padding removed</h1>')
print(newDecoded)
print('<hr>')
print('New length: ' + str(len(newDecoded)))
print('<hr>')
newDecodedReplaced = newDecoded.replace('Alice', 'Ines')
print('<h1>Binary padding removed and string replaced</h1>')
print(newDecodedReplaced)
newBase64 = base64.b64encode(newDecodedReplaced.encode())
print('<hr>')
print('<h1>Binary padding removed and string replaced to base64</h1>')
print(newBase64.decode())



