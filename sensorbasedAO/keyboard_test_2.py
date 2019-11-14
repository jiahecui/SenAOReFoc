import sys
 
if sys.version_info[0] > 2:
    from msvcrt import getwch as getch
else:
    from msvcrt import getch
 
c1 = getch()
print(c1)
if c1 in ("\x00", "\xe0"):
    arrows = {"H": "up", "P": "down", "M": "right", "K": "left"}
    c2 = getch()
    print(arrows.get(c2, c1 + c2))
print(c1)