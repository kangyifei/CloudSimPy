import pyautogui, sys,time
print('Press Ctrl-C to quit.')
try:
    while True:
        x, y = pyautogui.position()
        positionStr = 'X: ' + str(x).rjust(4) + ' Y: ' + str(y).rjust(4)
        print(positionStr)
        # print('\b' * len(positionStr), end='')
        time.sleep(0.5)
except KeyboardInterrupt:
    print('\n')

#重新开始：1790 1018
#ok: 1132 554