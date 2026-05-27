import serial
import time

ser = serial.Serial('COM3', 9600, timeout=1)
time.sleep(2)

print("로봇팔 수동 제어 모드 (종료하려면 q 입력)")
print("입력 예시: 0,90,90,110 (집게, 회전, 어깨, 팔꿈치)")

while True:
    val = input("각도를 입력하세요: ")
    if val == 'q': break
    
    try:
        ee, j1, j2, j3 = val.split(',')
        # move_time을 1000으로 해서 부드럽게 움직이게
        cmd = f"<M,{ee},{j1},{j2},{j3},1000,1000,1000,1000>"
        ser.write(cmd.encode())
        print(f"이동 명령: {cmd}")
    except:
        print("입력 형식이 틀렸습니다. 다시 입력하세요.")