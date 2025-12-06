from gpiozero import LED, Button
from signal import pause

led1 = LED(17)
led2 = LED(27)
led3 = LED(22)
button = Button(26)

led1.off()
led2.off()
led3.off()

led_index = 0

def switch_led(led_index=0):
    global led_index
    if led_index == 0:
        led1.on()
        led2.off()
        led3.off()
        led_index += 1
    elif led_index == 1:
        led1.off()
        led2.on()
        led3.off()
        led_index += 1
    else:
        led1.off()
        led2.off()
        led3.on()
        led_index = 0

button.when_pressed = switch_led

pause()