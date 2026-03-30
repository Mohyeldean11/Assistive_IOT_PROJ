import time
import random

# PIR Raw digital readings
pir_output = {
    'value': 1,           # 1 = motion detected, 0 = no motion
    'gpio_pin': 3,

}

dht22_raw = {
    'gpio_pin': 4,
}

def Init_env_sensor():
    pass
def Init_PIR_sensor():
    pass
def Init_BUTTON_MECHANISM():
    pass



def init_sensors()->bool:
    try:
        Init_env_sensor()
        Init_PIR_sensor()
        Init_BUTTON_MECHANISM()
        return True
    except:
        print("failed to initialize the sensors")
        return False


def  read_PIR501()->dict:

    pir_output['timestmp'] = time.asctime()
    pir_output['value'] =  random.randrange(0,2)

    return pir_output



def  read_dht22()->dict:

    dht22_raw['timestmp'] = time.asctime()
    dht22_raw['temperature_celsius'] =  random.randrange(-3,28,2)
    dht22_raw['humidity_percent'] = random.randrange(20,100,5)
    return dht22_raw




