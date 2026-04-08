##emergency Algo

from sensorfusion import *


def EmergencyALgorithm(DataPayloadgroup:list )->bool:

    ReturnFlag = False
    movementflag = False
    for payloadent in DataPayloadgroup :
        if not (18 <= payloadent['dht22']['Temp']<=24):
            ReturnFlag = True
            print(f"good temp{payloadent['dht22']['Temp']}")
            
        else:
            print(f"bad temp{payloadent['dht22']['Temp']}")

        if not (40<= payloadent['dht22']['humidity_percent'] <= 60):
            ReturnFlag = True
            print(f"bad humidity{payloadent['dht22']['humidity_percent']}")
        else :
            print(f"good humidity{payloadent['dht22']['humidity_percent']}")
        if payloadent['PIR501']['value'] == 1 : 
            movementflag =True
            print("movement detected")
    
    if(ReturnFlag != True  and movementflag == False): 
        ReturnFlag = True
        print("no movement detected")

    return ReturnFlag

