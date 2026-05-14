##emergency Algo

from sensorfusion import *


def EmergencyALgorithm(DataPayloadgroup:list )->bool:

    ReturnFlag = False
    movementflag = False
    for payloadent in DataPayloadgroup :
        # Check temperature
        if not (18 <= payloadent['dht22']['temperature_celsius'] <= 24):
            ReturnFlag = True
            print(f"bad temp: {payloadent['dht22']['temperature_celsius']}")
        else:
            print(f"good temp: {payloadent['dht22']['temperature_celsius']}")

        # Check humidity
        if not (40 <= payloadent['dht22']['humidity_percent'] <= 60):
            ReturnFlag = True
            print(f"bad humidity: {payloadent['dht22']['humidity_percent']}")
        else:
            print(f"good humidity: {payloadent['dht22']['humidity_percent']}")

        # Check motion
        if payloadent['PIR501']['value'] == 1 : 
            movementflag = True
            print("movement detected")

        # Check pose for fall detection
        pose = payloadent.get('pose', '')
        if pose == "LAYING ON THE FLOOR" and not movementflag:
            ReturnFlag = True
            print("fall detected: laying on floor with no motion")
    
    if not ReturnFlag and not movementflag: 
        ReturnFlag = True
        print("no movement detected")

    return ReturnFlag


