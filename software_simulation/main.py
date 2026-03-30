from sensorfusion import *
from azuresender import * # pyright: ignore[reportMissingImports]
from Emergency import *
import copy
import json, time,os

POLL_INTERVAL = 2
PayloadGroup = []
file = os.getcwd() + "\\file.json"



if __name__ == '__main__':
    print("Starting...")

    try:
        init_sensors()
        print("Sensors initialized")
    except Exception as e:
        print(f"Initialization error: {e}")

    try:
        for x in range(0, 3, 1):
            CurrentPAYLOAD = {}         
            print(f"Loop iteration {x}")
            CurrentPAYLOAD['deviceId'] = 'rpi-01'
            CurrentPAYLOAD['dht22'] = read_dht22()
            CurrentPAYLOAD['PIR501'] = read_PIR501()
            PayloadGroup.append(copy.deepcopy(CurrentPAYLOAD))
            print(f"Payload built: {CurrentPAYLOAD}")

            # message = json.dumps(PAYLOAD)
            # Initiate_Azure_connection_send(message)
            # print("Sent to IoT Hub")

            # save_to_table(PAYLOAD)
            # print("Saved to table")
            time.sleep(POLL_INTERVAL)

        # Read existing data if file exists
        if os.path.exists(file):
            with open(file, "r") as js:
                existing = json.load(js)
        else:
            existing = []

        # Merge and rewrite
        existing.extend(PayloadGroup)

        with open(file, "w") as js:
            json.dump(existing, js, indent=2)
            
        # printing the values
        print(EmergencyALgorithm(PayloadGroup))

    except Exception as e:
        print(f"Runtime error: {e}")