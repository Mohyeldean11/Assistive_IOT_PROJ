from sensorfusion import *
from Emergency import *
from payloadLog_saver import *
from Azure_handle import *
from AI_Module_layer import *


if __name__ == '__main__':

    # Retrieve raw sensor data
    Payloads = retreive_sensor_Data()
    
    # Initialize AI layer for pose and health processing
    ai_layer = AI_LAYER()
    
    # Get pose (AI_1)
    pose = ai_layer.Get_Elder_Pose()
    
    # Sensor fusion: combine pose with sensor data
    fused_payloads = fuse_data_with_pose(Payloads, pose)
    
    # Get health status (AI_2)
    health_status = ai_layer.Get_Elder_status(fused_payloads)
    
    # Emergency check
    emergency_flag = EmergencyALgorithm(fused_payloads)
    
    # Payload parsing (prepare for Azure)
    parsed_payloads = parse_payloads(fused_payloads, health_status, emergency_flag)
    
    # Send to Azure
    Currentclient = AzureAdmin_IOT()
    Currentclient.Initiate_Azure_connection_send(parsed_payloads)
    Currentclient = AzureAdmin_DATABASE()
    Currentclient.save_to_table(parsed_payloads)    
    
    # Save logs
    save_logs(parsed_payloads)
  
