import argparse
from sensorfusion import *
from Emergency import *
from payloadLog_saver import *
from Azure_handle import *
from AI_Module_layer import *
import pose_training
import concurrent.futures

def retreive_sensor_data_wrapper():

    with concurrent.futures.ProcessPoolExecutor() as exec :
        process1 = exec.submit(retreive_sensor_Data)
        Payloads = process1.result()
    return Payloads

if __name__ == '__main__':





    parser = argparse.ArgumentParser(description='Elderly pose and health processing application')
    parser.add_argument('--collect', type=str, help='Collect labeled pose sample into dataset')
    parser.add_argument('--samples', type=int, default=1, help='Number of samples to collect when using --collect')
    parser.add_argument('--dataset', type=str, default='pose_dataset.csv', help='Dataset CSV path')
    parser.add_argument('--model', type=str, default='pose_model.pkl', help='Saved model path')
    parser.add_argument('--train', action='store_true', help='Train pose classifier from dataset CSV')
    args = parser.parse_args()

    if args.collect:
        classifier = pose_training.Mediapipe_Class.frame_classifier()
        for i in range(args.samples):
            print(f'Collecting sample {i + 1}/{args.samples} for label {args.collect}')
            sequence = classifier.get_sequence(count=4)
            sample = pose_training.save_training_sample(sequence, args.collect, args.dataset)
            print('Saved sample:', sample)
        print('Collection complete. Run python main.py --train to create a model.')
        exit(0)

    if args.train:
        print('Training model from dataset:', args.dataset)
        model, df = pose_training.train_from_dataset(args.dataset, args.model)
        print('Training complete. Saved model to', args.model)
        print('Classes:', model.classes_)
        exit(0)

    # Initialize layers for pose and health processing and AZURE
    ai_layer = AI_LAYER(model_path=args.model)
    IOT_client = AzureAdmin_IOT()
    Database_client = AzureAdmin_DATABASE()
    

    
    while(True):
        #getting sensors Data
        payload = retreive_sensor_data_wrapper()
        
        # Get pose (AI_1)
        pose = ai_layer.Get_Elder_Pose()
        # print(pose)

        # Sensor fusion: combine pose with sensor data
        fused_payload = fuse_data_with_pose(payload, pose)
        # print(fused_payload)

        # # Get health status (AI_2)
        health_status = ai_layer.Get_Elder_status(fused_payload)
        # print(health_status)

        # Get stroke risk assessment
        stroke_risk = ai_layer.Get_Stroke_Risk()
        # print(stroke_risk)
    
        # # Emergency check
        emergency_flag = EmergencyALgorithm(fused_payload)
    
        # Payload parsing (prepare for Azure)
        parsed_payload = parse_payload(fused_payload, health_status, emergency_flag, stroke_risk)
    
        # # Send to Azure
        IOT_client.Initiate_Azure_connection_send(parsed_payload)
        Database_client.save_to_table(parsed_payload)    

        # Save logs
        save_logs(parsed_payload)
        time.sleep(600)
