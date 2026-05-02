import ollama ,sensorfusion,Mediapipe_Class

poses = ("INIT","SITTING","STANDING","LAYING ON THE FLOOR","LAYING ON THE BED")
Person_status = ('Normal','Not Normal','Not Found')

class AI_LAYER:
    def __init__(self,model: str = "llama3.2"):
        self.model = model
        self.old_pose = poses[0]
        self.person_pose = poses[0]
        self.frame_classifier_instance = Mediapipe_Class.frame_classifier()

        

    def Prompt_builder(self)->str :
        body_parts = self.frame_classifier_instance.get_whole_person()
        prompt = """act as a physiotherapist and a doctor and from the poses that're given in the form of x,y,z and presence as the result was taken from mediapose,
                    try to check whether this person is sitting normally or standing normally or falling down on the floor , you can use the previous pose for your reference"""
        
        full_prompt = prompt + "\n"f"the position for every part of the body : {body_parts}\n choose from the following poses : {poses}"
        return full_prompt
    
    def Get_Elder_Pose(self) -> str :
        self.old_pose = getattr(self,"person_pose",poses[0])
        current_pose =ollama.generate(model=self.model ,prompt= self.Prompt_builder())
        self.person_pose = current_pose.response
        self.old_pose = self.person_pose
        return self.person_pose
    
    
    def Get_Elder_status(self,sensor_grp_data : list )->str :
        final_prompt = f"""USING THE SENSOR DATA YOU HAVE HERE --> {str(sensor_grp_data)} tell me if the person is 'normal' or 'not normal' or 'not found' """
        AI_person_status = ollama.generate (self.model , prompt= final_prompt)
        sensorfusion.CurrentPAYLOAD["Person_status"] = AI_person_status.response
        return AI_person_status.response
"""testing    
# AI_INST = AI_LAYER()
# with open("texttryai.txt", "a") as fs:
#     fs.write(AI_INST.Get_Elder_Pose())
"""