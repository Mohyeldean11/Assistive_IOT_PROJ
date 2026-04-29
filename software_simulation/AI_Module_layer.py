import ollama
import sensorfusion

poses = ("INIT","SITTING","STANDING","LAYING ON THE FLOOR","LAYING ON THE BED")
Person_status = ('Normal','Not Normal','Not Found')

class AI_LAYER:
    def __init__(self,model: str = "llama3.2"):
        self.model = model
        self.old_pose = poses[0]
        self.person_pose = poses[0]

        

    def Prompt_builder(self,person_pose :dict = {"eye": "here",'leg' : "not here"} )->str :
        full_prompt 
        old_pose_value = getattr(self,"old_pose",poses[0])
        prompt = """act as a physiotherapist and a doctor and from the poses that're given in the form of x,y,z and presence as the result was taken from mediapose,
                    try to check whether this person is sitting normally or standing normally or falling down on the floor , you can use the previous pose for your reference"""
        
        full_prompt = prompt + "\n"f"the new position for every part of the body : {person_pose}\n the old position for every part of the body : {old_pose_value}"
        self.old_pose =  person_pose
        return full_prompt
    
    def Get_Elder_Pose(self, full_prompt: str ) -> str :
        current_pose =ollama.generate(model=self.model ,prompt= full_prompt)
        return current_pose
    
    def Get_Elder_status(self,sensor_grp_data : list )->str :
        final_prompt = f"""USING THE SENSOR DATA YOU HAVE HERE --> {str(sensor_grp_data)} tell me if the person is 'normal' or 'not normal' or 'not found' """
        AI_person_status = ollama.generate (self.model , prompt= final_prompt)
        sensorfusion.PayloadGroup["person status"] = AI_person_status
        return AI_person_status