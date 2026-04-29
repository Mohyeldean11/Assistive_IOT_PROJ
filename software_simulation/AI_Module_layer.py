import json,requests
import ollama

class AI_LAYER:
    def __init__(self,model: str = "llama3.2"):
        self.model = model

        

    def prompt_build(self,person_pose :dict = {"eye": "here",'leg' : "not here"} )->str :
        prompt = """act as a physiotherapist and a doctor and from the poses that're given in the form of x,y,z and presence,
                    try to check whether this person is in right position and in right health or he is suffering from a health proplem """
        
        full_prompt = prompt + "\n"f"the position for every part of the body : {person_pose}"

        return full_prompt
    
    def send_AI_request(self, full_prompt: str ) -> str :
        response =ollama.generate(model=self.model ,prompt= full_prompt,)
        return response

