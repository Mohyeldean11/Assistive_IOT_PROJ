import mediapipe as mp

class PoseProcessor :
    def __init__(self,model_path:str ="Case_study\Assistive_IOT_PROJ\pose_landmarker_lite.task" ):
        baseoptions = mp.tasks.BaseOptions
        self.poselandmarker = mp.tasks.vision.PoseLandmarker
        poselandmarker_options = mp.tasks.vision.PoseLandmarkerOptions
        poseVision_Running = mp.tasks.vision.RunningMode
        options = poselandmarker_options(base_options=baseoptions(model_asset_path = "Case_study\Assistive_IOT_PROJ\Helper_Models\pose_landmarker_lite.task"),
                                running_mode = poseVision_Running.IMAGE)
        