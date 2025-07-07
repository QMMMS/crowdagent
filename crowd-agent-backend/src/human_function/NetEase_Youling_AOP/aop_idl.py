#!/usr/bin/env python3


# For more information, please refer to https://youling-platform.apps-hp.danlu.netease.com/docs


from fuxi.aop.edsl import *
from typing import List, Literal
from fuxi.aop.ddl.builtins import Image,Audio,Video
# The following code demonstrates defining a classification task through AOP IDL

''' Define the output options of the task '''
class Lable (IntEnum):
    Volunteer_Service_or_Donation_Activity = 0
    Infrastructure_and_Public_Utility_Damage = 1

class Demo_Task(Task):
    ''' Description of the Task '''

    class Demo_Agent(Agent):
        ''' Description of the Agent '''
        input_info: Observation[Image]    # The input for the classification task is of image type
        # input_info: Observation[Audio]  # The input for the classification task is of audio type
        # input_info: Observation[Video]  # The input for the classification task is of video type
        output_info: MemState[Lable] 
        text_info: Observation[str]

        @cognition(
            category = "Human",
            input = ['input_info', 'text_info'],
            output = ['output_info'],
            persistent = True,           # When persistent = True, the platform will automatically persist input and output

        )
        def annotate():
            ''' Annotation capability of the Agent '''

if __name__ == "__main__":
    main(Demo_Task)
