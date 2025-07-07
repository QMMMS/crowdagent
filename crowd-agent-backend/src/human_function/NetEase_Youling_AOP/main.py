
# For more information, please refer to https://youling-platform.apps-hp.danlu.netease.com/docs

import os
import sdk
import asyncio
from typing import List, Set, Dict, Tuple, Optional, Callable, TypeVar, Generic, Any, Union, Literal
from fuxi.aop.core import AOP
import fuxi.aop.ddl.builtins as builtins
from fuxi.aop.ddl.builtins import Image
from fuxi.aop.edsl import State, Action, Observation, MemState
from sdk.aop_dataclasses import Lable

async def main():

    question_list = [
        ("./imgs/915973750276857856_0.jpg", "The world's only 747 supertanker is fighting California's wildfires right now. https://t.co/2XsL0eSnoJ https://t.co/MJuZk8Ds2q"),
        ("./imgs/919718726890545152_3.jpg", "Travelling Aid #HurricaneMaria Thanks a lot !!! https://t.co/EkAw0VfR6O"),
        ("./imgs/919719288377708545_0.jpg", "@realDonaldTrump People in Puerto Rico don't have electricity with which to read your tweet. https://t.co/fA7RAuHlkb https://t.co/5umzbJjuQw"),
        ("./imgs/919719914881142784_0.jpg", "Stop by your local @McD_Nashville's area to help Hurricane Harvey and Irma victims! Get all the information HERE:‚Ä¶ https://t.co/ukAuG3goJl"),
    ]

    aop = await AOP.init(task_type = sdk.Demo_Task, config = sdk.get_server_config("xxxxx"))  # TODO: replace with your server id
    agent = await aop.create_agent(sdk.Demo_Agent)


    for question in question_list[:1]:
        image = await builtins.Image.from_path_async(question[0],'jpg')
        a = Observation[Image](image)
        b = question[1]
        output = await agent.annotate(a,b)
        print(output)
        await asyncio.sleep(1)

if __name__ == '__main__':
    asyncio.get_event_loop().run_until_complete(main())

