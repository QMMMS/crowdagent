from streamlit.components.v1 import components
import os


_RELEASE = True

if not _RELEASE:
    _component_func = components.declare_component(
        "map_nodes_component",
        url="http://localhost:5173",
    )
else:
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(parent_dir, "frontend/dist")
    _component_func = components.declare_component(
        "map_nodes_component", 
        path=build_dir
    )


def agent_map_component(agent_name, width=None, key=None):
    component_value = _component_func(
        agent_name=agent_name, 
        width=width,
        key=key, 
    )
    return component_value