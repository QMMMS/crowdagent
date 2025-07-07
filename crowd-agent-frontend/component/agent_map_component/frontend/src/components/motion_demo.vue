<script setup>  
import { DotLottieVue } from '@lottiefiles/dotlottie-vue'
import FinanceAgentLottie from '../assets/finance.lottie?url'
import errorSample from'../assets/errorSample.lottie?url'
import userGif from'../assets/user.lottie?url'
import scheduleAgent from'../assets/schedule.lottie?url'
import qualityAgent from'../assets/quality.lottie?url'
import annotateAgent from'../assets/annotate.lottie?url'
import annotatorProfileUpdate from'../assets/profile.lottie?url'
import {ref, watch, onMounted, onUpdated, nextTick} from 'vue'
import { useStreamlit } from "../streamlit"

const props=defineProps({
    agentNames:{
        type:Array,
        default:()=>[]
    },
    width:Number,
    disable:Boolean
})

const vMarigin = 5
const hMarigin = 10
const containerWidth = ref(0)
const agentMapRef = ref(0)
const highLightAgent =ref(null)

const scrollToRight = () =>{
    nextTick(()=>{
        if(agentMapRef.value){
            agentMapRef.value.scrollLeft = agentMapRef.value.scrollWidth
        }
    })
}

const getStreamlitContainerWidth = () => {
    try {
        const width = window.innerWidth || document.documentElement.clientWidth
        console.log('使用iframe宽度:', width)
        return Math.max(200, width - 10)
    } catch (error) {
        console.log('无法获取iframe宽度:', error)
        return 600
    }
}

const updateContainerWidth = () => {
    containerWidth.value = getStreamlitContainerWidth()
}

const StyledComponent = () => {
    let componentWidth
    
    if (props.width && props.width > 0) {
        componentWidth = props.width
    } else if (containerWidth.value > 0) {
        const extraSpace = 10
        componentWidth = Math.max(200, containerWidth.value-extraSpace)
    } else {
        componentWidth = 600
    }
    
    const styles = {
        width: componentWidth + 'px',
        margin: `${vMarigin}px ${hMarigin}px`,
        padding: '10px',
        boxSizing: 'border-box',
        maxWidth: '100%'
    }
    
    console.log('组件计算宽度:', componentWidth, '容器宽度:', containerWidth.value)
    return styles
}

const allAgents= [
    {"key": 1, "name": "Scheduling Agent", "message": "Scheduling Agent Launched","image":scheduleAgent,"below_name":"🤖Scheduling Agent"},
    {"key": 2, "name": "Financing Agent", "message": "Financing Agent Launched","image":FinanceAgentLottie,"below_name":"💰Financing Agent"},
    {"key": 3, "name": "Wrong Sample Analysis", "message": "Wrong Sample Analysis Launched","image":errorSample,"below_name":"📊Wrong Sample Analysis"},
    {"key": 4, "name": "QA Agent", "message": "QA Agent Launched","image":qualityAgent,"below_name":"🔍Quality Agent"},
    {"key": 5, "name": "Annotation Agent", "message": "Annotation Agent Launched","image":annotateAgent,"below_name":"🔧Annotation Agent"},
    {"key": 6, "name": "Annotator Profile Update", "message": "Annotator Profile Update Launched","image":annotatorProfileUpdate,"below_name":"👤Profile Update"}
]

const map_step=ref([
    {"key":0, "image":userGif, "name":"User", "message":"User Launched","below_name":"👤User"},
])

useStreamlit()

onMounted(() => {
    nextTick(() => {
        updateContainerWidth()
        
        const handleResize = () => {
            updateContainerWidth()
        }
        
        window.addEventListener('resize', handleResize)
        
        try {
            if (window.parent && window.parent !== window) {
                window.parent.addEventListener('resize', handleResize)
            }
        } catch (error) {
            console.log('无法监听父窗口resize事件:', error)
        }
        
        setTimeout(() => {
            updateContainerWidth()
        }, 100)
    })
})

onUpdated(() => {
    nextTick(() => {
        updateContainerWidth()
    })
    scrollToRight()
})

watch(() => props.width, (newWidth) => {
    console.log('Width prop changed:', newWidth)
}, { immediate: true })

watch(() => containerWidth.value, (newWidth) => {
    console.log('Container width changed:', newWidth)
}, { immediate: true })

watch(() => props.agentNames, (newList) => {
    const existingNames = map_step.value.slice(1).map(a => a.name)

    const newNames = newList.filter(name => !existingNames.includes(name))

    let lastAddedAgentIndex = -1
    let shouldEndHighlight = false
    
    newNames.forEach(name => {
        if (name === "end") {
            console.log("收到结束信号，将取消高亮")
            shouldEndHighlight = true
        } else {
            const agent = allAgents.find(a => a.name === name)
            if (agent) {
                console.log("添加新的智能体:", name)
                map_step.value.push(agent)
                lastAddedAgentIndex = map_step.value.length - 1
            } else {
                console.log("未找到该智能体:", name)
            }
        }
    })
    
    if (shouldEndHighlight) {
        highLightAgent.value = -1
        console.log("已取消高亮")
    } else if (lastAddedAgentIndex !== -1) {
        highLightAgent.value = lastAddedAgentIndex
        console.log("设置高亮智能体索引:", highLightAgent.value)
    }
}, { immediate: true})

watch(
    () => map_step.value.length,()=>{
        scrollToRight()
    }
)

</script>

<template>

    <div 
        ref ="agentMapRef"
        class="agent_map" 
        :style="StyledComponent()"
        >
        <template v-for="(agent, index) in map_step" :key="index">
            <div class="agent_item"
            :class="{active: index!=0 , highLight:highLightAgent==index}">
                <div class="agent_item_image">
                    <DotLottieVue :src="agent.image" :loop="true" :autoplay="true" :speed="1" :style="{width: '80px', height: '80px'}" />
                </div>
                <div class="agent_item_name">
                    {{ agent.below_name }}
                </div>
            </div>
            <div v-if="index < map_step.length - 1" class="arrow_connection">
                <svg width="80" height="40" viewBox="0 0 80 40">
                    <defs>
                        <marker id="arrowhead" markerWidth="10" markerHeight="7" 
                            refX="9" refY="3.5" orient="auto">
                            <polygon points="0 0, 10 3.5, 0 7" fill="#4A90E2" />
                        </marker>
                    </defs>
                    <line x1="5" y1="10" x2="60" y2="10" 
                        stroke="#4A90E2" stroke-width="2" 
                        marker-end="url(#arrowhead)" />
                </svg>
            </div>
        </template>
    </div>
</template>

<style scoped>
.agent_map {
    display: flex;
    justify-content: flex-start;
    align-items: center;
    height: 160px;
    overflow-x: auto;
    overflow-y: hidden;
    background-color: #f9f9f9;
    border: 1px solid #ddd;
    border-radius: 10px;
    min-width: 200px;
}
.agent_item.active{
    display: flex;
    flex-direction: column; 
    align-items: center;
    justify-content: center;
    animation: fadeInAgent 0.8s ease;
    transition: all 0.5s ease;
}
.agent_item.highLight{
    border: 2px solid #4A90E2;
    border-radius: 10px;
    box-shadow: 0 0 12px rgba(74, 144, 226, 0.6);
    background-color: #eaf4ff;
}

@keyframes fadeInAgent {
    from {
        opacity: 0;
        transform: scale(0.8);
    }
    to {
        opacity: 1;
        transform: scale(1);
    }
}
.agent_item_name{
    text-align: center;
    font-size: 15px;
}

.arrow_connection {
    display: flex;
    align-items: center;
    justify-content: center;
}

.arrow_connection svg {
    animation: fadeIn 0.4s ease;
    transition: opacity 0.3s ease;
    }

@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateX(-10px);
    }
    to {
        opacity: 0.3;
        transform: translateX(0);
    }
}
</style>
