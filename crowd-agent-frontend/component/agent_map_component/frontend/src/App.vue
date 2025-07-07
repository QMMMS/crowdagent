<script setup>
import MotionDemo from './components/motion_demo.vue'
import WithStreamlitConnection from "./streamlit/WithStreamlitConnection.vue"
</script>

<template>
  <div class="app-container">
    <WithStreamlitConnection v-slot="{args}">
      <MotionDemo 
        :agentNames="args?.agent_name" 
        :width="args?.width"
      />
    </WithStreamlitConnection>
  </div>
</template>

<style scoped>
.app-container {
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 100vh;
  width: 100%;
}
</style>
