<template>
  <div>
    <v-content id="mainContent">
      <router-view/>
    </v-content>
  </div>
</template>

<script>

export default {
  data: () => ({ })
};
</script>

<style lang="less" scoped>
  #mainContent {
    padding: 64px 15px 0px 0px !important;
  }
</style>