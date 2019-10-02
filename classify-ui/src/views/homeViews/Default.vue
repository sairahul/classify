<template>
  <div class="default">
    <h1>This is an Default page</h1>
  </div>
</template>

<script>

export default {
  mounted: function(){
    /* Hardcoded route to navigate to Analyze page */
    this.$router.push({ name: 'analyze', params: { id: 1011 } });
  },
  data: () => ({ })
};
</script>
