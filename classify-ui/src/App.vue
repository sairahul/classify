<template>
  <v-app>
    <v-app-bar app>
      <v-app-bar-nav-icon @click.stop="showNavigationDrawer = !showNavigationDrawer"></v-app-bar-nav-icon>
      <router-link to="/">
        <v-img src="./assets/logo.png" alt="Xplore" width="12%" height="12%" />
      </router-link>
      <v-spacer></v-spacer>
      <router-link to="/about">About</router-link>
    </v-app-bar>

    <v-navigation-drawer 
      style="min-height: 100vh; max-height: calc(100% - 64px);"
      v-model="showNavigationDrawer"
      absolute
      temporary>

      <v-list-item align="center" justify="center">
        <v-list-item-title>
          <router-link to="/">
            <v-img src="./assets/logo.png" alt="Xplore" width="50%" height="50%" />
          </router-link>
        </v-list-item-title>
      </v-list-item>

      <v-divider></v-divider>
      <v-list 
        nav
        expand>

        <v-list-group
          v-for="category in navigationTreeList"
          :key="category.categoryId"
          :value="true"
          no-action>
          <template v-slot:activator>
            <v-list-item-content>
              <v-list-item-title v-text="category.categoryName"></v-list-item-title>
            </v-list-item-content>
          </template>

          <v-list-item 
            v-model="listGroup"
            v-for="item in category.itemsList"
            :to="{ name: category.routePath, params: { id: item.itemId} }"
            :key="item.itemId">
              <v-list-item-content>
                <v-list-item-title v-text="item.itemName"></v-list-item-title>
              </v-list-item-content>
          </v-list-item>
        </v-list-group>

      </v-list>
    </v-navigation-drawer>
    <router-view/>
  </v-app>
</template>

<script>
import { mapState } from 'vuex'

export default {
  name: 'App',
  computed: {
    ...mapState([
      'navigationTreeList'
    ])
  },
  watch: {
    listGroup () {
      this.showNavigationDrawer = false
    },
  },
  data: () => ({ 
    showNavigationDrawer: false
    , listGroup: null
  })
};
</script>