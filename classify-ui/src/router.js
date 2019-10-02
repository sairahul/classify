import Vue from 'vue'
import Router from 'vue-router'
import Home from './views/Home.vue'
import Default from './views/homeViews/Default.vue'
import Analyze from './views/homeViews/Analyze.vue'
import Diagnose from './views/homeViews/Diagnose.vue'

Vue.use(Router)

export default new Router({
  mode: 'history',
  routes: [
    {
      path: '/',
      name: 'home',
      component: Home,
      children: [
        {
          name: 'analyze',
          path: 'analyze/:id',
          component: Analyze
        },
        {
          name: 'analyzeClassification',
          path: 'analyze/:id/:classificationId',
          component: Analyze
        },
        {
          name: 'analyzeModel',
          path: 'analyze/:id/:classificationId/:modelId',
          component: Analyze
        },
        {
          name: 'analyzeImage',
          path: 'analyze/:id/:classificationId/:modelId/:imageId',
          component: Analyze
        },
        {
          name: 'diagnose',
          path: 'diagnose/:id',
          component: Diagnose
        },
        { 
          path: "*",
          component: Default
        }
      ]
    },
    {
      path: '/about',
      name: 'about',
      // route level code-splitting
      // this generates a separate chunk (about.[hash].js) for this route
      // which is lazy-loaded when the route is visited.
      component: () => import(/* webpackChunkName: "about" */ './views/About.vue')
    },
    { 
      path: "*",
      redirect: '/'
    }
  ]
})
