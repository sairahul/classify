import Vue from 'vue'
import Vuex from 'vuex'
import axios from 'axios'
import router from '@/router'
import { find } from 'lodash'

import NavigationTreeList from './assets/jsons/NavigationTreeList.json'

Vue.use(Vuex)

export default new Vuex.Store({
  state: {
    BASE_URL: 'http://148.251.87.118:8000'
    , navigationTreeList: NavigationTreeList.data
    , classificationList: []
    , classificationModelsList: []
    , classificationImagesList: []
    , classificationImagesListLoader: false
    , similarImagesList: []
    , similarImagesListLoader: false
  },
  mutations: {
    resetData(state) {
      state.classificationModelsList = [];
      state.classificationImagesList = [];
      state.classificationImagesListLoader = false;
      state.similarImagesList = [];
      state.similarImagesListLoader = false;
    }
  },
  actions: {
    getClassificationList: ({ state }) => {
      return axios
      .get(`${state.BASE_URL}/api/v1/datasets/list`)
      .then(response => {
        state.classificationList = response.data;
        return find(state.classificationList, { id: parseInt(router.currentRoute.params.classificationId) });
      })
    },
    getClassificationImagesList: ({ state }, payload) => {
      state.classificationImagesListLoader = true;
      return axios
      .get(`${state.BASE_URL}/api/v1/datasets/browse/${payload}?skip=${state.classificationImagesList.length}`)
      .then(response => {
        state.classificationModelsList = response.data.models;
        state.classificationImagesList = [...state.classificationImagesList, ...response.data.items];
        state.classificationImagesListLoader = false;
        return {
          selectedClassificationModel: find(state.classificationModelsList, { id: parseInt(router.currentRoute.params.modelId) })
          , selectedImage: find(state.classificationImagesList, { id: parseInt(router.currentRoute.params.imageId) })
        }
      })
    },
    getSimilarImages: ({ state }, payload) => {
      state.similarImagesListLoader = true;
      return axios
      .get(`${state.BASE_URL}/api/v1/datasets/similar/${payload.modelId}/${payload.imageId}`)
      .then(response => {
        state.similarImagesList = response.data.items;
        state.similarImagesListLoader = false;
      })
    }
  },
  getters:{
    
  }
})
