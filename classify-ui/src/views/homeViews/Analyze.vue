<template>
  <v-container fluid>
    <v-row>
      <v-col cols="6">
        <v-card>
          <v-container>

            <v-row no-gutters>
              <v-col cols="5">
                <v-combobox
                  auto-select-first
                  v-model="selectedClassification"
                  label="Select Dataset"
                  :items="classificationList"
                  item-text="name"
                  item-value="id"
                  return-object>
                </v-combobox>
              </v-col>
              <v-col cols="2"></v-col>
              <v-col cols="5">
                <v-combobox
                  v-if="selectedClassification"
                  v-model="selectedClassificationModel"
                  label="Select Model"
                  :items="classificationModelsList"
                  item-text="name"
                  item-value="id"
                  return-object>
                </v-combobox>
              </v-col>
            </v-row>

            <v-row
              ref="classificationImagesListContainer"
              style="height: calc(100vh - 220px)" 
              class="overflow-y-auto">
              <v-col
                v-for="image in classificationImagesList"
                :key="image.id"
                cols="3">
                <v-card flat tile>
                  <v-img
                    class="classificationImage"
                    :src="BASE_URL+image.thumbnail"
                    :lazy-src="BASE_URL+image.thumbnail"
                    aspect-ratio="1"
                    @click="selectImage(image)">
                  </v-img>
                </v-card>
              </v-col>
              <v-col cols="12" 
                v-show="classificationImagesListLoader">
                <div class="text-center">
                  <v-progress-circular
                    indeterminate
                    color="primary">
                  </v-progress-circular>
                </div>
              </v-col>
            </v-row>
          </v-container>
        </v-card>
      </v-col>
      <v-col cols="6">
          <v-row no-gutters>
            <v-col>
              <v-card
                align="center"
                justify="center"
                style="position: relative;height: calc(100vh - 340px)">
                <v-container>
                  <div id="normalViewImageContainer">
                    <ImageViewer
                      v-if="selectedImage.image"
                      :selected-model-id="selectedClassificationModel.id"
                      :selected-image="selectedImage">
                    </ImageViewer>
                    <div v-else>
                      <div>Select an image from the list</div>
                      <!-- <div>OR</div>
                      <div>Upload Image</div> -->
                    </div>
                  </div>
                  <!-- <v-btn
                    absolute
                    dark
                    fab
                    bottom
                    right
                    color="light blue"
                    style="bottom: 16px"
                    @click="$refs.inputUpload.click()">
                    <v-icon>mdi-upload</v-icon>
                  </v-btn>
                  <input
                    v-show="false"
                    ref="inputUpload" 
                    type="file" 
                    @change="onImageChange($event)"> -->
                    <v-btn
                      absolute
                      dark
                      fab
                      bottom
                      right
                      color="light blue"
                      style="bottom: 16px"
                      v-if="selectedImage.image"
                      @click="toggleDialog(true)">
                      <v-icon>mdi-fullscreen</v-icon>
                    </v-btn>
                </v-container>
              </v-card>
            </v-col>
          </v-row>
          <v-row>
            <v-col>
              <v-card>
                <v-tabs v-model="selectedTab" vertical>
                  <v-tab>Related</v-tab>
                  <v-tab>Patient Info</v-tab>
                  <v-tabs-items v-model="selectedTab">
                    <v-tab-item>
                      <v-card flat>
                        <v-row
                          style="height: 200px; width: 102%" 
                          class="overflow-y-auto">
                          <v-col
                            v-for="image in similarImagesList"
                            :key="image.id"
                            cols="2" 
                            v-show="!similarImagesListLoader">
                            <v-card flat tile>
                              <v-img
                                class="similarImagesList"
                                :src="BASE_URL+image.thumbnail"
                                :lazy-src="BASE_URL+image.thumbnail"
                                aspect-ratio="1"
                                @click="selectImage(image)">
                              </v-img>
                            </v-card>
                          </v-col>
                          <v-col cols="12" 
                            v-show="similarImagesListLoader">
                            <div class="text-center">
                              <v-progress-circular
                                indeterminate
                                color="primary">
                              </v-progress-circular>
                            </div>
                          </v-col>
                        </v-row>
                      </v-card>
                    </v-tab-item>
                    <v-tab-item>
                      <v-card flat>
                        <v-simple-table 
                          id="patientInfo" 
                          height="200"
                          fixed-header>
                          <thead>
                            <tr>
                              <th class="font-weight-bold" colspan="2">Patient Details</th>
                            </tr>
                          </thead>
                          <tbody>
                            <tr v-for="(value, name) in selectedImage.meta" :key="name">
                              <td class="font-weight-bold">{{ name }}:</td>
                              <td>{{ value }}</td>
                            </tr>
                          </tbody>
                        </v-simple-table>
                      </v-card>
                    </v-tab-item>
                  </v-tabs-items>
                </v-tabs>

              </v-card>
            </v-col>
          </v-row>
      </v-col>

      <v-dialog 
        v-model="showDialog" 
        fullscreen 
        hide-overlay 
        transition="dialog-bottom-transition">
        <v-card
          align="center"
          justify="center">
          <v-toolbar dark color="primary">
            <v-toolbar-title>Full Screen</v-toolbar-title>
            <div class="flex-grow-1"></div>
            <v-btn icon dark @click="toggleDialog(false)">
              <v-icon>mdi-close</v-icon>
            </v-btn>
          </v-toolbar>
          <div id="fullScreenViewImageContainer">
            <ImageViewer
              v-if="selectedImageForFullScreen.image"
              :selected-model-id="selectedClassificationModel.id"
              :selected-image="selectedImageForFullScreen">
            </ImageViewer>
          </div>
        </v-card>
      </v-dialog>
    </v-row>
  </v-container>
</template>

<script>
import ImageViewer from '../../components/ImageViewer'
import { mapState } from 'vuex'

const SELECTED_IMAGE_DEFAULT_OBJECT = {
      image: null,
      id: -1,
      meta: {
        name: ''
        , age: ''
        , dateOfBirth: ''
        , gender: ''
        , bloodType: ''
      }
    }

export default {
  components: {
    ImageViewer
  },
  mounted () {
    this.$refs.classificationImagesListContainer.addEventListener('scroll', (event) => {
      if(event.srcElement.scrollTop + event.srcElement.offsetHeight >= event.srcElement.scrollHeight && !this.classificationImagesListLoader){
        this.getImagesList(this.$router.currentRoute.params.classificationId, false);
      }
    });

    this.$store.dispatch('getClassificationList')
    .then((result) => {
      this.selectedClassification = result;
      if(this.$router.currentRoute.params.classificationId == undefined){
        this.$store.commit('resetData');
      }
    });
  },
  methods: {
    updateRoute(_type, _id){
      switch(_type){
        case 'classification':{
          this.$router.push({ name: 'analyzeClassification', params: { classificationId: _id } });
          break;
        }
        case 'model':{
          if(this.$router.currentRoute.params.imageId){
            this.$router.push({ name: 'analyzeImage', params: { modelId: _id, imageId: this.$router.currentRoute.params.imageId } });
            this.$store.dispatch('getSimilarImages', {
              imageId: this.$router.currentRoute.params.imageId
              , modelId: _id
            });
          }else{
            this.$router.push({ name: 'analyzeModel', params: { modelId: _id } });
          }
          break;
        }
        case 'image':{
          this.$router.push({ name: 'analyzeImage', params: { imageId: _id } });
          break;
        }
      }
    },
    onImageChange(event) {
      let reader = new FileReader()
      reader.onload = () => {
        this.selectedImage = SELECTED_IMAGE_DEFAULT_OBJECT;
        this.selectedImage.image = reader.result;
      }
      reader.readAsDataURL(event.target.files[0])
    },
    getImagesList(_classificationId, _reloadControls){
      if(_reloadControls && this.$router.currentRoute.params.classificationId != _classificationId){
        this.updateRoute('classification', _classificationId);
      }
      this.$store.dispatch('getClassificationImagesList', _classificationId).then((result) => {
        if(_reloadControls){
          this.selectedClassificationModel = (result.selectedClassificationModel ? result.selectedClassificationModel : this.classificationModelsList[0]);
          this.selectImage(result.selectedImage);
          this.changeModel(this.selectedClassificationModel.id);
        }
      });
    },
    changeModel(_modelId){
      if(this.$router.currentRoute.params.modelId != _modelId){
        this.updateRoute('model', _modelId);
      }
    },
    selectImage(_image){
      _image = _image || this.classificationImagesList[0];
      if(this.$router.currentRoute.params.imageId != _image.id){
        this.updateRoute('image', _image.id);
      }
      this.selectedImage = _image || SELECTED_IMAGE_DEFAULT_OBJECT;
    },
    toggleDialog(_value){
      this.showDialog = _value;
      setTimeout(() => {
        this.selectedImageForFullScreen = _value ? this.selectedImage : SELECTED_IMAGE_DEFAULT_OBJECT;
      }, 300);
    }
  },
  computed: {
    ...mapState([
      'BASE_URL'
      , 'classificationList'
      , 'classificationModelsList'
      , 'classificationImagesList'
      , 'classificationImagesListLoader'
      , 'similarImagesList'
      , 'similarImagesListLoader'
    ])
  },
  watch: {
    selectedClassification: {
      immediate: true,
      handler(value) {
        if(value && value.id){
          this.getImagesList(value.id, true);
        }else{
          this.$store.commit('resetData');
        }
      }
    },
    selectedClassificationModel: {
      immediate: true,
      handler(value) {
        if(value){
          this.changeModel(value.id);
        }
      }
    }
  },
  data: () => ({
    selectedClassification: null
    , selectedClassificationModel: null
    , selectedImage: SELECTED_IMAGE_DEFAULT_OBJECT
    , selectedImageForFullScreen: SELECTED_IMAGE_DEFAULT_OBJECT
    , selectedTab: null
    , showDialog: false
  })
};
</script>

<style lang="less" scoped>
  #normalViewImageContainer{
    height: calc(100vh - 360px);
  }
  #fullScreenViewImageContainer{
    height: calc(100vh - 64px);
  }
  .classificationImage, .similarImagesList {
    cursor: pointer;
  }
  #patientInfo{
    thead{
      tr{
        th{
          font-size: 16px;
        }
      }
    }
    tbody{
      tr{
        td{
          &:nth-child(1){
            width: 200px;
          }
          border-bottom: 0px;
        }
      }
    }
  }
</style>