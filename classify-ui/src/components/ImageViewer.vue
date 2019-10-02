<template>
  <div id="viewContainer">
    <template v-if="selectedImageType === 'dcm'">
      <div ref="imageController" id="imageController"></div>
      <div ref="imageViewer" id="imageViewer"></div>
    </template>
    <v-img
      v-else-if="selectedImageType === 'png' || selectedImageType === 'jpg' || selectedImageType === 'jpeg'"
      :src="BASE_URL+selectedImage.image"
      :lazy-src="BASE_URL+selectedImage.image"
      contain>
    </v-img>
  </div>
</template>

<script>
import { mapState } from 'vuex'
import * as THREE from 'three';
import * as dat from 'dat.gui';
import {stackHelperFactory, VolumeLoader, orthographicCameraFactory, trackballOrthoControlFactory} from 'ami.js';

export default {
  props: {
    selectedImage: Object
    , selectedModelId: Number
  }, 
  mounted: function(){
    this.validateImageType();
  },
  computed: {
    ...mapState([
      'BASE_URL'
    ])
  },
  methods: {
    validateImageType(){
      this.selectedImageType = this.selectedImage.image.substring(this.selectedImage.image.lastIndexOf('.') + 1).toLowerCase();

      this.$store.dispatch('getSimilarImages', {
        imageId: this.selectedImage.id
        , modelId: this.selectedModelId
      });
      
      setTimeout(() => {
        if(this.selectedImageType === 'dcm') this.buildDicomData();
      }, 10);
    },
    buildDicomData(){
      const container = this.$refs.imageViewer;
      const file = `${this.BASE_URL}${this.selectedImage.image}`;

      const renderer = new THREE.WebGLRenderer({
        antialias: true,
      });
      //window.console.log("Container width: " + container.offsetWidth + " height: " + container.offsetHeight);
      renderer.setSize(container.offsetWidth, container.offsetHeight);
      renderer.setClearColor(0x353535, 1); // darkGrey
      renderer.setPixelRatio(window.devicePixelRatio);
      container.innerHTML = '';
      container.appendChild(renderer.domElement);

      const scene = new THREE.Scene();
      const OrthographicCamera = new orthographicCameraFactory(THREE); 
      const TrackballOrthoControl = new trackballOrthoControlFactory(THREE);
      const StackHelper = stackHelperFactory(THREE);

      const camera = new OrthographicCamera(
        container.clientWidth / -2,
        container.clientWidth / 2,
        container.clientHeight / 2,
        container.clientHeight / -2,
        0.1,
        10000
      );

      // Setup controls
      const controls = new TrackballOrthoControl(camera, container);
      controls.staticMoving = true;
      controls.noRotate = true;
      camera.controls = controls;
      
      const onWindowResize = () => {
        camera.canvas = {
          width: container.offsetWidth,
          height: container.offsetHeight,
        };
        camera.fitBox(2);

        renderer.setSize(container.offsetWidth, container.offsetHeight);
      };
      window.addEventListener('resize', onWindowResize, false);
     
      const loader = new VolumeLoader(container);
      loader
        .load(file)
        .then(() => {
          const series = loader.data[0].mergeSeries(loader.data);
          const stack = series[0].stack[0];
          loader.free();


          const stackHelper = new StackHelper(stack);
          stackHelper.bbox.visible = false;
          stackHelper.border.color = 0xff0000; // Red
          scene.add(stackHelper);

          gui(stackHelper);

          // center camera and interactor to center of bouding box
          // for nicer experience
          // set camera
          const worldbb = stack.worldBoundingBox();
          const lpsDims = new THREE.Vector3(
            worldbb[1] - worldbb[0],
            worldbb[3] - worldbb[2],
            worldbb[5] - worldbb[4]
          );

          const box = {
            center: stack.worldCenter().clone(),
            halfDimensions: new THREE.Vector3(lpsDims.x + 10, lpsDims.y + 10, lpsDims.z + 10),
          };

          // init and zoom
          const canvas = {
            width: container.clientWidth,
            height: container.clientHeight,
          };

          camera.directions = [stack.xCosine, stack.yCosine, stack.zCosine];
          camera.box = box;
          camera.canvas = canvas;
          camera.update();
          camera.fitBox(2);
        })
        .catch(error => {
          window.console.log('oops... something went wrong...');
          window.console.log(error);
        });

      const animate = () => {
        controls.update();
        renderer.render(scene, camera);

        requestAnimationFrame(function() {
          animate();
        });
      };

      animate();

      const gui = stackHelper => {
        const gui = new dat.GUI({
          autoPlace: false,
        });

        const customContainer = this.$refs.imageController;
        customContainer.innerHTML = '';
        customContainer.appendChild(gui.domElement);
        const camUtils = {
          invertRows: false,
          invertColumns: false,
          rotate45: false,
          rotate: 0,
          orientation: 'default',
          convention: 'radio',
        };

        // camera
        const cameraFolder = gui.addFolder('Camera');
        const invertRows = cameraFolder.add(camUtils, 'invertRows');
        invertRows.onChange(() => {
          camera.invertRows();
        });

        const invertColumns = cameraFolder.add(camUtils, 'invertColumns');
        invertColumns.onChange(() => {
          camera.invertColumns();
        });

        const rotate45 = cameraFolder.add(camUtils, 'rotate45');
        rotate45.onChange(() => {
          camera.rotate();
        });

        cameraFolder
          .add(camera, 'angle', 0, 360)
          .step(1)
          .listen();

        const orientationUpdate = cameraFolder.add(camUtils, 'orientation', [
          'default',
          'axial',
          'coronal',
          'sagittal',
        ]);
        orientationUpdate.onChange(value => {
          camera.orientation = value;
          camera.update();
          camera.fitBox(2);
          stackHelper.orientation = camera.stackOrientation;
        });

        const conventionUpdate = cameraFolder.add(camUtils, 'convention', ['radio', 'neuro']);
        conventionUpdate.onChange(value => {
          camera.convention = value;
          camera.update();
          camera.fitBox(2);
        });

        cameraFolder.open();

        const stackFolder = gui.addFolder('Stack');
        stackFolder
          .add(stackHelper, 'index', 0, stackHelper.stack.dimensionsIJK.z - 1)
          .step(1)
          .listen();
        stackFolder
          .add(stackHelper.slice, 'interpolation', 0, 1)
          .step(1)
          .listen();
        stackFolder.add(stackHelper.slice, 'invert');
        stackFolder.open();
      };
    }
  },
  watch: {
    selectedImage () {
      this.validateImageType();
    }
  },
  data: () => ({
    selectedImageType: null
  })
};
</script>

<style lang="less">
  #viewContainer{
    position: relative;
    height: 100%;
    #imageController{
      position: absolute;
      top: 10px;
      right: 10px;
      z-index: 3;
      ul, ol{
        padding-left: initial !important;
      }
    }
    #imageViewer{
      height: 100%;
      width: 100%;
      position: relative;
      .container{
        padding: 0px;
      }
    }
  }
</style>