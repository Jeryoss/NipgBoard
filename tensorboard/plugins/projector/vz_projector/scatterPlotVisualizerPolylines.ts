/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
namespace vz_projector {

const RGB_NUM_ELEMENTS = 3;
const XYZ_NUM_ELEMENTS = 3;

/**
 * Renders polylines that connect multiple points in the dataset.
 */
export class ScatterPlotVisualizerPolylines implements ScatterPlotVisualizer {
  private dataSet: DataSet;
  private scene: THREE.Scene;
  private polylines: THREE.Line[];
  private greenColors: any;
  private redColors: any;
  private polylinePositionBuffer:
      {[polylineIndex: number]: THREE.BufferAttribute} = {};
  private polylineColorBuffer:
      {[polylineIndex: number]: THREE.BufferAttribute} = {};

  private updateSequenceIndicesInDataSet(ds: DataSet) {
    for (let i = 0; i < ds.sequences.length; i++) {
      const sequence = ds.sequences[i];
      for (let j = 0; j < sequence.pointIndices.length - 1; j++) {

        var j1 = sequence.pointIndices[j],
            j2 = sequence.pointIndices[j+1],
            c = 0;
        while(c < ds.points.length) {
          if (ds.points[c].metadata["_sync_id"] == j1) {
            j1 = c;
          }
          if (ds.points[c].metadata["_sync_id"] == j2) {
            j2 = c;
          }
          c = c + 1;
        }

        ds.points[j1].sequenceIndex = i;
        ds.points[j2].sequenceIndex = i;
      }
    }
  }

  private createPolylines(scene: THREE.Scene) {
    if (!this.dataSet || !this.dataSet.sequences) {
      return;
    }

    this.updateSequenceIndicesInDataSet(this.dataSet);
    this.polylines = [];

    for (let i = 0; i < this.dataSet.sequences.length; i++) {
      const geometry = new THREE.BufferGeometry();
      geometry.addAttribute('position', this.polylinePositionBuffer[i]);
      if (this.dataSet.sequences[i].color == 1) {
        geometry.addAttribute('color', new THREE.BufferAttribute(this.greenColors,3));
      } else {
        geometry.addAttribute('color', new THREE.BufferAttribute(this.redColors,3));
      }

      const material = new THREE.LineBasicMaterial({
        linewidth: 1,  // unused default, overwritten by width array.
        opacity: 1.0,  // unused default, overwritten by opacity array.
        transparent: true,
        vertexColors: THREE.VertexColors
      });

      const polyline = new THREE.LineSegments(geometry, material);
      polyline.frustumCulled = false;
      this.polylines.push(polyline);
      scene.add(polyline);
    }
  }

  dispose() {
    if (this.polylines == null) {
      return;
    }
    for (let i = 0; i < this.polylines.length; i++) {
      this.scene.remove(this.polylines[i]);
      this.polylines[i].geometry.dispose();
    }
    this.polylines = null;
    this.polylinePositionBuffer = {};
    this.polylineColorBuffer = {};
  }

  setScene(scene: THREE.Scene) {
    this.scene = scene;
  }

  setDataSet(dataSet: DataSet) {
    this.dataSet = dataSet;
  }

  onPointPositionsChanged(newPositions: Float32Array) {
    if ((newPositions == null) || (this.polylines != null)) {
      this.dispose();
    }
    if ((newPositions == null) || (this.dataSet == null)) {
      return;
    }
    // Set up the position buffer arrays for each polyline.
    for (let i = 0; i < this.dataSet.sequences.length; i++) {
      let sequence = this.dataSet.sequences[i];
      const vertexCount = 2 * (sequence.pointIndices.length - 1);

      this.greenColors = new Uint8Array(vertexCount * RGB_NUM_ELEMENTS);
      this.redColors = new Uint8Array(vertexCount * RGB_NUM_ELEMENTS)
      //for(var color of this.greenColors) {
      //  color = 100;
      //}

      var ii = 0;
      for(var color in this.greenColors) {
        if (ii == 0) {
          this.greenColors[color] = 0;
          ii++;
        } else if (ii == 1) {
          this.greenColors[color] = 100;
          ii++;
        } else {
          this.greenColors[color] = 0;
          ii = 0;
        }
      }
      var ii = 0;
      for(var color in this.redColors) {
        if (ii == 0) {
          this.redColors[color] = 100;
          ii++;
        } else if (ii == 1) {
          this.redColors[color] = 0;
          ii++;
        } else {
          this.redColors[color] = 0;
          ii = 0;
        }
      }
      /*
      for(i = 0; i < this.greenColors.length; i=i+3) {
        this.greenColors[i] = 51;
        this.greenColors[i+1] = 255;
        this.greenColors[i+2] = 32;
      }*/

      let polylines = new Float32Array(vertexCount * XYZ_NUM_ELEMENTS);
      this.polylinePositionBuffer[i] =
          new THREE.BufferAttribute(polylines, XYZ_NUM_ELEMENTS);

      let colors = new Float32Array(vertexCount * RGB_NUM_ELEMENTS);
      this.polylineColorBuffer[i] =
          new THREE.BufferAttribute(colors, RGB_NUM_ELEMENTS);
    }
    for (let i = 0; i < this.dataSet.sequences.length; i++) {
      const sequence = this.dataSet.sequences[i];
      let src = 0;
      for (let j = 0; j < sequence.pointIndices.length - 1; j++) {
        const p1Index = sequence.pointIndices[j];
        const p2Index = sequence.pointIndices[j + 1];
        const p1 = util.vector3FromPackedArray(newPositions, p1Index);
        const p2 = util.vector3FromPackedArray(newPositions, p2Index);
        this.polylinePositionBuffer[i].setXYZ(src, p1.x, p1.y, p1.z);
        this.polylinePositionBuffer[i].setXYZ(src + 1, p2.x, p2.y, p2.z);
        src += 2;
      }
      this.polylinePositionBuffer[i].needsUpdate = true;
    }

    if (this.polylines == null) {
      this.createPolylines(this.scene);
    }
  }

  onRender(renderContext: RenderContext) {
    if (this.polylines == null) {
      return;
    }
    for (let i = 0; i < this.polylines.length; i++) {
      this.polylines[i].material.opacity = renderContext.polylineOpacities[i];
      (this.polylines[i].material as THREE.LineBasicMaterial).linewidth =
          renderContext.polylineWidths[i];
      (this.polylineColorBuffer[i] as any)
          .setArray(renderContext.polylineColors[i]);
      this.polylineColorBuffer[i].needsUpdate = true;
    }
  }

  onPickingRender(renderContext: RenderContext) {}
  onResize(newWidth: number, newHeight: number) {}
}

}  // namespace vz_projector
