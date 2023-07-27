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

  const LABEL_FONT_SIZE = 10;
  const LABEL_SCALE_DEFAULT = 1.0;
  const LABEL_SCALE_LARGE = 2;
  const LABEL_FILL_COLOR_SELECTED = 0x000000;
  const LABEL_FILL_COLOR_HOVER = 0x000000;
  const LABEL_FILL_COLOR_NEIGHBOR = 0x000000;
  const LABEL_STROKE_COLOR_SELECTED = 0xFFFFFF;
  const LABEL_STROKE_COLOR_HOVER = 0xFFFFFF;
  const LABEL_STROKE_COLOR_NEIGHBOR = 0xFFFFFF;

  const POINT_COLOR_UNSELECTED = 0xE3E3E3;
  const POINT_COLOR_NO_SELECTION = 0x7575D9;
  const POINT_COLOR_SELECTED = 0xF5EF42;
  const POINT_COLOR_HOVER = 0xF5EF42;

  const POINT_SCALE_DEFAULT = 1.0;
  const POINT_SCALE_SELECTED = 1.2;
  const POINT_SCALE_NEIGHBOR = 1.2;
  const POINT_SCALE_HOVER = 1.2;

  const LABELS_3D_COLOR_UNSELECTED = 0xFFFFFF;
  const LABELS_3D_COLOR_NO_SELECTION = 0xFFFFFF;

  const SPRITE_IMAGE_COLOR_UNSELECTED = 0xFFFFFF;
  const SPRITE_IMAGE_COLOR_NO_SELECTION = 0xFFFFFF;

  const POLYLINE_START_HUE = 60;
  const POLYLINE_END_HUE = 360;
  const POLYLINE_SATURATION = 1;
  const POLYLINE_LIGHTNESS = .3;

  const POLYLINE_DEFAULT_OPACITY = 0.2;
  const POLYLINE_DEFAULT_LINEWIDTH = 4;
  const POLYLINE_SELECTED_OPACITY = 0.5;
  const POLYLINE_SELECTED_LINEWIDTH = 6;
  const POLYLINE_DESELECTED_OPACITY = .2;
  const POLYLINE_DESELECTED_LINEWIDTH = 1;

  const SCATTER_PLOT_CUBE_LENGTH = 2;

  /** Color scale for nearest neighbors. */
  const NN_COLOR_SCALE =
      d3.scaleLinear<string, string>()
          .domain([1, 0.7, 0.4])
          .range(['hsl(285, 80%, 40%)', 'hsl(0, 80%, 65%)', 'hsl(40, 70%, 60%)'])
          .clamp(true);

  /**
   * Interprets projector events and assembes the arrays and commands necessary
   * to use the ScatterPlot to render the current projected data set.
   */
  export class ProjectorScatterPlotAdapter {
    public scatterPlot: ScatterPlot;
    public noNonSelected: Boolean = false;
    public noNonPaired: Boolean = false;
    public imageSize: Number = 1.0;
    private projection: Projection;
    private hoverPointIndex: number;
    private selectedPointIndices: number[];
    private neighborsOfFirstSelectedPoint: knn.NearestEntry[];
    private renderLabelsIn3D: boolean = false;
    private labelPointAccessor: string;
    private legendPointColorer: (ds: DataSet, index: number) => string;
    private distanceMetric: DistanceFunction;
    private routePrefix: String;

    private spriteVisualizer: ScatterPlotVisualizerSprites;
    private labels3DVisualizer: ScatterPlotVisualizer3DLabels;
    private canvasLabelsVisualizer: ScatterPlotVisualizerCanvasLabels;
    private polylineVisualizer: ScatterPlotVisualizerPolylines;

    constructor(
        private scatterPlotContainer: HTMLElement,
        projectorEventContext: ProjectorEventContext) {
      this.scatterPlot =
          new ScatterPlot(scatterPlotContainer, projectorEventContext);
      projectorEventContext.registerProjectionChangedListener(projection => {
        this.projection = projection;
        this.updateScatterPlotWithNewProjection(projection);
      });
      this.imageSize = 1.0;
      projectorEventContext.registerSelectionChangedListener(
          (selectedPointIndices, neighbors) => {
            this.selectedPointIndices = selectedPointIndices;
            this.neighborsOfFirstSelectedPoint = neighbors;
            this.updateScatterPlotPositions();
            this.updateScatterPlotAttributes();
            this.scatterPlot.render();
          });
      projectorEventContext.registerHoverListener(hoverPointIndex => {
        this.hoverPointIndex = hoverPointIndex
        this.updateScatterPlotAttributes();
        this.scatterPlot.render();
      });
      projectorEventContext.registerDistanceMetricChangedListener(
          distanceMetric => {
            this.distanceMetric = distanceMetric;
            this.updateScatterPlotAttributes();
            this.scatterPlot.render();
          });
      addEventListener("addPos", (e) => {
        let found = false,
            first = (e as any).detail.first[0] as number,
            second = (e as any).detail.second[0] as number;
        //Find matching pair.
        for (var s_i in this.projection.dataSet.sequences) {
          let s = this.projection.dataSet.sequences[s_i];
          if ((!found) && (   ((s.pointIndices[0][0] == first) && (s.pointIndices[1][0] == second))
                           || ((s.pointIndices[1][0] == first) && (s.pointIndices[0][0] == second)) )) {
             found = true;
             if (s.color == 1) {
                //Quick Undo feature: if pair already exists, delete. This helps with accidental pair creations, misclicks.
                this.projection.dataSet.sequences.splice(+s_i, 1);
                this.sendPairs(first, second, 0);
             } else if (s.color == 2) {
                //Override feature: if pair exists but with other connotation, override it with new one.
                s.color = 1;
                this.sendPairs(first, second, 1);
             }
          }
        }
        //If not found, create it.
        if (!found) {
          this.projection.dataSet.sequences.push({
            pointIndices: [(e as any).detail.first, (e as any).detail.second],
            color: 1
          })
          this.sendPairs(first, second, 1);
        }
        this.renderCanvas();
        //Save data on server-side array.
      });
      addEventListener("addNeg", (e) => {
        let found = false,
            first = (e as any).detail.first[0] as number,
            second = (e as any).detail.second[0] as number;
        //Find matching pair.
        for (var s_i in this.projection.dataSet.sequences) {
          let s = this.projection.dataSet.sequences[s_i];
          if ((!found) && (   ((s.pointIndices[0][0] == first) && (s.pointIndices[1][0] == second))
                           || ((s.pointIndices[1][0] == first) && (s.pointIndices[0][0] == second)) )) {
             found = true;
             if (s.color == 2) {
                //Quick Undo feature: if pair already exists, delete. This helps with accidental pair creations, misclicks.
                this.projection.dataSet.sequences.splice(+s_i, 1);
                this.sendPairs(first, second, 0);
             } else if (s.color == 1) {
                //Override feautre: if pair exists but with other connotation, override it with new one.
                s.color = 2;
                this.sendPairs(first, second, -1);
             }
          }
        }
        //If not found, create it.
        if (!found) {
          this.projection.dataSet.sequences.push({
            pointIndices: [(e as any).detail.first, (e as any).detail.second],
            color: 2
          })
          this.sendPairs(first, second, -1);
        }
        this.renderCanvas();
      });
      addEventListener("removeP", (e) => {
        let found = false,
            first = (e as any).detail.first[0] as number,
            second = (e as any).detail.second[0] as number;
        //Find matching pair and delete it.
        for (var s_i in this.projection.dataSet.sequences) {
          let s = this.projection.dataSet.sequences[s_i];
          if ((!found) && (   ((s.pointIndices[0][0] == first) && (s.pointIndices[1][0] == second))
                           || ((s.pointIndices[1][0] == first) && (s.pointIndices[0][0] == second)) )) {
             found = true;
             this.projection.dataSet.sequences.splice(+s_i, 1);
          }
        }
        this.renderCanvas();
        //Save data on server-side array.
        this.sendPairs(first, second, 0);
      })
      this.createVisualizers(false);
    }

    /**
     * Send a pair to the server for consistensy.
     * @param first First point in pair.
     * @param second Second point in pair.
     * @param val Pair type value: (Negative = -1, Positive = 1, Delete Pair = 0)
     */
    sendPairs(first: number, second: number, val: number) {
      var xhr = new XMLHttpRequest();
      let projectorDataPanel = (document.querySelector("vz-projector-data-panel") as any);
      let selectedRun = projectorDataPanel.selectedRun.split("/")[1];
      xhr.open('POST', `${this.routePrefix}/pairs?n=${this.projection.dataSet.points.length}&subfolder=${tf_tensorboard.sublogdir}&selectedRun=${selectedRun}`);
      xhr.onload = () => {
        if (xhr.status == 200) {
          let values = xhr.responseText.split("&");
          window.dispatchEvent(new CustomEvent("changeValues", {detail: {
            pos1: values[0],
            pos2: values[1],
            neg1: values[2],
            neg2: values[3]
          }}))
        }
      }
      xhr.send(JSON.stringify({"first": first, "second": second, "val": val}));
    }

    setRoute(route: String) {
      this.routePrefix = route;
    }

    renderCanvas() {
      this.updateScatterPlotAttributes();
      this.updateScatterPlotPositions();
      this.resize();
      this.scatterPlot.render();
    }

    notifyProjectionPositionsUpdated() {
      this.updateScatterPlotPositions();
      this.scatterPlot.render();
    }

    setDataSet(dataSet: DataSet) {
      if (this.projection != null) {
        // TODO(@charlesnicholson): setDataSet needs to go away, the projection is the
        // atomic unit of update.
        this.projection.dataSet = dataSet;
      }
      if (this.polylineVisualizer != null) {
        this.polylineVisualizer.setDataSet(dataSet);
      }
      if (this.labels3DVisualizer != null) {
        this.labels3DVisualizer.setLabelStrings(
            this.generate3DLabelsArray(dataSet, this.labelPointAccessor));
      }
      if (this.spriteVisualizer == null) {
        return;
      }
      this.spriteVisualizer.clearSpriteAtlas();
      if ((dataSet == null) || (dataSet.spriteAndMetadataInfo == null)) {
        return;
      }
      const metadata = dataSet.spriteAndMetadataInfo;
      if ((metadata.spriteImage == null) || (metadata.spriteMetadata == null)) {
        return;
      }
      const n = dataSet.points.length;
      const spriteIndices = new Float32Array(n);
      for (let i = 0; i < n; ++i) {
        spriteIndices[i] = dataSet.points[i].index;
      }
      this.spriteVisualizer.setSpriteAtlas(
          metadata.spriteImage, metadata.spriteMetadata.singleImageDim,
          spriteIndices);
    }

    set3DLabelMode(renderLabelsIn3D: boolean) {
      this.renderLabelsIn3D = renderLabelsIn3D;
      this.createVisualizers(renderLabelsIn3D);
      this.updateScatterPlotAttributes();
      this.scatterPlot.render();
    }

    setOutlierMode(mode: boolean) {
      this.scatterPlot.outlierMode = mode;
    }

    setClusterMode(mode: boolean) {
      this.scatterPlot.clusterMode = mode;
      if(!mode) {
        var projector = (document.querySelector('vz-projector') as any);
    
        projector.cluster1 = null;
        projector.cluster2 = null;
      }
    }

    setAddPosPairMode(mode: boolean) {
      this.scatterPlot.addPosMode = mode;
      if (mode) {
        this.scatterPlot.addNegMode = false;
        this.scatterPlot.removeMode = false;
      }
    }

    setAddNegPairMode(mode: boolean) {
      this.scatterPlot.addNegMode = mode;
      if (mode) {
        this.scatterPlot.addPosMode = false;
        this.scatterPlot.removeMode = false;
      }
    }

    setRemovePairMode(mode: boolean) {
      this.scatterPlot.removeMode = mode;
      if (mode) {
        this.scatterPlot.addPosMode = false;
        this.scatterPlot.addNegMode = false;
      }
    }

    setGroupMode(mode: boolean) {
      this.scatterPlot.groupMode = mode;
    }

    setLegendPointColorer(
        legendPointColorer: (ds: DataSet, index: number) => string) {
      this.legendPointColorer = legendPointColorer;
    }

    setLabelPointAccessor(labelPointAccessor: string) {
      this.labelPointAccessor = labelPointAccessor;
      if (this.labels3DVisualizer != null) {
        const ds = (this.projection == null) ? null : this.projection.dataSet;
        this.labels3DVisualizer.setLabelStrings(
            this.generate3DLabelsArray(ds, labelPointAccessor));
      }
    }

    resize() {
      this.scatterPlot.resize();
    }

    populateBookmarkFromUI(state: State) {
      state.cameraDef = this.scatterPlot.getCameraDef();
    }

    restoreUIFromBookmark(state: State) {
      this.scatterPlot.setCameraParametersForNextCameraCreation(
          state.cameraDef, false);
    }

    updateScatterPlotPositions() {
      const ds = (this.projection == null) ? null : this.projection.dataSet;
      const projectionComponents =
          (this.projection == null) ? null : this.projection.projectionComponents;
      const newPositions =
          this.generatePointPositionArray(ds, projectionComponents);
      this.scatterPlot.setPointPositions(newPositions);
    }

    updateScatterPlotAttributes() {
      if (this.projection == null) {
        return;
      }

      let ds = this.projection.dataSet;
      const dataSet = ds;
      const selectedSet = this.selectedPointIndices;
      const hoverIndex = this.hoverPointIndex;
      const neighbors = this.neighborsOfFirstSelectedPoint;
      const pointColorer = this.legendPointColorer;

      const pointColors = this.generatePointColorArray(
          dataSet, pointColorer, this.distanceMetric, selectedSet, neighbors,
          hoverIndex, this.renderLabelsIn3D, this.getSpriteImageMode());
      const pointScaleFactors = this.generatePointScaleFactorArray(
          dataSet, selectedSet, neighbors, hoverIndex);
      const labels = this.generateVisibleLabelRenderParams(
          dataSet, selectedSet, neighbors, hoverIndex);
      const polylineColors =
          this.generateLineSegmentColorMap(dataSet, pointColorer);
      const polylineOpacities =
          this.generateLineSegmentOpacityArray(dataSet, selectedSet);
      const polylineWidths =
          this.generateLineSegmentWidthArray(dataSet, selectedSet);

      this.scatterPlot.setPointColors(pointColors);
      this.scatterPlot.setPointScaleFactors(pointScaleFactors);
      this.scatterPlot.setLabels(labels);
      this.scatterPlot.setPolylineColors(polylineColors);
      this.scatterPlot.setPolylineOpacities(polylineOpacities);
      this.scatterPlot.setPolylineWidths(polylineWidths);
    }

    render() {
      this.scatterPlot.render();
    }

    generatePointPositionArray(
        ds: DataSet, projectionComponents: ProjectionComponents3D): Float32Array {
      if (ds == null) {
        return null;
      }

      const xScaler = d3.scaleLinear();
      const yScaler = d3.scaleLinear();
      let zScaler = null;
      {
        // Determine max and min of each axis of our data.
        const xExtent = d3.extent(
            ds.points,
            (p, i) => ds.points[i].projections[projectionComponents[0]]);
        const yExtent = d3.extent(
            ds.points,
            (p, i) => ds.points[i].projections[projectionComponents[1]]);

        const range =
            [-SCATTER_PLOT_CUBE_LENGTH / 2, SCATTER_PLOT_CUBE_LENGTH / 2];

        xScaler.domain(xExtent).range(range);
        yScaler.domain(yExtent).range(range);

        if (projectionComponents[2] != null) {
          const zExtent = d3.extent(
              ds.points,
              (p, i) => ds.points[i].projections[projectionComponents[2]]);
          zScaler = d3.scaleLinear();
          zScaler.domain(zExtent).range(range);
        }
      }

      const positions = new Float32Array(ds.points.length * 3);
      let dst = 0;

      ds.points.forEach((d, i) => {
        positions[dst++] =
            xScaler(ds.points[i].projections[projectionComponents[0]]);
        positions[dst++] =
            yScaler(ds.points[i].projections[projectionComponents[1]]);
        positions[dst++] = 0.0;
      });

      if (zScaler) {
        dst = 2;
        ds.points.forEach((d, i) => {
          positions[dst] =
              zScaler(ds.points[i].projections[projectionComponents[2]]);
          dst += 3;
        });
      }

      return positions;
    }

    generateVisibleLabelRenderParams(
        ds: DataSet, selectedPointIndices: number[],
        neighborsOfFirstPoint: knn.NearestEntry[],
        hoverPointIndex: number): LabelRenderParams {
      if (ds == null) {
        return null;
      }

      const selectedPointCount =
          (selectedPointIndices == null) ? 0 : selectedPointIndices.length;
      const neighborCount =
          (neighborsOfFirstPoint == null) ? 0 : neighborsOfFirstPoint.length;
      const n = selectedPointCount + neighborCount +
          ((hoverPointIndex != null) ? 1 : 0);

      const visibleLabels = new Uint32Array(n);
      const scale = new Float32Array(n);
      const opacityFlags = new Int8Array(n);
      const fillColors = new Uint8Array(n * 3);
      const strokeColors = new Uint8Array(n * 3);
      const labelStrings: string[] = [];

      scale.fill(LABEL_SCALE_DEFAULT);
      opacityFlags.fill(1);

      let dst = 0;

      if (hoverPointIndex != null) {
        labelStrings.push(
            this.getLabelText(ds, hoverPointIndex, this.labelPointAccessor));
        visibleLabels[dst] = hoverPointIndex;
        scale[dst] = LABEL_SCALE_LARGE;
        opacityFlags[dst] = 0;
        const fillRgb = styleRgbFromHexColor(LABEL_FILL_COLOR_HOVER);
        packRgbIntoUint8Array(
            fillColors, dst, fillRgb[0], fillRgb[1], fillRgb[2]);
        const strokeRgb = styleRgbFromHexColor(LABEL_STROKE_COLOR_HOVER);
        packRgbIntoUint8Array(
            strokeColors, dst, strokeRgb[0], strokeRgb[1], strokeRgb[1]);
        ++dst;
      }

      // Selected points
      {
        const n = selectedPointCount;
        const fillRgb = styleRgbFromHexColor(LABEL_FILL_COLOR_SELECTED);
        const strokeRgb = styleRgbFromHexColor(LABEL_STROKE_COLOR_SELECTED);
        for (let i = 0; i < n; ++i) {
          const labelIndex = selectedPointIndices[i];
          labelStrings.push(
              this.getLabelText(ds, labelIndex, this.labelPointAccessor));
          visibleLabels[dst] = labelIndex;
          scale[dst] = LABEL_SCALE_LARGE;
          opacityFlags[dst] = (n === 1) ? 0 : 1;
          packRgbIntoUint8Array(
              fillColors, dst, fillRgb[0], fillRgb[1], fillRgb[2]);
          packRgbIntoUint8Array(
              strokeColors, dst, strokeRgb[0], strokeRgb[1], strokeRgb[2]);
          ++dst;
        }
      }

      // Neighbors
      {
        const n = neighborCount;
        const fillRgb = styleRgbFromHexColor(LABEL_FILL_COLOR_NEIGHBOR);
        const strokeRgb = styleRgbFromHexColor(LABEL_STROKE_COLOR_NEIGHBOR);
        for (let i = 0; i < n; ++i) {
          const labelIndex = neighborsOfFirstPoint[i].index;
          labelStrings.push(
              this.getLabelText(ds, labelIndex, this.labelPointAccessor));
          visibleLabels[dst] = labelIndex;
          packRgbIntoUint8Array(
              fillColors, dst, fillRgb[0], fillRgb[1], fillRgb[2]);
          packRgbIntoUint8Array(
              strokeColors, dst, strokeRgb[0], strokeRgb[1], strokeRgb[2]);
          ++dst;
        }
      }

      return new LabelRenderParams(
          new Float32Array(visibleLabels), labelStrings, scale, opacityFlags,
          LABEL_FONT_SIZE, fillColors, strokeColors);
    }

    generatePointScaleFactorArray(
        ds: DataSet, selectedPointIndices: number[],
        neighborsOfFirstPoint: knn.NearestEntry[],
        hoverPointIndex: number): Float32Array {
      if (ds == null) {
        return new Float32Array(0);
      }

      const scale = new Float32Array(ds.points.length);
      scale.fill((this.noNonPaired || this.noNonSelected) ? 0.00000000000001 : (this.imageSize) as any);
      if (this.noNonPaired) {
        for (var s of ds.sequences) {
          scale[s.pointIndices[0][0]] = (this.imageSize) as any;
          scale[s.pointIndices[1][0]] = (this.imageSize) as any;
        }
        return scale;
      } else if (this.noNonSelected) {
        for (var index of selectedPointIndices) {
          scale[index] = (this.imageSize) as any;
          let filtered = ds.sequences.filter(s => ((s.pointIndices[0][0] == index) || (s.pointIndices[1][0] == index)));
          for (var f_s of filtered) {
            scale[f_s.pointIndices[0][0]] = (this.imageSize) as any;
            scale[f_s.pointIndices[1][0]] = (this.imageSize) as any;
          }
        }
        return scale;
      }
      scale.fill(this.imageSize as any);

      const selectedPointCount =
          (selectedPointIndices == null) ? 0 : selectedPointIndices.length;
      const neighborCount =
          (neighborsOfFirstPoint == null) ? 0 : neighborsOfFirstPoint.length;

      // Scale up all selected points.
      {
        const n = selectedPointCount;
        for (let i = 0; i < n; ++i) {
          const p = selectedPointIndices[i];
          scale[p] = (this.imageSize as any) * 1.4;
        }
      }

      // Scale up the neighbor points.
      {
        const n = neighborCount;
        for (let i = 0; i < n; ++i) {
          const p = neighborsOfFirstPoint[i].index;
          scale[p] = (this.imageSize as any) * 1.4;
        }
      }

      // Scale up the hover point.
      if (hoverPointIndex != null) {
        scale[hoverPointIndex] = (this.imageSize as any) * 1.4;
      }

      return scale;
    }

    generateLineSegmentColorMap(
        ds: DataSet, legendPointColorer: (ds: DataSet, index: number) => string):
        {[polylineIndex: number]: Float32Array} {
      let polylineColorArrayMap: {[polylineIndex: number]: Float32Array} = {};
      if (ds == null) {
        return polylineColorArrayMap;
      }

      for (let i = 0; i < ds.sequences.length; i++) {
        let sequence = ds.sequences[i];
        let colors = new Float32Array(2 * (sequence.pointIndices.length - 1) * 3);
        let colorIndex = 0;

        if (legendPointColorer) {
          for (let j = 0; j < sequence.pointIndices.length - 1; j++) {
            const c1 =
                new THREE.Color(legendPointColorer(ds, sequence.pointIndices[j]));
            const c2 = new THREE.Color(
                legendPointColorer(ds, sequence.pointIndices[j + 1]));
            colors[colorIndex++] = c1.r;
            colors[colorIndex++] = c1.g;
            colors[colorIndex++] = c1.b;
            colors[colorIndex++] = c2.r;
            colors[colorIndex++] = c2.g;
            colors[colorIndex++] = c2.b;
          }
        } else {
          for (let j = 0; j < sequence.pointIndices.length - 1; j++) {
            const c1 =
                getDefaultPointInPolylineColor(j, sequence.pointIndices.length);
            const c2 = getDefaultPointInPolylineColor(
                j + 1, sequence.pointIndices.length);
            colors[colorIndex++] = c1.r;
            colors[colorIndex++] = c1.g;
            colors[colorIndex++] = c1.b;
            colors[colorIndex++] = c2.r;
            colors[colorIndex++] = c2.g;
            colors[colorIndex++] = c2.b;
          }
        }

        polylineColorArrayMap[i] = colors;
      }

      return polylineColorArrayMap;
    }

    generateLineSegmentOpacityArray(ds: DataSet, selectedPoints: number[]):
        Float32Array {
      if (ds == null) {
        return new Float32Array(0);
      }
      const opacities = new Float32Array(ds.sequences.length);
      const selectedPointCount =
          (selectedPoints == null) ? 0 : selectedPoints.length;
      if (selectedPointCount > 0) {
        opacities.fill(POLYLINE_DESELECTED_OPACITY);

        for (var index of selectedPoints) {
          for (var s_i in ds.sequences) {
            let s = ds.sequences[s_i];
            if ((s.pointIndices[0][0] == index) || (s.pointIndices[1][0] == index)) {
              opacities[s_i] = POLYLINE_SELECTED_OPACITY;
            }
          }
        }

        const i = ds.points[selectedPoints[0]].sequenceIndex;
        opacities[i] = POLYLINE_SELECTED_OPACITY;
      } else {
        opacities.fill(POLYLINE_DEFAULT_OPACITY);
      }
      return opacities;
    }

    generateLineSegmentWidthArray(ds: DataSet, selectedPoints: number[]):
        Float32Array {
      if (ds == null) {
        return new Float32Array(0);
      }
      const widths = new Float32Array(ds.sequences.length);
      if (selectedPoints.length > 0) {
        widths.fill(POLYLINE_DESELECTED_LINEWIDTH);
      } else {
        widths.fill(POLYLINE_DEFAULT_LINEWIDTH);
      }
      const selectedPointCount =
          (selectedPoints == null) ? 0 : selectedPoints.length;
      if (selectedPointCount > 0) {
        //const i = ds.points[selectedPoints[0]].sequenceIndex;
        //widths[i] = POLYLINE_SELECTED_LINEWIDTH;

        for (var index of selectedPoints) {
          for (var s_i in ds.sequences) {
            let s = ds.sequences[s_i];
            if ((s.pointIndices[0][0] == index) || (s.pointIndices[1][0] == index)) {
              widths[s_i] = POLYLINE_SELECTED_LINEWIDTH;
            }
          }
        }
      }
      return widths;
    }

    generatePointColorArray(
        ds: DataSet, legendPointColorer: (ds: DataSet, index: number) => string,
        distFunc: DistanceFunction, selectedPointIndices: number[],
        neighborsOfFirstPoint: knn.NearestEntry[], hoverPointIndex: number,
        label3dMode: boolean, spriteImageMode: boolean): Float32Array {
      if (ds == null) {
        return new Float32Array(0);
      }
      // TODO: a hack to keep the label colors when selecting an image as it helps with the pairing
      const selectedPointCount = 0;
      // set to 0 so even when an image is selected the color by label will stay functional 
      // const selectedPointCount =
      //     (selectedPointIndices == null) ? 0 : selectedPointIndices.length;

      const neighborCount =
          (neighborsOfFirstPoint == null) ? 0 : neighborsOfFirstPoint.length;
      const colors = new Float32Array(ds.points.length * 3);

      let point_unselected = POINT_COLOR_UNSELECTED,
          point_no_selection = POINT_COLOR_NO_SELECTION,
          point_selected = POINT_COLOR_SELECTED,
          point_hover = POINT_COLOR_HOVER;

      if (this.scatterPlot.addPosMode) {
        point_selected = 0x00FF00;
        point_hover = 0x00FF00;
      } else if (this.scatterPlot.addNegMode) {
        point_selected = 0xFF0000;
        point_hover = 0xFF0000;
      } else if (this.scatterPlot.removeMode) {
        point_selected = 0x1C1C1C;
        point_hover = 0x1C1C1C;
      }

      let unselectedColor = point_unselected;
      let noSelectionColor = point_no_selection;

      if (label3dMode) {
        unselectedColor = LABELS_3D_COLOR_UNSELECTED;
        noSelectionColor = LABELS_3D_COLOR_NO_SELECTION;
      }

      if (spriteImageMode) {
        unselectedColor = SPRITE_IMAGE_COLOR_UNSELECTED;
        noSelectionColor = SPRITE_IMAGE_COLOR_NO_SELECTION;
      }

      // Give all points the unselected color.
      {
        const n = ds.points.length;
        let dst = 0;
        if (selectedPointCount > 0) {
          const c = new THREE.Color(unselectedColor);
          for (let i = 0; i < n; ++i) {
            colors[dst++] = c.r;
            colors[dst++] = c.g;
            colors[dst++] = c.b;
          }
        } else {
          if (legendPointColorer != null) {
            for (let i = 0; i < n; ++i) {
              const c = new THREE.Color(legendPointColorer(ds, i));
              colors[dst++] = c.r;
              colors[dst++] = c.g;
              colors[dst++] = c.b;
            }
          } else {
            const c = new THREE.Color(noSelectionColor);
            for (let i = 0; i < n; ++i) {
              colors[dst++] = c.r;
              colors[dst++] = c.g;
              colors[dst++] = c.b;
            }
          }
        }
      }

      // Color the selected points.
      {
        const n = selectedPointCount;
        const c = new THREE.Color(point_selected);
        for (let i = 0; i < n; ++i) {
          let dst = selectedPointIndices[i] * 3;
          colors[dst++] = c.r;
          colors[dst++] = c.g;
          colors[dst++] = c.b;
        }
      }

      // Color the neighbors.
      {
        const n = neighborCount;
        let minDist = n > 0 ? neighborsOfFirstPoint[0].dist : 0;
        for (let i = 0; i < n; ++i) {
          const c = new THREE.Color(
              dist2color(distFunc, neighborsOfFirstPoint[i].dist, minDist));
          let dst = neighborsOfFirstPoint[i].index * 3;
          colors[dst++] = c.r;
          colors[dst++] = c.g;
          colors[dst++] = c.b;
        }
      }

      // Color the hover point.
      if (hoverPointIndex != null) {
        const c = new THREE.Color(point_hover);
        let dst = hoverPointIndex * 3;
        colors[dst++] = c.r;
        colors[dst++] = c.g;
        colors[dst++] = c.b;
      }

      return colors;
    }

    generate3DLabelsArray(ds: DataSet, accessor: string) {
      if ((ds == null) || (accessor == null)) {
        return null;
      }
      let labels: string[] = [];
      const n = ds.points.length;
      for (let i = 0; i < n; ++i) {
        labels.push(this.getLabelText(ds, i, accessor));
      }
      return labels;
    }

    private getLabelText(ds: DataSet, i: number, accessor: string) {
      return ds.points[i].metadata[accessor].toString();
    }

    private updateScatterPlotWithNewProjection(projection: Projection) {
      if (projection == null) {
        this.createVisualizers(this.renderLabelsIn3D);
        this.scatterPlot.render();
        return;
      }
      this.setDataSet(projection.dataSet);
      this.scatterPlot.setDimensions(projection.dimensionality);
      if (projection.dataSet.projectionCanBeRendered(projection.projectionType)) {
        this.updateScatterPlotAttributes();
        this.notifyProjectionPositionsUpdated();
      }
      this.scatterPlot.setCameraParametersForNextCameraCreation(null, false);
    }

    private createVisualizers(inLabels3DMode: boolean) {
      const ds = (this.projection == null) ? null : this.projection.dataSet;
      const scatterPlot = this.scatterPlot;
      scatterPlot.removeAllVisualizers();
      this.labels3DVisualizer = null;
      this.canvasLabelsVisualizer = null;
      this.spriteVisualizer = null;
      this.polylineVisualizer = null;
      if (inLabels3DMode) {
        this.labels3DVisualizer = new ScatterPlotVisualizer3DLabels();
        this.labels3DVisualizer.setLabelStrings(
            this.generate3DLabelsArray(ds, this.labelPointAccessor));
      } else {
        this.spriteVisualizer = new ScatterPlotVisualizerSprites();
        scatterPlot.addVisualizer(this.spriteVisualizer);
        this.canvasLabelsVisualizer =
            new ScatterPlotVisualizerCanvasLabels(this.scatterPlotContainer);
      }
      this.polylineVisualizer = new ScatterPlotVisualizerPolylines();
      this.setDataSet(ds);
      if (this.spriteVisualizer) {
        scatterPlot.addVisualizer(this.spriteVisualizer);
      }
      if (this.labels3DVisualizer) {
        scatterPlot.addVisualizer(this.labels3DVisualizer);
      }
      if (this.canvasLabelsVisualizer) {
        scatterPlot.addVisualizer(this.canvasLabelsVisualizer);
      }
      scatterPlot.addVisualizer(this.polylineVisualizer);
    }

    private getSpriteImageMode(): boolean {
      if (this.projection == null) {
        return false;
      }
      const ds = this.projection.dataSet;
      if ((ds == null) || (ds.spriteAndMetadataInfo == null)) {
        return false;
      }
      return ds.spriteAndMetadataInfo.spriteImage != null;
    }
  }

  function packRgbIntoUint8Array(
      rgbArray: Uint8Array, labelIndex: number, r: number, g: number, b: number) {
    rgbArray[labelIndex * 3] = r;
    rgbArray[labelIndex * 3 + 1] = g;
    rgbArray[labelIndex * 3 + 2] = b;
  }

  function styleRgbFromHexColor(hex: number): [number, number, number] {
    const c = new THREE.Color(hex);
    return [(c.r * 255) | 0, (c.g * 255) | 0, (c.b * 255) | 0];
  }

  function getDefaultPointInPolylineColor(
      index: number, totalPoints: number): THREE.Color {
    let hue = POLYLINE_START_HUE +
        (POLYLINE_END_HUE - POLYLINE_START_HUE) * index / totalPoints;

    let rgb = d3.hsl(hue, POLYLINE_SATURATION, POLYLINE_LIGHTNESS).rgb();
    return new THREE.Color(rgb.r / 255, rgb.g / 255, rgb.b / 255);
  }

  /**
   * Normalizes the distance so it can be visually encoded with color.
   * The normalization depends on the distance metric (cosine vs euclidean).
   */
  export function normalizeDist(
      distFunc: DistanceFunction, d: number, minDist: number): number {
    return (distFunc === vector.dist) ? (minDist / d) : (1 - d);
  }

  /** Normalizes and encodes the provided distance with color. */
  export function dist2color(
      distFunc: DistanceFunction, d: number, minDist: number): string {
    return NN_COLOR_SCALE(normalizeDist(distFunc, d, minDist));
  }

  }  // namespace vz_projector
