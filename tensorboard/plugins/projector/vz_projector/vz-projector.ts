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

/**
 * The minimum number of dimensions the data should have to automatically
 * decide to normalize the data.
 */
const THRESHOLD_DIM_NORMALIZE = 50;
const POINT_COLOR_MISSING = 'black';

export let ProjectorPolymer = PolymerElement({
  is: 'vz-projector',
  properties: {
    routePrefix: String,
    projectionMode: {type: String, observer: "handleModeChanged", notify: true},
    dataProto: {type: String, observer: '_dataProtoChanged'},
    servingMode: String,
    projectorConfigJsonPath: String,
    pageViewLogging: Boolean,
    eventLogging: Boolean,
    imageSizeChange: Object,
    accuracy: Number,
  }
});

const INDEX_METADATA_FIELD = '__index__';

/** The class responsible for the Embedding Projector. */
export class Projector extends ProjectorPolymer implements
    ProjectorEventContext {
  // The working subset of the data source's original data set.
  dataSet: DataSet;
  servingMode: ServingMode;
  // The path to the projector config JSON file for demo mode.
  projectorConfigJsonPath: string;
  loaded: boolean;
  
  private pos1: any;
  private pos2: any;
  private neg1: any;
  private neg2: any;
  private hideStatus: number;
  private alert_prevention; // workaround boolean so that server connection loss alert isn't displayed 3 times.

  private selectionChangedListeners: SelectionChangedListener[];
  private hoverListeners: HoverListener[];
  private projectionChangedListeners: ProjectionChangedListener[];
  private distanceMetricChangedListeners: DistanceMetricChangedListener[];

  private originalDataSet: DataSet;
  private dataSetBeforeFilter: DataSet;
  private projectorScatterPlotAdapter: ProjectorScatterPlotAdapter;
  private dim: number;

  private dataSetFilterIndices: number[];
  private selectedPointIndices: number[];
  private neighborsOfFirstPoint: knn.NearestEntry[];
  private hoverPointIndex: number;
  private editMode: boolean;

  private dataProvider: DataProvider;
  private inspectorPanel: InspectorPanel;

  private selectedColorOption: ColorOption;
  private selectedLabelOption: string;
  private routePrefix: string;
  private normalizeData: boolean;
  private projection: Projection;

  /** Polymer component panels */
  private dataPanel: DataPanel;
  private bookmarkPanel: BookmarkPanel;
  private projectionsPanel: ProjectionsPanel;
  private metadataCard: MetadataCard;

  private statusBar: HTMLDivElement;
  private analyticsLogger: AnalyticsLogger;
  private eventLogging: boolean;
  private pageViewLogging: boolean;
  private imagesize: Number;
  private labelmap: any;

  private clusterMode: boolean = false;
  private outlierMode: boolean = false;

  public cluster1 = null;
  public cluster2 = null;
  public outlier;

  /** Other plugins can use this to access the projector's backend services. */
  getRoutePrefix(): string {
    return this.routePrefix;
  }

  /** Runs when the projector plugin is instantiated. */
  ready() {

    if(!tf_tensorboard.disablepairwise){
      let removebtn = this.getRemoveAllButton();
      let opaq = 1;
      removebtn.style = `color:rgba(0,0,0,${opaq})`;
      removebtn.disabled = false;
    }
    else{
      let removebtn = this.getRemoveAllButton();
      let opaq = 0.2;
      removebtn.style = `color:rgba(0,0,0,${opaq})`;
      removebtn.disabled = true;
    }

    if (window.localStorage.getItem(tf_tensorboard.sublogdir+'-imagesize') === null) {
      this.imagesize = 1.0;
    }
    else{
      this.imagesize = Number(window.localStorage.getItem(tf_tensorboard.sublogdir+'-imagesize'));
    }
    
    this.loaded = false;
    this.alert_prevention = false;
    //Set the popup message/error interface on the projector plugin.
    logging.setDomContainer(this);

    //Register the listener responsible for outside changes on the selected elements.
    tf_tensorboard.registerSelectionChangedListener(
      (selection) =>
          this.selectionChanged(selection), "projector");

    this.analyticsLogger =
        new AnalyticsLogger(this.pageViewLogging, this.eventLogging);
    this.analyticsLogger.logPageView('embeddings');

    if (!util.hasWebGLSupport()) {
      this.analyticsLogger.logWebGLDisabled();
      logging.setErrorMessage(
          'Your browser or device does not have WebGL enabled. Please enable ' +
          'hardware acceleration, or use a browser that supports WebGL.');
      return;
    }

    this.pos1 = document.querySelector('#pos1');
    this.pos2 = document.querySelector('#pos2');
    this.neg1 = document.querySelector('#neg1');
    this.neg2 = document.querySelector('#neg2');


    addEventListener("changeValues", (e) => {
      (document.querySelector('#pos1') as any).innerText = (e as any).detail.pos1;
      (document.querySelector('#pos2') as any).innerText = (e as any).detail.pos2;
      (document.querySelector('#neg1') as any).innerText = (e as any).detail.neg1;
      (document.querySelector('#neg2') as any).innerText = (e as any).detail.neg2;
    })

    this.selectionChangedListeners = [];
    this.hoverListeners = [];
    this.projectionChangedListeners = [];
    this.distanceMetricChangedListeners = [];
    this.selectedPointIndices = [];
    this.neighborsOfFirstPoint = [];
    this.editMode = false;

    this.dataPanel = this.$['data-panel'] as DataPanel;
    // this.inspectorPanel = this.$['inspector-panel'] as InspectorPanel;
    // this.inspectorPanel.initialize(this, this as ProjectorEventContext);
    this.projectionsPanel = this.$['projections-panel'] as ProjectionsPanel;
    this.projectionsPanel.initialize(this);
    this.bookmarkPanel = this.$['bookmark-panel'] as BookmarkPanel;
    this.bookmarkPanel.initialize(this, this as ProjectorEventContext);
    this.metadataCard = this.$['metadata-card'] as MetadataCard;
    this.statusBar = this.querySelector('#status-bar') as HTMLDivElement;
    this.scopeSubtree(this.$$('#notification-dialog'), true);
    this.setupUIControls();
    this.initializeDataProvider();
    //this.setInfo((true) ?
    //"Welcome to the NIPGBoard! To get started, please select a Base Learner model and a dataset to create an embedding, or select an existing embedding for analysis!" :
    //"Welcome to the NIPGBoard! To get started, please select a Base Learner model and a dataset to create an embedding, or select an existing embedding for analysis!");
    tf_tensorboard.notifyHelpChanged((true) ?
    "Welcome to the NIPGBoard! To get started, please select a Base Learner model and a dataset to create an embedding, or select an existing embedding for analysis!" :
    "Welcome to the NIPGBoard! To get started, please select a Base Learner model and a dataset to create an embedding, or select an existing embedding for analysis!","projector");
    this.serverWorker();

  }

  /** Worker function that alerts the user if they lost connection with the server. */
  serverWorker() {
    var ref = this;
    var worker = window.setInterval(function(){
      var checker = new XMLHttpRequest();
      checker.open('GET', `${ref.routePrefix}/check`);
      checker.onerror = () => {
        clearInterval(worker);
        if(!this.alert_prevention) alert("Lost connection with server! Please contact administrator.");
        this.alert_prevention = true;
      }
      checker.send();
    }, 5000);
  }

  setSelectedLabelOption(labelOption: string) {
    this.selectedLabelOption = labelOption;
    this.metadataCard.setLabelOption(this.selectedLabelOption);
    this.projectorScatterPlotAdapter.setLabelPointAccessor(labelOption);
    this.projectorScatterPlotAdapter.updateScatterPlotAttributes();
    this.projectorScatterPlotAdapter.render();
  }

  setSelectedColorOption(colorOption: ColorOption) {
    this.selectedColorOption = colorOption;
    this.projectorScatterPlotAdapter.setLegendPointColorer(
        this.getLegendPointColorer(colorOption));
    this.projectorScatterPlotAdapter.updateScatterPlotAttributes();
    this.projectorScatterPlotAdapter.render();
  }

  setNormalizeData(normalizeData: boolean) {
    this.normalizeData = normalizeData;
    this.setCurrentDataSet(this.originalDataSet.getSubset());
  }

  updateSelectedPoint(selection: number[]) {
  }

  /** Just in case, call a render on the projector to make sure it stays up to date visually.
   *  TODO: minimise the number of calls for this function.
   */
  renderCanvas() {
    this.projectorScatterPlotAdapter.updateScatterPlotAttributes();
    this.projectorScatterPlotAdapter.updateScatterPlotPositions();
    this.projectorScatterPlotAdapter.resize();
    this.projectorScatterPlotAdapter.scatterPlot.render();
  }

  /** Updates the projector dataset and metadata after loading. */
  updateDataSet(
      ds: DataSet, spriteAndMetadata?: SpriteAndMetadataInfo,
      metadataFile?: string) {
    this.dataSetFilterIndices = null;
    this.originalDataSet = ds;

    if (ds != null) {
      this.normalizeData =
          this.originalDataSet.dim[1] >= THRESHOLD_DIM_NORMALIZE;
      spriteAndMetadata = spriteAndMetadata || {};
      if (spriteAndMetadata.pointsInfo == null) {
        let [pointsInfo, stats] = this.makeDefaultPointsInfoAndStats(ds.points);
        spriteAndMetadata.pointsInfo = pointsInfo;
        spriteAndMetadata.stats = stats;
      }
      let metadataMergeSucceeded = ds.mergeMetadata(spriteAndMetadata);
      if (!metadataMergeSucceeded) {
        return;
      }
    }
    if (this.projectorScatterPlotAdapter != null) {
      if (ds == null) {
        this.projectorScatterPlotAdapter.setLabelPointAccessor(null);
        this.setProjection(null);
      } else {
        //Load the already annotated pairs from the server for consistensy.
        let pair_loader = new XMLHttpRequest();
        var pairs = {};

        let projectorDataPanel = (document.querySelector("vz-projector-data-panel") as any);
        let selectedRun = projectorDataPanel.selectedRun.split("/")[1];

        pair_loader.open('GET', `${this.routePrefix}/load?n=${ds.points.length}&subfolder=${tf_tensorboard.sublogdir}&selectedRun=${selectedRun}`);
        pair_loader.onload = () => {
          let string_data = pair_loader.responseText;
          if (string_data === "") {
            (document.querySelector('#pos1') as any).innerText = 0;
            (document.querySelector('#pos2') as any).innerText = 0;
            (document.querySelector('#neg1') as any).innerText = 0;
            (document.querySelector('#neg2') as any).innerText = 0;
            this.dataSet.sequences = [];
          } else {
            pairs = JSON.parse(string_data);
            (document.querySelector('#pos1') as any).innerText = pairs["pos1"];
            (document.querySelector('#pos2') as any).innerText = pairs["pos2"];
            (document.querySelector('#neg1') as any).innerText = pairs["neg1"];
            (document.querySelector('#neg2') as any).innerText = pairs["neg2"];
            for (var p of pairs["pos"]) {
              this.dataSet.sequences.push({
                pointIndices: [([p[0]] as any), ([p[1]] as any)],
                color: 1
              })
            }
            for (var p of pairs["neg"]) {
              this.dataSet.sequences.push({
                pointIndices: [([p[0]] as any), ([p[1]] as any)],
                color: 2
              })
            }
          }
          this.renderCanvas();
        }
        pair_loader.send();
        this.renderCanvas();
      }
    }
    if (ds != null) {
      this.dataPanel.setNormalizeData(this.normalizeData);
      this.setCurrentDataSet(ds.getSubset());
      this.projectorScatterPlotAdapter.setLabelPointAccessor(
          this.selectedLabelOption);
      // this.inspectorPanel.datasetChanged();

      // this.inspectorPanel.metadataChanged(spriteAndMetadata);
      this.projectionsPanel.metadataChanged(spriteAndMetadata);
      this.dataPanel.metadataChanged(spriteAndMetadata, metadataFile);
      this.renderCanvas();
    } else {
      this.setCurrentDataSet(null);
    }
    this.loaded = true;
    this.renderCanvas();


    //let gc = document.querySelector("vz-graphcut") as any;
    //gc.handleRunGc();
    //this.setInfo("Graph Cut results are not updated or are missing. Please run the Graph Cut algorithm.");
    tf_tensorboard.notifyHelpChanged("Graph Cut results are not updated or are missing. Please run the Graph Cut algorithm.","projector");

  }

  metadataEdit(metadataColumn: string, metadataLabel: string) {
    this.selectedPointIndices.forEach(i =>
        this.dataSet.points[i].metadata[metadataColumn] = metadataLabel);
    
    this.neighborsOfFirstPoint.forEach(p =>
        this.dataSet.points[p.index].metadata[metadataColumn] = metadataLabel);
    
    this.dataSet.spriteAndMetadataInfo.stats = analyzeMetadata(
        this.dataSet.spriteAndMetadataInfo.stats.map(s => s.name),
        this.dataSet.points.map(p => p.metadata));
    this.metadataChanged(this.dataSet.spriteAndMetadataInfo);
    this.metadataEditorContext(true, metadataColumn);
  }

  metadataChanged(spriteAndMetadata: SpriteAndMetadataInfo,
      metadataFile?: string) {
    if (metadataFile != null) {
      this.metadataFile = metadataFile;
    }

    this.dataSet.spriteAndMetadataInfo = spriteAndMetadata;
    this.projectionsPanel.metadataChanged(spriteAndMetadata);
    // this.inspectorPanel.metadataChanged(spriteAndMetadata);
    this.dataPanel.metadataChanged(spriteAndMetadata, this.metadataFile);
    
    if (this.selectedPointIndices.length > 0) {  // at least one selected point
      this.metadataCard.updateMetadata(  // show metadata for first selected point
          this.dataSet.points[this.selectedPointIndices[0]].metadata);
    }
    else {  // no points selected
      this.metadataCard.updateMetadata(null);  // clear metadata
    }
    this.setSelectedLabelOption(this.selectedLabelOption);
  }

  metadataEditorContext(enabled: boolean, metadataColumn: string) {
    // if (this.inspectorPanel) {
    //   this.inspectorPanel.metadataEditorContext(enabled, metadataColumn);
    // }
  }

  setSelectedTensor(run: string, tensorInfo: EmbeddingInfo) {
    this.bookmarkPanel.setSelectedTensor(run, tensorInfo, this.dataProvider);
  }

  getEmbeddingInfo(run: string): any {
    let data = this.dataPanel.getEmbeddingInfoByName(run);
    return data;
  }

  /**
   * Registers a listener to be called any time the selected point set changes.
   */
  registerSelectionChangedListener(listener: SelectionChangedListener) {
    this.selectionChangedListeners.push(listener);
  }

  filterDataset(pointIndices: number[]) {
    const selectionSize = this.selectedPointIndices.length;
    if (this.dataSetBeforeFilter == null) {
      this.dataSetBeforeFilter = this.dataSet;
    }
    this.setCurrentDataSet(this.dataSet.getSubset(pointIndices));
    this.dataSetFilterIndices = pointIndices;
    this.adjustSelectionAndHover(util.range(selectionSize));
    this.renderCanvas();
  }

  resetFilterDataset() {
    const originalPointIndices = this.selectedPointIndices.map(
        filteredIndex => this.dataSet.points[filteredIndex].index);
    this.setCurrentDataSet(this.dataSetBeforeFilter);
    if (this.projection != null) {
      this.projection.dataSet = this.dataSetBeforeFilter;
    }
    this.dataSetBeforeFilter = null;
    this.renderCanvas();
    this.dataSetFilterIndices = [];
    this.adjustSelectionAndHover(originalPointIndices);
  }

  /**
   * Used by clients to indicate that a selection has occurred.
   */
  selectionChanged(newSelectedPointIndices: number[]) {
    let neighbors: knn.NearestEntry[] = [];

    if (this.editMode) { // point selection toggle in existing selection
        //&& newSelectedPointIndices.length > 0) {
        let updatedSelectedPointIndices = this.selectedPointIndices.filter(n =>
            newSelectedPointIndices.filter(p => p == n).length == 0);  // deselect
        newSelectedPointIndices.forEach(p => {  // add additional selections
          if (this.selectedPointIndices.filter(s => s == p).length == 0)  // unselected
            updatedSelectedPointIndices.push(p);
        });
        this.selectedPointIndices = updatedSelectedPointIndices;
          // update selection

        if (this.selectedPointIndices.length > 0) {  // at least one selected point
          this.metadataCard.updateMetadata(  // show metadata for first selected point
               this.dataSet.points[this.selectedPointIndices[0]].metadata);
        } else {  // no points selected
          this.metadataCard.updateMetadata(null);  // clear metadata
        }
      }
    else {  // normal selection mode
      this.selectedPointIndices = newSelectedPointIndices;

      if (newSelectedPointIndices.length === 1) {
        //neighbors = this.dataSet.findNeighbors(
        //    newSelectedPointIndices[0], this.inspectorPanel.distFunc,
        //    this.inspectorPanel.numNN);
        this.metadataCard.updateMetadata(
            this.dataSet.points[newSelectedPointIndices[0]].metadata);
      } else {
        this.metadataCard.updateMetadata(null);
      }
    }
    
    this.selectionChangedListeners.forEach(
        l => l(this.selectedPointIndices, neighbors));
  }

  notifySelectionChanged(newSelectedPointIndices: number[]) {
    if (this.dataSet) {
      var syncs = [];
      for(var index of newSelectedPointIndices) {
        if(this.dataSet.points[index].metadata["_sync_id"] != undefined) {
          syncs.push(this.dataSet.points[index].metadata["_sync_id"]);
        }
      }

      tf_tensorboard.notifySelectionChanged(syncs, false, "projector");
    }
    this.selectionChanged(newSelectedPointIndices);
  }

  /**
   * Registers a listener to be called any time the mouse hovers over a point.
   */
  registerHoverListener(listener: HoverListener) {
    this.hoverListeners.push(listener);
  }

  handleModeChanged(){
    //let gc = document.querySelector("vz-graphcut") as any;
    //gc.handleRunGc();
  }


  /**
   * Used by clients to indicate that a hover is occurring.
   */
  notifyHoverOverPoint(pointIndex: number) {
    this.hoverListeners.forEach(l => l(pointIndex));
  }

  registerProjectionChangedListener(listener: ProjectionChangedListener) {
    this.projectionChangedListeners.push(listener);
  }

  notifyProjectionChanged(projection: Projection) {
    this.projectionChangedListeners.forEach(l => l(projection));
  }

  registerDistanceMetricChangedListener(l: DistanceMetricChangedListener) {
    this.distanceMetricChangedListeners.push(l);
  }

  notifyDistanceMetricChanged(distMetric: DistanceFunction) {
    this.distanceMetricChangedListeners.forEach(l => l(distMetric));
  }

  _dataProtoChanged(dataProtoString: string) {
    let dataProto =
        dataProtoString ? JSON.parse(dataProtoString) as DataProto : null;
    this.initializeDataProvider(dataProto);
  }

  private makeDefaultPointsInfoAndStats(points: DataPoint[]):
      [PointMetadata[], ColumnStats[]] {
    let pointsInfo: PointMetadata[] = [];
    points.forEach(p => {
      let pointInfo: PointMetadata = {};
      pointInfo[INDEX_METADATA_FIELD] = p.index;
      pointsInfo.push(pointInfo);
    });
    let stats: ColumnStats[] = [{
      name: INDEX_METADATA_FIELD,
      isNumeric: false,
      tooManyUniqueValues: true,
      min: 0,
      max: pointsInfo.length - 1
    }];
    return [pointsInfo, stats];
  }

  private initializeDataProvider(dataProto?: DataProto) {
    if (this.servingMode === 'demo') {
      let projectorConfigUrl: string;

      // Only in demo mode do we allow the config being passed via URL.
      let urlParams = util.getURLParams(window.location.search);
      if ('config' in urlParams) {
        projectorConfigUrl = urlParams['config'];
      } else {
        projectorConfigUrl = this.projectorConfigJsonPath;
      }
      this.dataProvider = new DemoDataProvider(projectorConfigUrl);
    } else if (this.servingMode === 'server') {
      if (!this.routePrefix) {
        throw 'route-prefix is a required parameter';
      }
      this.dataProvider = new ServerDataProvider(this.routePrefix);
    } else if (this.servingMode === 'proto' && dataProto != null) {
      this.dataProvider = new ProtoDataProvider(dataProto);
    } else {
      // The component is not ready yet - waiting for the dataProto field.
      return;
    }

    this.dataPanel.initialize(this, this.dataProvider);
  }

  private getLegendPointColorer(colorOption: ColorOption):
      (ds: DataSet, index: number) => string {
    if ((colorOption == null) || (colorOption.map == null)) {
      return null;
    }
    const colorer = (ds: DataSet, i: number) => {
      let value = ds.points[i].metadata[this.selectedColorOption.name];
      if (value == null) {
        return POINT_COLOR_MISSING;
      }
      return colorOption.map(value);
    };
    return colorer;
  }

  private get3DLabelModeButton(): any {
    return this.querySelector('#labels3DMode');
  }

  private getAddPosPairButton(): any {
    return this.querySelector('#addPosPairs')
  }

  private getAddNegPairButton(): any {
    return this.querySelector('#addNegPairs')
  }

  private getNoNonSelectedButton(): any {
    return this.querySelector('#noNonSelected')
  }

  private getNoNonPairedButton(): any {
    return this.querySelector('#noNonPaired')
  }

  private getRemovePairsButton(): any {
    return this.querySelector('#removePairs')
  }

  private getRemoveAllButton(): any {
    return this.querySelector('#removeAll')
  }

  private getLeftPanelButton(): any {
    return this.querySelector('#showLeftPanel')
  }

  private getInfoPanelButton(): any {
    return this.querySelector('#showInfo')
  }

  private getGroupModeButton(): any {
    return this.querySelector('#groupMode')
  }

  private getUnifyClusterButton(): any {
    return this.querySelector('#unifyClusters')
  }

  private getCreateOutlierButton(): any {
    return this.querySelector('#createOutlier')
  }

  private get3DLabelMode(): boolean {
    const label3DModeButton = this.get3DLabelModeButton();
    return (label3DModeButton as any).active;
  }

  private getAddPosPairMode(): boolean {
    const addPosPairButton = this.getAddPosPairButton();
    return (addPosPairButton as any).active;
  }

  private getAddNegPairMode(): boolean {
    const addNegPairButton = this.getAddNegPairButton();
    return (addNegPairButton as any).active;
  }


  private getNoNonSelectedMode(): boolean {
    const noNonSelectedButton = this.getNoNonSelectedButton();
    return (noNonSelectedButton as any).active;
  }

  private getNoNonPairedMode(): boolean {
    const noNonPairedButton = this.getNoNonPairedButton();
    return (noNonPairedButton as any).active;
  }

  private getRemovePairsMode(): boolean {
    const removePairsButton = this.getRemovePairsButton();
    return (removePairsButton as any).active;
  }

  private getGroupMode(): boolean {
    const groupModeButton = this.getGroupModeButton();
    return (groupModeButton as any).active;
  }

  private getLeftPanelMode(): boolean {
    const leftPanelButton = this.getLeftPanelButton();
    return (leftPanelButton as any).active;
  }

  adjustSelectionAndHover(selectedPointIndices: number[], hoverIndex?: number) {
    this.notifySelectionChanged(selectedPointIndices);
    this.notifyHoverOverPoint(hoverIndex);
    this.setMouseMode(MouseMode.CAMERA_AND_CLICK_SELECT);
  }

  private setMouseMode(mouseMode: MouseMode) {
    let selectModeButton = this.querySelector('#selectMode');
    (selectModeButton as any).active = (mouseMode === MouseMode.AREA_SELECT);
    this.projectorScatterPlotAdapter.scatterPlot.setMouseMode(mouseMode);
  }

  private setCurrentDataSet(ds: DataSet) {
    this.adjustSelectionAndHover([]);
    if (this.dataSet != null) {
      this.dataSet.stopTSNE();
    }
    if ((ds != null) && this.normalizeData) {
      ds.normalize();
    }
    this.dim = (ds == null) ? 0 : ds.dim[1];
    (this.querySelector('span.numDataPoints') as HTMLSpanElement).innerText =
        (ds == null) ? '0' : '' + ds.dim[0];
    (this.querySelector('span.dim') as HTMLSpanElement).innerText =
        (ds == null) ? '0' : '' + ds.dim[1];

    this.dataSet = ds;

    this.projectionsPanel.dataSetUpdated(
        this.dataSet, this.originalDataSet, this.dim);

    this.projectorScatterPlotAdapter.setDataSet(this.dataSet);
    this.projectorScatterPlotAdapter.scatterPlot
        .setCameraParametersForNextCameraCreation(null, true);
  }

  private setupUIControls() {

    this.querySelector('#reset-zoom').addEventListener('click', () => {
      this.projectorScatterPlotAdapter.scatterPlot.resetZoom();
      this.projectorScatterPlotAdapter.scatterPlot.startOrbitAnimation();
    });

    let selectModeButton = this.querySelector('#selectMode');
    selectModeButton.addEventListener('click', (event) => {
      this.setMouseMode(
          (selectModeButton as any).active ? MouseMode.AREA_SELECT :
                                             MouseMode.CAMERA_AND_CLICK_SELECT);
    });

    let editModeButton = this.querySelector('#editMode');
      editModeButton.addEventListener('click', (event) => {
        this.editMode = (editModeButton as any).active;
        if (this.editMode == false) {
          this.notifySelectionChanged(this.selectedPointIndices);
        }
    });

    let leftPanelButton = this.getLeftPanelButton();
    leftPanelButton.addEventListener('click', () => {
      let mode = this.getLeftPanelMode(),
          leftPanel = (this.querySelector('#left-pane') as HTMLElement);
      leftPanel.style["visibility"] = (mode ? "hidden" : "visible");
      leftPanel.style["min-width"] = (mode ? "0px" : "312px");
      leftPanel.style["width"] = (mode ? "0px" : "312px");
      this.projectorScatterPlotAdapter.updateScatterPlotAttributes;
      this.projectorScatterPlotAdapter.updateScatterPlotPositions;
      this.projectorScatterPlotAdapter.resize();
      this.projectorScatterPlotAdapter.render();
    })

    let showInfoButton = this.getInfoPanelButton();
    showInfoButton.addEventListener('click', () => {
      this.$.info.open();
    })

    let groupModeButton = this.getGroupModeButton();
    groupModeButton.addEventListener('click', () => {
      this.projectorScatterPlotAdapter.setGroupMode(this.getGroupMode());
    })

    const imageSize = this.querySelector('#imageSize');
    var _ret = this;
    imageSize.addEventListener('value-change', function() {
      _ret.projectorScatterPlotAdapter.imageSize = (imageSize as any).value;
      _ret.projectorScatterPlotAdapter.updateScatterPlotAttributes();
      _ret.projectorScatterPlotAdapter.render();
    })

    const unifyClustersButton = this.getUnifyClusterButton();
    unifyClustersButton.addEventListener('click', () => {
      this.clusterMode = !this.clusterMode;
      if (this.clusterMode) {
        //this.setInfo("Please select the other cluster's element.");
        tf_tensorboard.notifyHelpChanged("Please select the other cluster's element.","projector");
      } else {
        if(this.cluster1&&this.cluster2) {
          this.mergeClusters(this.cluster1, this.cluster2);
        }
        //this.setInfo((this.getGroupMode()) ?
        // "Welcome to the NIPGBoard! To get started, please select a Base Learner model and a dataset to create an embedding, or select an existing embedding for analysis!" :
        //  "Welcome to the NIPGBoard! To get started, please select a Base Learner model and a dataset to create an embedding, or select an existing embedding for analysis!");
          tf_tensorboard.notifyHelpChanged((this.getGroupMode()) ?
          "Welcome to the NIPGBoard! To get started, please select a Base Learner model and a dataset to create an embedding, or select an existing embedding for analysis!" :
          "Welcome to the NIPGBoard! To get started, please select a Base Learner model and a dataset to create an embedding, or select an existing embedding for analysis!","projector");
      }
      this.projectorScatterPlotAdapter.setClusterMode(this.clusterMode);
    })

    const createOutlierButton = this.getCreateOutlierButton();
    createOutlierButton.addEventListener('click', () => {
      this.outlierMode = !this.outlierMode;
      if (this.outlierMode) {
        //this.setInfo("Please select an element as a cluster outlier");
        tf_tensorboard.notifyHelpChanged("Please select an element as a cluster outlier","projector");
      } else {
        this.createOutlier(this.outlier);
        //this.setInfo((this.getGroupMode()) ?
        //  "Welcome to the NIPGBoard! To get started, please select a Base Learner model and a dataset to create an embedding, or select an existing embedding for analysis!" :
        //  "Welcome to the NIPGBoard! To get started, please select a Base Learner model and a dataset to create an embedding, or select an existing embedding for analysis!");
          tf_tensorboard.notifyHelpChanged((this.getGroupMode()) ?
          "Welcome to the NIPGBoard! To get started, please select a Base Learner model and a dataset to create an embedding, or select an existing embedding for analysis!" :
          "Welcome to the NIPGBoard! To get started, please select a Base Learner model and a dataset to create an embedding, or select an existing embedding for analysis!","projector");
      }
      this.projectorScatterPlotAdapter.setOutlierMode(this.outlierMode);
    })

    const addPosPairButton = this.getAddPosPairButton();
    addPosPairButton.addEventListener('click', () => {
      let pMode = this.getAddPosPairMode();
      this.projectorScatterPlotAdapter.setAddPosPairMode(pMode);
      if (pMode == true) {
        if ((this.selectedPointIndices == []) && (this.getGroupMode())) {
          //this.setInfo("Please select a focus item first before adding or removing new pairs!");
          tf_tensorboard.notifyHelpChanged("Please select a focus item first before adding or removing new pairs!","projector");
        } else {
        //this.setInfo((this.getGroupMode()) ?
        // "Select the items that are in positive relation with the focus item!" :
        //  "Select the first item in the new positive relation!");
          tf_tensorboard.notifyHelpChanged((this.getGroupMode()) ?
          "Select the items that are in positive relation with the focus item!" :
          "Select the first item in the new positive relation!","projector");
        }
        let nButton = this.getAddNegPairButton(),
            rButton = this.getRemovePairsButton();
        (nButton as any).active = (rButton as any).active = false;
      } else {
        //this.setInfo((this.getGroupMode()) ?
        //  "Welcome to the NIPGBoard! To get started, please select a Base Learner model and a dataset to create an embedding, or select an existing embedding for analysis!" :
        //  "Welcome to the NIPGBoard! To get started, please select a Base Learner model and a dataset to create an embedding, or select an existing embedding for analysis!");
          tf_tensorboard.notifyHelpChanged((this.getGroupMode()) ?
          "Welcome to the NIPGBoard! To get started, please select a Base Learner model and a dataset to create an embedding, or select an existing embedding for analysis!" :
          "Welcome to the NIPGBoard! To get started, please select a Base Learner model and a dataset to create an embedding, or select an existing embedding for analysis!","projector");
      }
      this.renderCanvas();
    })

    const addNegPairButton = this.getAddNegPairButton();
    addNegPairButton.addEventListener('click', () => {
      let nMode = this.getAddNegPairMode();
      this.projectorScatterPlotAdapter.setAddNegPairMode(nMode);
      if (nMode == true) {
        if ((this.selectedPointIndices == []) && (this.getGroupMode())) {
          //this.setInfo("Please select a focus item first before adding or removing new pairs!");
          tf_tensorboard.notifyHelpChanged("Please select a focus item first before adding or removing new pairs!","projector");
        } else {
        //this.setInfo((this.getGroupMode()) ?
        //  "Select the items that are in negative relation with the focus item!" :
        //  "Select the first item in the new negative relation!");
          tf_tensorboard.notifyHelpChanged((this.getGroupMode()) ?
          "Select the items that are in negative relation with the focus item!" :
          "Select the first item in the new negative relation!","projector");
        }
        let pButton = this.getAddPosPairButton(),
            rButton = this.getRemovePairsButton();
        (pButton as any).active = (rButton as any).active = false;
      } else {
        //this.setInfo((this.getGroupMode()) ?
        //  "Welcome to the NIPGBoard! To get started, please select a Base Learner model and a dataset to create an embedding, or select an existing embedding for analysis!" :
        //  "Welcome to the NIPGBoard! To get started, please select a Base Learner model and a dataset to create an embedding, or select an existing embedding for analysis!");
          tf_tensorboard.notifyHelpChanged((this.getGroupMode()) ?
          "Welcome to the NIPGBoard! To get started, please select a Base Learner model and a dataset to create an embedding, or select an existing embedding for analysis!" :
          "Welcome to the NIPGBoard! To get started, please select a Base Learner model and a dataset to create an embedding, or select an existing embedding for analysis!","projector");
      }
      this.renderCanvas();
    })

    const removePairsButton = this.getRemovePairsButton();
    removePairsButton.addEventListener('click', () => {
      let rMode = this.getRemovePairsMode();
      this.projectorScatterPlotAdapter.setRemovePairMode(rMode);
      if (rMode == true) {
        if ((this.selectedPointIndices == []) && (this.getGroupMode())) {
          //this.setInfo("Please select a focus item first before adding or removing new pairs!");
          tf_tensorboard.notifyHelpChanged("Please select a focus item first before adding or removing new pairs!","projector");
        } else {
          //this.setInfo((this.getGroupMode()) ?
          //  "Select the items whose relation with the focus item you want to delete!" :
          //  "Select the first item of the relation you want to delete!");
            tf_tensorboard.notifyHelpChanged((this.getGroupMode()) ?
            "Select the items whose relation with the focus item you want to delete!" :
            "Select the first item of the relation you want to delete!","projector");
        }
        let pButton = this.getAddPosPairButton(),
            nButton = this.getAddNegPairButton();
        (pButton as any).active = (nButton as any).active = false;
      } else {
        //this.setInfo((this.getGroupMode()) ?
        //  "Welcome to the NIPGBoard! To get started, please select a Base Learner model and a dataset to create an embedding, or select an existing embedding for analysis!" :
        //  "Welcome to the NIPGBoard! To get started, please select a Base Learner model and a dataset to create an embedding, or select an existing embedding for analysis!");
          tf_tensorboard.notifyHelpChanged((this.getGroupMode()) ?
          "Welcome to the NIPGBoard! To get started, please select a Base Learner model and a dataset to create an embedding, or select an existing embedding for analysis!" :
          "Welcome to the NIPGBoard! To get started, please select a Base Learner model and a dataset to create an embedding, or select an existing embedding for analysis!","projector");
      }
      this.renderCanvas();
    })

    const removeAllButton = this.getRemoveAllButton();
    removeAllButton.addEventListener('click', () => {
      if(confirm("This will permanently remove all pairs, are you sure?")) {
        (document.querySelector('#pos1') as any).innerText = 0;
        (document.querySelector('#pos2') as any).innerText = 0;
        (document.querySelector('#neg1') as any).innerText = 0;
        (document.querySelector('#neg2') as any).innerText = 0;
        this.dataSet.sequences = [];
        this.renderCanvas();
        var xhr = new XMLHttpRequest();

        let projectorDataPanel = (document.querySelector("vz-projector-data-panel") as any);
        let selectedRun = projectorDataPanel.selectedRun.split("/")[1];

        xhr.open('GET', `${this.routePrefix}/deleteAll?subfolder=${tf_tensorboard.sublogdir}&selectedRun=${selectedRun}`);
        xhr.onload = () => {};
        xhr.send();
        //this.setInfo("Deleted all pairs! Contact administrator to restore changes.");
        tf_tensorboard.notifyHelpChanged("Deleted all pairs! Contact administrator to restore changes.","projector");
      }
    })

    const noNonSelectedButton = this.getNoNonSelectedButton();
    noNonSelectedButton.addEventListener('click', () => {
      let hMode = this.getNoNonSelectedMode();
      this.projectorScatterPlotAdapter.noNonSelected = hMode;
      if (hMode == true) {
        let hButton = this.getNoNonPairedButton();
        (hButton as any).active = false;
        this.projectorScatterPlotAdapter.noNonPaired = false;
      }
      this.renderCanvas();
    })

    const noNonPairedButton = this.getNoNonPairedButton();
    noNonPairedButton.addEventListener('click', () => {
      let hMode = this.getNoNonPairedMode();
      this.projectorScatterPlotAdapter.noNonPaired = hMode;
      if (hMode = true) {
        let hButton = this.getNoNonSelectedButton();
        (hButton as any).active = false;
        this.projectorScatterPlotAdapter.noNonSelected = false;
      }
      this.renderCanvas();
    })

    window.addEventListener('resize', () => {
      const container = this.parentNode as HTMLDivElement;
      container.style.height = document.body.clientHeight + 'px';
      this.projectorScatterPlotAdapter.resize();
    });

    {
      this.projectorScatterPlotAdapter = new ProjectorScatterPlotAdapter(
          this.getScatterContainer(), this as ProjectorEventContext);
      this.projectorScatterPlotAdapter.setLabelPointAccessor(
          this.selectedLabelOption);
    }

    this.projectorScatterPlotAdapter.setRoute(this.routePrefix);

    this.projectorScatterPlotAdapter.scatterPlot.onCameraMove(
        (cameraPosition: THREE.Vector3, cameraTarget: THREE.Vector3) =>
            this.bookmarkPanel.clearStateSelection());

    this.registerHoverListener(
        (hoverIndex: number) => this.onHover(hoverIndex));

    this.registerProjectionChangedListener((projection: Projection) =>
        this.onProjectionChanged(projection));

    this.registerSelectionChangedListener(
        (selectedPointIndices: number[],
         neighborsOfFirstPoint: knn.NearestEntry[]) =>
            this.onSelectionChanged(
                selectedPointIndices, neighborsOfFirstPoint));
    
    this.projectorScatterPlotAdapter.updateScatterPlotAttributes();
    this.projectorScatterPlotAdapter.render();
  }

  private onHover(hoverIndex: number) {
    this.hoverPointIndex = hoverIndex;
    let hoverText = null;
    if (hoverIndex != null) {
      const point = this.dataSet.points[hoverIndex];
      let imageWindow = document.querySelector('vz-image') as any;
      imageWindow.changePicture(point.metadata["Filename"]);
      if (point.metadata[this.selectedLabelOption]) {
        hoverText = point.metadata[this.selectedLabelOption].toString();
      }
    } else {
      let imageWindow = document.querySelector('vz-image') as any;
      imageWindow.changePicture("");
    }
    if (this.selectedPointIndices.length === 0) {
      //this.statusBar.style.display = hoverText ? null : 'none';
      //this.statusBar.innerText = hoverText;
    }
  }

  private getScatterContainer(): HTMLDivElement {
    return this.querySelector('#scatter') as HTMLDivElement;
  }

  private onSelectionChanged(
      selectedPointIndices: number[],
      neighborsOfFirstPoint: knn.NearestEntry[]) {
    this.selectedPointIndices = selectedPointIndices;
    this.neighborsOfFirstPoint = neighborsOfFirstPoint;
    this.dataPanel.onProjectorSelectionChanged(selectedPointIndices, 
        neighborsOfFirstPoint);
    let totalNumPoints =
        this.selectedPointIndices.length + neighborsOfFirstPoint.length;
    this.statusBar.innerText = `Selected ${totalNumPoints} points`;
    this.statusBar.style.display = totalNumPoints > 0 ? null : 'none';

    //Disable pair buttons if no element selected.
    let posB = this.getAddPosPairButton(),
        negB = this.getAddNegPairButton(),
        remB = this.getRemovePairsButton(),
        noElem = (selectedPointIndices.length == 0 || tf_tensorboard.disablepairwise) as boolean,
        opaq = (noElem ? 0.2 : 1);
    posB.style = `color:rgba(0,128,0,${opaq})`;
    negB.style = `color:rgba(255,0,0,${opaq})`;
    remB.style = `color:rgba(0,0,0,${opaq})`;
    posB.disabled = negB.disabled = remB.disabled = noElem;
  }

  onProjectionChanged(projection?: Projection) {
    this.dataPanel.projectionChanged(projection);
    this.projectorScatterPlotAdapter.updateScatterPlotAttributes();
    this.projectorScatterPlotAdapter.render();
  }

  setProjection(projection: Projection) {
    this.projection = projection;
    if (projection != null) {
      this.analyticsLogger.logProjectionChanged(projection.projectionType);
    }
    this.notifyProjectionChanged(projection);
    this.projectorScatterPlotAdapter.updateScatterPlotAttributes();
    this.projectorScatterPlotAdapter.render();
  }

  notifyProjectionPositionsUpdated() {
    this.projectorScatterPlotAdapter.notifyProjectionPositionsUpdated();
  }

  /*setInfo(text: string) {
    var card = (this.querySelector("#infoCard") as HTMLElement);
    card.innerText = text;
  }*/

  /**
   * Gets the current view of the embedding and saves it as a State object.
   */
  getCurrentState(): State {
    const state = new State();

    // Save the individual datapoint projections.
    state.projections = [];
    for (let i = 0; i < this.dataSet.points.length; i++) {
      const point = this.dataSet.points[i];
      const projections: {[key: string]: number} = {};
      const keys = Object.keys(point.projections);
      for (let j = 0; j < keys.length; ++j) {
        projections[keys[j]] = point.projections[keys[j]];
      }
      state.projections.push(projections);
    }
    state.selectedProjection = this.projection.projectionType;
    state.dataSetDimensions = this.dataSet.dim;
    state.tSNEIteration = this.dataSet.tSNEIteration;
    state.selectedPoints = this.selectedPointIndices;
    state.filteredPoints = this.dataSetFilterIndices;
    this.projectorScatterPlotAdapter.populateBookmarkFromUI(state);
    state.selectedColorOptionName = this.dataPanel.selectedColorOptionName;
    state.forceCategoricalColoring = this.dataPanel.forceCategoricalColoring;
    state.selectedLabelOption = this.selectedLabelOption;
    this.projectionsPanel.populateBookmarkFromUI(state);
    return state;
  }

  /** Loads a State object into the world. */
  loadState(state: State) {
    this.setProjection(null);
    {
      this.projectionsPanel.disablePolymerChangesTriggerReprojection();
      if (this.dataSetBeforeFilter != null) {
        this.resetFilterDataset();
      }
      if (state.filteredPoints != null) {
        this.filterDataset(state.filteredPoints);
      }
      this.projectionsPanel.enablePolymerChangesTriggerReprojection();
    }
    for (let i = 0; i < state.projections.length; i++) {
      const point = this.dataSet.points[i];
      const projection = state.projections[i];
      const keys = Object.keys(projection);
      for (let j = 0; j < keys.length; ++j) {
        point.projections[keys[j]] = projection[keys[j]];
      }
    }
    this.dataSet.hasTSNERun = (state.selectedProjection === 'tsne');
    this.dataSet.tSNEIteration = state.tSNEIteration;
    this.projectionsPanel.restoreUIFromBookmark(state);
    // this.inspectorPanel.restoreUIFromBookmark(state);
    this.dataPanel.selectedColorOptionName = state.selectedColorOptionName;
    this.dataPanel.setForceCategoricalColoring(
        !!state.forceCategoricalColoring);
    this.selectedLabelOption = state.selectedLabelOption;
    this.projectorScatterPlotAdapter.restoreUIFromBookmark(state);
    {
      const dimensions = stateGetAccessorDimensions(state);
      const components =
          getProjectionComponents(state.selectedProjection, dimensions);
      const projection = new Projection(
          state.selectedProjection, components, dimensions.length,
          this.dataSet);
      this.setProjection(projection);
    }
    this.notifySelectionChanged(state.selectedPoints);
  }

  /** Function to select manually for unit testing, instead of clicking on the canvas
   * @param start select each point from this ID
   * @param end select each point to and including this ID
  */
  selectPoints(start: number, end: number) {
    var points = [], i = start;
    while (i <= end) {
      points.push(i);
      i++;
    }
    this.notifySelectionChanged(points);
  }

  private imageSizeChange() {
    let value = this.imagesize;
    window.localStorage.setItem(tf_tensorboard.sublogdir+'-imagesize',value.toString());
  };

  createOutlier(id: number) {
    if(id) {
      this.dataSet.points[id].metadata["Clusters"] = "Unassigned"
      const outl = new XMLHttpRequest();
      outl.open('GET', `${this.routePrefix}/outlier?id=${id}`);
      outl.onload = () => {

      };
      outl.send();
    }
  }

  mergeClusters(id1: number, id2: number) {
    let clusterID1 = this.dataSet.points[id1].metadata["Clusters"];
    let clusterID2 = this.dataSet.points[id2].metadata["Clusters"];
    for (let point of this.dataSet.points) {
      if (point.metadata["Clusters"] === clusterID2) point.metadata["Clusters"] = clusterID1;
    }
    const merg = new XMLHttpRequest();
    merg.open('GET', `${this.routePrefix}/merge?id1=${id1}&id2=${id2}`);
    merg.onload = () => {

    };
    merg.send();
  }


}

document.registerElement(Projector.prototype.is, Projector);

}  // namespace vz_projector
