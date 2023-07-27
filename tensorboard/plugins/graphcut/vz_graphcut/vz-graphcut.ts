namespace vz_graphcut {

    export let GraphcutPolymer = PolymerElement({
        is: 'vz-graphcut',
        properties: {
            _gcfraction: {type: Number, value: 0.01, observer: 'handleFractionChanged', notify:true},
            _gccluster: {type: Number, value: window.localStorage.getItem('gccluster'), observer: 'handleClusterChanged', notify:true},
            _frac: {type:Number},
            _cm : {type: Array},
            _classes: {type: Array},
            _tick_marks: {type: Array},
            _disableExport: {type: Boolean, value: false}
        }
    });
    
    /** The instantiated class of the plotter */
    export class graphcut extends GraphcutPolymer {
        private _projector: any;
        private _port: string;
        private labelmap: any;
        private _reducerAlg: string;
        private _gcRunning: boolean;
        
        /** Runs initially when plugin becomes ready */
        ready() {
          this._gcRunning = false;
          if(tf_tensorboard.disableexport){
            this._disableExport = true;
          }
          const a = window.localStorage.getItem('gcfraction')
          const b = window.localStorage.getItem('gccluster')
          this._gcfraction = a;
          this._gccluster = b;

            //In case plugin isn't loaded from Dashboard, manually get the routePrefix.
            if (!(this.routePrefix)) {
                this.routePrefix = tf_backend.getRouter().pluginRoute('graphcut','');
            }

          /*let input = this.querySelector("#gcfracinput") as HTMLInputElement;
          input.addEventListener("onchange", function() {
            console.log(parseInt(this._gcfraction.toFixed(4)));
            input.value = this._gcfraction.toFixed(4);
          })*/

          setInterval( () => this.checkFrac(), 250);


            //Fetch the webservice port number.
            const xhr = new XMLHttpRequest();
            xhr.open('GET', `${this.routePrefix}/port`);
            xhr.onerror = (err) => {}
            xhr.onload = () => {
                this._port = xhr.responseText;
            }
            xhr.send();

            //Wait for the projector to be instantiated.
            let _ref = this;
            var callHandler = window.setInterval(function() {
                let projector = document.querySelector("vz-projector") as any;
                if ((projector) && (projector.dataSet) && (projector.dataSet.points) && (projector.dataSet.points.length > 0)) {
                    clearInterval(callHandler);
                    _ref._projector = projector;
                }
            })
        };

        async handleExport(){
          const accuracyButton = document.querySelector("#graphcutaccuracy");
          if(accuracyButton.innerHTML === ""){
            tf_tensorboard.handleAddNewNotification({title: "You didn't run graphcut!", icon: "close", plugin: "Graphcut"});
            return;
          }
          const runBtn = this.getGraphcutBtn();
          if(runBtn.disabled){
            tf_tensorboard.handleAddNewNotification({title: "Graphcut didn't finish yet!", icon: "close", plugin: "Graphcut"});
            return;
          }
          console.log(accuracyButton.innerHTML);
          const ul = document.querySelector("#accuracies");
          const accuracies = ul.querySelectorAll("li");
          console.log(accuracies);
          console.log(this._gccluster);
          console.log(this._gcfraction);
          const run = tf_tensorboard.selectedRun.split("__")
          const dataset = run[run.length-1]
          console.log(dataset);

          let filename = "";

          let txtFile = "";
          txtFile += `Date: ${new Date().toString()}\n`
          txtFile += `End-user: ${tf_tensorboard.username}\n`;
          txtFile += `Embedding: ${tf_tensorboard.selectedRun}\n`;
          txtFile += "PARAMETERS:\n";
          txtFile += `Fraction to keep: ${this._gcfraction}\n`;
          txtFile += `Clusters to detect: ${this._gccluster}\n`;
          txtFile += `Dataset: ${dataset}\n\n`;

          filename += `${this._reducerAlg}-`
          filename += `Clusters:${this._gccluster}-`

          if(this._reducerAlg === "PCA"){
            txtFile += `Dimension reduction algorithm: PCA\n`;
            let x = (document.getElementById("#input-41"))
            console.log(x);
            
            let y = (<HTMLInputElement>document.getElementById("#input-42")).value
            let z = (<HTMLInputElement>document.getElementById("#input-43")).value
            txtFile += ` - X: Component #${this._projector.projectionsPanel.pcaX+1}\n`;
            txtFile += ` - Y: Component #${this._projector.projectionsPanel.pcaY+1}\n`;
            txtFile += ` - Z: Component #${this._projector.projectionsPanel.pcaZ+1}\n`;
          }
          if(this._reducerAlg === "TSNE"){
            txtFile += `Dimension reduction algorithm: T-SNE\n`;
            txtFile += ` - Perplexity: ${document.querySelector("#perplexitySpan").innerHTML}\n`;
            txtFile += ` - Learning rate: ${document.querySelector("#learningRateSpan").innerHTML}\n`;
            txtFile += ` - Supervise: ${document.querySelector("#superviseSpan").innerHTML}\n`;
          }
          if(this._reducerAlg === "UMAP"){
            txtFile += `Dimension reduction algorithm: UMAP\n`;
            txtFile += ` - Neighbors: ${document.querySelector("#umapNeighborsSpan").innerHTML}\n`;
            filename += `Neighbors:${document.querySelector("#umapNeighborsSpan").innerHTML}-`
          }

          filename += `${accuracyButton.innerHTML}`
          txtFile += "RESULT\n";
          txtFile += `${accuracyButton.innerHTML}\n`;
          for(let i = 0; i < accuracies.length; i++){
            txtFile += `  - ${accuracies[i].innerHTML}\n`;
          }

          const organized = tf_tensorboard.selectedRun.includes("_organized_");
          let splitted_run = tf_tensorboard.selectedRun.split("_");
          //if(organized){
          if(splitted_run[1] == "organized") {
            console.log(splitted_run);
            let organizedindex = splitted_run.indexOf("organized");
            let name = splitted_run[0].split("/")[1]
            txtFile += `Weights: logdir/${tf_tensorboard.sublogdir}/model_${name}_organized_ver_${splitted_run[2]} (download link: ${window.location.href.split(":")[0]}:${window.location.href.split(":")[1]}:${this._port}/${tf_tensorboard.sublogdir}/model_${name}_organized_ver_${splitted_run[2]}?`+tf_tensorboard.username+"&"+tf_tensorboard.password +`)\n`;
          }
          //else if (!tf_tensorboard.selectedRun.includes("_paired_")) {
          else if(splitted_run[1] != "paired") {
            let first_part = splitted_run[0].split('/')
            const algorithmname = first_part[first_part.length - 1];
            txtFile += `Weights: logdir/${tf_tensorboard.sublogdir}/model${algorithmname} (download link: ${window.location.href.split(":")[0]}:${window.location.href.split(":")[1]}:${this._port}/${tf_tensorboard.sublogdir}/model_${algorithmname}?`+tf_tensorboard.username+"&"+tf_tensorboard.password +`)\n`;
          }

          txtFile += `Confusion matrix download link: ${window.location.href.split(":")[0]}:${window.location.href.split(":")[1]}:${this._port}/${tf_tensorboard.sublogdir}/confusion.png?`+tf_tensorboard.username+"&"+tf_tensorboard.password +`\n`;

          for(let i = 0; i < this._classes.length; i++){
            txtFile += `${this._classes[i]}\t`
            for(let j = 0; j < this._classes.length; j++){
              txtFile += `${this._cm[i][j]}\t`
            }
            txtFile += `\n`
          }
          txtFile += `\t`;
          for(let i = 0; i < this._classes.length; i++){
            txtFile += `${this._classes[i]}\t`
          }
    
          const textBlob = new Blob([txtFile], {type: 'text/plain'});
          let blobURL = URL.createObjectURL(textBlob);
          var a = document.createElement("a");
          a.href = blobURL;
          a.download = filename + ".txt"
          a.click();
        };

        handleRunGc(): void{

            if(this._gcRunning) {
              tf_tensorboard.handleAddNewNotification({title: "There is already a graph cut running, please run it again once the previous finishes!", icon: "close", plugin: "Graphcut"});
              return;
            }
            this._gcRunning = true;

            if(!(this._projector)) {
              tf_tensorboard.handleAddNewNotification({title: "There's no selected embedding!", icon: "close", plugin: "Graphcut"});
              this._gcRunning = false;
              return;
            }

            if(!(this._projector.dataSet)) {
              tf_tensorboard.handleAddNewNotification({title: "There's no selected dataset!", icon: "close", plugin: "Graphcut"});
              this._gcRunning = false;
              return;
            }
          
            const btn = this.getGraphcutBtn();
            btn.disabled = true;
            let numOfDimension:number = 0;

            tf_tensorboard.handleAddNewNotification({title: "Graphcut started!", icon: "hourglass-empty", plugin: "Graphcut"}); 
            
            this.labelmap = [];
            for(var elem of this._projector.dataSet.points) {
                this.labelmap.push(elem.metadata["Label"]);
            }
            let numOfPoints = this._projector.dataSet.points.length;
            let pcaVectors: number[][] = new Array(numOfPoints);

            //PCA is the selected
            if ((this._projector.dataSet.mode == "PCA") || (this._projector.dataSet.mode == "pca")) {
                this._reducerAlg = "PCA";
                numOfDimension = 10;
                let NUM_PCA_COMPONENTS = 10;
                let points = this._projector.dataSet.points;
                for (let i = 0; i < numOfPoints; i++) {
                    let newV = new Array(NUM_PCA_COMPONENTS);
                    for (let d = 0; d < NUM_PCA_COMPONENTS; d++) {
                        let label = 'pca-' + d;
                        newV[d] = points[i].projections[label];
                    }
                    pcaVectors[i] = newV;
                }

                
            }
            //TSNE is the selected
            else if ((this._projector.dataSet.mode == "TSNE") || (this._projector.dataSet.mode == "tsne"))
            {
                this._reducerAlg = "TSNE";
                numOfDimension = 3;
                if(this._projector.dataSet.hasTSNERun == false &&
                    this._projector.dataSet.tSNEShouldStop
                    && this._projector.dataSet.tSNEShouldPause==false &&
                    this._projector.dataSet.tSNERunOnce == false){
             
                      tf_tensorboard.handleAddNewNotification({title: "Run T-SNE first!", icon: "close", plugin: "Graphcut"});
                     return;
                    }
             
                 if (this._projector.dataSet.tSNEShouldStop==false && this._projector.dataSet.tSNEShouldPause==false) {
                   tf_tensorboard.handleAddNewNotification({title: "You should pause T-SNE first!", icon: "close", plugin: "Graphcut"});
                   return;
                 }
             
                 let points = this._projector.dataSet.points;
             
                 let numOfPoints = points.length;
                 let labels: string[] = new Array(numOfPoints);
                 let NUM_TSNE_DIM = this._projector.dataSet.tSNEDim;
             
                 for (let i = 0; i < numOfPoints; i++) {
                   let newV = new Array(NUM_TSNE_DIM);
                   for (let d = 0; d < NUM_TSNE_DIM; d++) {
                     let label = 'tsne-' + d;
                     newV[d] = points[i].projections[label];
                   }
             
                   pcaVectors[i] = newV;
                   if (points[i].metadata.Label) {
                     labels[i] = points[i].metadata.Label.toString();
                   } else {
                     labels[i] = "undefined";
                   }
                 }
            }
            //UMAP is the selected
            else if ((this._projector.dataSet.mode == "UMAP") || (this._projector.dataSet.mode == "umap"))
            {
              this._reducerAlg = "UMAP";
                numOfDimension = 3;
                if (!this._projector.dataSet.projections['umap']) {
                    tf_tensorboard.handleAddNewNotification({title: "Run UMAP first!", icon: "close", plugin: "Graphcut"});
                    return;
                  }
              
                    let points = this._projector.dataSet.points;
              
                    let numOfPoints = points.length;
                    let labels: string[] = new Array(numOfPoints);
                    let NUM_UMAP_DIM = this._projector.dataSet.uMAPDim;
              
                    for (let i = 0; i < numOfPoints; i++) {
                      let newV = new Array(NUM_UMAP_DIM);
                      for (let d = 0; d < NUM_UMAP_DIM; d++) {
                        let label = 'umap-' + d;
                        newV[d] = points[i].projections[label];
                      }
              
                      pcaVectors[i] = newV;
                      if (points[i].metadata.Label) {
                        labels[i] = points[i].metadata.Label.toString();
                      } else {
                        labels[i] = "undefined";
                      }
                    }
            }
            
            const gcut = new XMLHttpRequest();
            
            gcut.open('POST', `${this.routePrefix}/clusters?fraction=${this._gcfraction}&count=${this._gccluster}&sublogdir=${tf_tensorboard.sublogdir}&dimension=${numOfDimension}`);
            gcut.onload = () => {
                console.log("Graph cut finished");
                let results = JSON.parse(gcut.responseText);
                let node_groups = results['groups'];
                this._classes = results['classes']
                this._cm = results['cm']
                this._tick_marks = results['tick_marks']
                console.log(this._classes);
                console.log(this._cm);
                console.log(this._tick_marks);
                
                let lengths = []
                for(let i of node_groups){
                  if(lengths.length > 0){
                    lengths.push(i.length+lengths[lengths.length-1])
                  }
                  else{
                    lengths.push(i.length)
                  }
                }

                console.log("LENGTHS:",lengths);

                let idx = []
                for (let row of node_groups) for (let e of row) idx.push(parseInt(e));
                this.showClusters(results['nodes'], idx, lengths);
                document.querySelector("#graphcutaccuracy").innerHTML = "Accuracy: " + results['accuracy'].toString();
                const gcim = (document.querySelector("#graphcutimage") as any);
                gcim.src = " ";

                //expanding accuracy list toggles open-close
                var coll = document.querySelector(".collapsible");
                coll.addEventListener("click", function() {
                  this.classList.add("added");
                  this.classList.toggle("active");
                  var content = this.nextElementSibling;
                  if (content.style.maxHeight){
                    content.style.maxHeight = null;
                  } else {
                    content.style.maxHeight = content.scrollHeight + "px";
                  } 
                });	 

                //display each cluster's accuracy in the expanding list
                let accuracies = document.getElementById("accuracies");
                accuracies.innerHTML = "";
                for (let acc of Object.keys(results['accuracies'])) {
                  let li = document.createElement("li");
                  let clusterName = acc;
                  let index = Object.keys(results['accuracies']).indexOf(acc);
                  let clusterAcc = results['accuracies'][acc]
                  li.innerHTML = clusterName + ' (' + node_groups[index].length + ' elements) ' + ": " + clusterAcc.toFixed(2);
                  accuracies.appendChild(li);
                }

                btn.disabled = false;
                gcim.src = (tf_tensorboard.sublogdir == " ") ? ("http://" + window.location.hostname + ":" + this._port + "/" + "confusion.png?"+tf_tensorboard.username+"&"+tf_tensorboard.password + "&" + new Date().getTime().toString()) : ("http://" + window.location.hostname + ":" + this._port + "/" + tf_tensorboard.sublogdir + "/" + "confusion.png?"+tf_tensorboard.username+"&"+tf_tensorboard.password + "&" + new Date().getTime().toString());
                if(!tf_tensorboard.unifyandoutlier){
                  this._projector.getUnifyClusterButton().style.color = "rgba(0,255,0)";
                  this._projector.getUnifyClusterButton().disabled = false;
                  this._projector.getCreateOutlierButton().style.color = "rgba(255,0,0)";
                  this._projector.getCreateOutlierButton().disabled = false;
                }
                ////this.setInfo("Graph Cut results are not updated or are missing. Please run the Graph Cut algorithm.");
                tf_tensorboard.notifyHelpChanged("Graphcut finished!","graphcut");
                tf_tensorboard.handleAddNewNotification({title: "Graphcut finished!", icon: "check", plugin: "Graphcut"}); 
                this._gcRunning = false;
            }
            let obj = {'labels': this.labelmap, 'vectors': pcaVectors};
            gcut.send(JSON.stringify(obj));
        };

        /** Runs after fraction slider changes, display the slider's value in text.*/
        handleFractionChanged():void{
            //window.localStorage.setItem('gcfraction', this._gcfraction);
            //const gcfractext = this.querySelector('#gcfractext');
            //(gcfractext as any).innerHTML = `Fraction to keep: ${this._gcfraction.toFixed(4)}`;
            //let input = this.querySelector("#gcfracinput") as HTMLInputElement;
            //input.value = this._gcfraction.toFixed(4);
            //console.log("inf??")
        };

        checkFrac() {
          let input = this.querySelector("#gcfracinput") as HTMLInputElement;
          let i = parseFloat(input.value);

          let r = parseFloat(this._gcfraction.toFixed(4));

          if (i != this._frac && i>=0.0001 && i <= 0.1) {
            this._frac = i;
            this._gcfraction = i;
            const gcfractext = this.querySelector('#gcfractext');
            (gcfractext as any).innerHTML = `Fraction to keep: ${this._gcfraction.toFixed(4)}`;

          }
          else if (r != this._frac && r >= 0.0001 && r <= 0.1) {
            this._frac = r;
            input.value = r.toFixed(4);
            const gcfractext = this.querySelector('#gcfractext');
            (gcfractext as any).innerHTML = `Fraction to keep: ${r.toFixed(4)}`;
          }
        };

        /** Runs after number of clusters slider changes, display the slider's value in text.*/
        handleClusterChanged():void{
            window.localStorage.setItem('gccluster', this._gccluster);
            const gcclustext = this.querySelector('#gcclustext');
            (gcclustext as any).innerHTML = `Clusters to detect: ${this._gccluster}`;
        };

        private getGraphcutBtn(): any {
          return this.querySelector('#graphcut')
        }      

        /** Add new "Cluster" column to the metadata */
        showClusters(clusters, idx, lengths) {
          let i = 0;
          let group = 0;
          for (let id of idx){
            if(i < lengths[group]){
              console.log(id, group.toString() + '_' + clusters[i]);
              if (this._projector.dataSet) {
              this._projector.dataSet.points[id].metadata["Clusters"] = group.toString() + '_' + clusters[i];
              }
              i++;     
            }
            else{
              group++;
            }
          }

          //for(let clusterID in clusters) {
          //  for(let point of clusters[clusterID]) {
          //    this.dataSet.points[point].metadata["Cluster"] = clusterID;
          //  }
          //}
        }
    };
    
    document.registerElement(graphcut.prototype.is, graphcut);

    } //namespace