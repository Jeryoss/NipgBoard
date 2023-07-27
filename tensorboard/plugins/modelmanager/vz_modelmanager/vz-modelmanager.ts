namespace vz_modelmanager {

export let ModelmanagerPolymer = PolymerElement({
  is: 'vz-modelmanager',
  properties: {
    routePrefix: String,
    pageViewLogging: Boolean,
    eventLogging: Boolean,
    destination: String,
    selectedRun: { type: Number, observer: "selectedRunChange", notify: true },
    selectedBase: {type: Number, observer: "selectedBaseChanged", notify: true},
    selectedTraining: {type: Number, observer: "selectedTrainingChanged", notify: true},
    selectedDatabase: {type: Number, observer: "selectedDatabaseChanged", notify: true},
  }
});

/** This is the class responsible for the window that enables switching between projector views or running
 *  the training script from the front-end.
*/

export class Modelmanager extends ModelmanagerPolymer {
  private algorithms: any;          //The name of neural networks to the dropdown menu
  private selectedAlgorithm: number;    //The selected neural network algorithm
  private runs: any;
  private bases: any;
  private baseIDs: any;
  private trainings: any;
  //private videotrainings:any;
  private trainings_all: any;
  private databases: any;
  //private cnfinfo: any;
  //private numnames: number;
  private projector: any;      //This refers to the invisible old Data panel which we manipulate.
  //private projectorMain: any;  //This refers to the entire Projector itslef.
  private vggbutton: any;      //The DEFAULT button.
  private kirabutton: any;     //The TRAINING button.
  //private _selectedRunTest: string;
  private config: {};          //The configuraton data from logdir/cnf.json
  //private runnable: string;
  /** Runs when the class is instantiated. */
  ready() {
    if(tf_tensorboard.disableclustering){
      this.runnable = 'true';
    }
    this.routePrefix = tf_backend.getRouter().pluginRoute('modelmanager', '');

    let ret = this;
    var callHandler = window.setInterval(function() {
      //This plugin must not do anything substantial until the Data panel has been instantiated.
      ret.projector = (document.querySelector("vz-projector-data-panel") as any);
      if (ret.projector) {
          clearInterval(callHandler);
          //ret.projectorMain = (document.querySelector("vz-projector") as any);
          ret.getConfig();
      }

    ret.selectedAlgorithm = Number(window.localStorage.getItem(tf_tensorboard.sublogdir+'-algorithm'));
    if (window.localStorage.getItem(tf_tensorboard.sublogdir+'-algorithm') === null) {
      ret.selectedAlgorithm = -1;
    }
    

    }, 2)
  };

  selectedRunChange(){
    this.projector.selectedRun = this.runs[this.selectedRun];
  }

  /** Fetch the configuration file from the server and check all the fields for correct values. */
  getConfig(): void {
     const xhr = new XMLHttpRequest();
     xhr.open('GET', `${this.routePrefix}/cnf?subfolder=${tf_tensorboard.sublogdir}`);
     xhr.onerror = (err) => {
       //this.projectorMain.setInfo("Couldn't fetch configuration file, please try refreshing the website or contact an administrator!");
       tf_tensorboard.notifyHelpChanged("Couldn't fetch configuration file, please try refreshing the website or contact an administrator!","modelmanager");
       tf_tensorboard.handleAddNewNotification({title: "Couldn't fetch configuration file, please try refreshing the website or contact an administrator!", icon: "close", plugin: "Modelmanager"}); 
     }
     //TODO:: újra az egészet
     xhr.onload = () => {
       if(xhr.status == 500) {
          let error_msg = xhr.responseText;
          //this.projectorMain.setInfo(`${error_msg} Contact administrator!`);
          tf_tensorboard.notifyHelpChanged(`${error_msg} Contact administrator!`,"modelmanager");
          tf_tensorboard.handleAddNewNotification({title: `${error_msg} Contact administrator!`, icon: "close", plugin: "Modelmanager"}); 
          alert(`${error_msg} Contact administrator!`);
       } else {
          this.config = (JSON.parse(xhr.responseText));
          if(!(this.config["default"])) {
            //this.projectorMain.setInfo("Incorrect config file, missing default! Contact administrator!");
            tf_tensorboard.notifyHelpChanged("Incorrect config file, missing default! Contact administrator!","modelmanager");
            tf_tensorboard.handleAddNewNotification({title: "Incorrect config file, missing default! Contact administrator!", icon: "close", plugin: "Modelmanager"}); 
            alert("Incorrect config file, missing default! Contact administrator!");
          } else if (!(this.config["trainings"])) {
            //this.projectorMain.setInfo("Incorrect config file, missing trainings! Contact administrator!");
            tf_tensorboard.notifyHelpChanged("Incorrect config file, missing trainings! Contact administrator!","modelmanager");
            tf_tensorboard.handleAddNewNotification({title: "Incorrect config file, missing trainings! Contact administrator!", icon: "close", plugin: "Modelmanager"}); 
            alert("Incorrect config file, missing trainings! Contact administrator!");
          } else if (this.config["default"]["embedding_folder"] === '') {
            //this.projectorMain.setInfo("Missing default embedding data in configuration file! Contact administrator!");
            tf_tensorboard.notifyHelpChanged("Missing default embedding data in configuration file! Contact administrator!","modelmanager");
            tf_tensorboard.handleAddNewNotification({title: "Missing default embedding data in configuration file! Contact administrator!", icon: "close", plugin: "Modelmanager"}); 
            alert("Missing data in configuration file! Contact administrator!");
          }

          let count = this.config["trainings"].length;
          for(let i = 0; i < count; i++){

          if (this.config["trainings"][i]["embedding_folder"] === '') {
            //this.projectorMain.setInfo("Missing training output in configuration file! Contact administrator!");
            tf_tensorboard.notifyHelpChanged("Missing training output in configuration file! Contact administrator!","modelmanager");
            tf_tensorboard.handleAddNewNotification({title: "Missing training output in configuration file! Contact administrator!", icon: "close", plugin: "Modelmanager"}); 
            alert("Missing training output in configuration file! Contact administrator!");
          } else if (this.config["trainings"][i]["algorithm_path"] === '') {
            //this.projectorMain.setInfo("Missing path for executable module / algorithm! Contact administrator!");
            tf_tensorboard.notifyHelpChanged("Missing path for executable module / algorithm! Contact administrator!","modelmanager");
            tf_tensorboard.handleAddNewNotification({title: "Missing path for executable module / algorithm! Contact administrator!", icon: "close", plugin: "Modelmanager"}); 
            alert("Missing path for executable module / algorithm! Contact administrator!");
          } else if (!(this.config["trainings"][i]["algorithm"])) {
            //this.projectorMain.setInfo("Missing module / executable configuration! Contact administrator!");
            tf_tensorboard.notifyHelpChanged("Missing module / executable configuration! Contact administrator!","modelmanager");
            tf_tensorboard.handleAddNewNotification({title: "Missing module / executable configuration! Contact administrator!", icon: "close", plugin: "Modelmanager"}); 
            alert("Missing module / executable configuration! Contact administrator!");
          } else if (this.config["trainings"][i]["algorithm"]["file"] === '') {
            //this.projectorMain.setInfo("Missing executable module name! Contact administrator!");
            tf_tensorboard.notifyHelpChanged("Missing executable module name! Contact administrator!","modelmanager");
            tf_tensorboard.handleAddNewNotification({title: "Missing executable module name! Contact administrator!", icon: "close", plugin: "Modelmanager"}); 
            alert("Missing executable module name! Contact administrator!");
          } else if (this.config["trainings"][i]["algorithm"]["callable"] === '') {
            //this.projectorMain.setInfo("Missing callable name! Contact administrator!");
            tf_tensorboard.notifyHelpChanged("Missing callable name! Contact administrator!","modelmanager");
            tf_tensorboard.handleAddNewNotification({title: "Missing callable name! Contact administrator!", icon: "close", plugin: "Modelmanager"}); 
            alert("Missing callable name! Contact administrator!");
          } else {
            if(this.config["default"]["image_folders"]) {
              this.doubleCheckSpriteAndMetadata();
            }
          }

        }
       }
     };
     xhr.send();
  }

  /** Generate any missing metadata.tsv or sprite.png files in the image folder, then alert the Projector. */
  doubleCheckSpriteAndMetadata() {
    const xhr = new XMLHttpRequest();
    xhr.open('GET', `${this.routePrefix}/metacheck`);
    xhr.onerror = () => {
      //this.projectorMain.setInfo("Couldn't find or create sprite/metadata files! Contact administrator!");
      tf_tensorboard.notifyHelpChanged("Couldn't find or create sprite/metadata files! Contact administrator!","modelmanager");
      tf_tensorboard.handleAddNewNotification({title: "Couldn't find or create sprite/metadata files! Contact administrator!", icon: "close", plugin: "Modelmanager"}); 
      alert("Couldn't find or create sprite/metadata files! Contact administrator!");
    }
    xhr.onload = () => {
      let ret = this;
      let count = 0;
      for(let i in ret.config["trainings"]){
        count++;
      }

      let namesd = [];
      ret.trainings_all = [];
      //ret.videotrainings = [];
      ret.baseIDs = [];
      for(let j = 0; j < count; j++){
        if(ret.config["trainings"][j]["type"] == "base" || ret.config["trainings"][j]["type"] == "video"){
          let base_name = ret.config["trainings"][j]["name"];
          let base = ret.config["trainings"][j]["embedding_folder"];
          namesd.push(base_name);
          ret.baseIDs.push(base);
          ret.trainings_all[base] = [];
          for(let i of ret.config["runs"]) {
            if ((i as String).startsWith(base)) ret.trainings_all[base].push(i);
          }
        }
        /*if(ret.config["trainings"][j]["type"] == "video"){
          let base_name = ret.config["trainings"][j]["name"];
          ret.videotrainings.push(j);
        }*/
      }
      //console.log("video algs");
      //console.log(ret.videotrainings);
      ret.algorithms = namesd;
      ret.bases = ret.algorithms;
      ret.databases = ret.config["default"]["image_folders"];
      
      var callHandler = window.setInterval(function() {
          let im = (document.querySelector("vz-image") as any),
              se = (document.querySelector("vz-selected") as any),
              vd = (document.querySelector("vz-video") as any);
          if ((im) && (se) && (vd)) {
            //TODO: migrate this to the individual plugins. Each plugin should read whatever they need
            //from the configuration file.
            clearInterval(callHandler);

            im.setPath(ret.config["default"]["image_folders"][0]);
            se.setPath(ret.config["default"]["image_folders"][0]);
            vd.setPath(ret.config["default"]["video_folders"][0]);

          }
      }, 2);

      //Pass the embedding names so the Projector can check their validity.
      let train_folders = []
      for(let i in ret.config['trainings']){
        train_folders.push(ret.config['trainings'][i]['embedding_folder']);
      }

      this.projector.sendNames(ret.config["default"]["embedding_folder"], train_folders);
      
      //Wait for the projector to find them and set the correct default embedding to view.
      var callHandler2 = window.setInterval(function() {
        let sRun = ret.projector.selectedRun;
        if (sRun) {
          clearInterval(callHandler2);
          ret.setButtons();
        }
      }, 2);
    }
    xhr.send();
  }


  //Set the button visuals according to the configurations.
  setButtons(): void {
    //this._selectedRunTest = this.projector.selectedRun;


    //DATABASES
    //this.databases = ["melanoma", "traffic"];

    //GET runs
    this.runs = this.projector.runNames;
    
    //SET bases + get training info results from the cnf
    //this.cnfinfo =  { "traffic_densenet": ["base", "traffic_train1", "traffic_train2", "traffic_train3"], "melanoma_vgg": ["base","melanoma_train1", "melanoma_train2"] }
    //this.bases = Object.keys(this.cnfinfo);
    


    this.selectedRun = this.runs.indexOf(this.projector.selectedRun);
  }

  selectedBaseChanged(): void{
    //this.projector.selectedRun = tf_tensorboard.sublogdir+'/'+this.selectedBase;
    this.trainings = this.trainings_all[this.baseIDs[this.selectedBase]];
    if (this.trainings.length == 0) alert("No embeddings found for " + this.bases[this.selectedBase] + ", start Clustering.");
    this.selectedTraining = -1;
  }

  selectedTrainingChanged(): void{
    if(this.selectedTraining != -1) {
      let run = tf_tensorboard.sublogdir + '/' + this.trainings[this.selectedTraining];
      this.projector.selectedRun = run;
      let temp = run.split('/');
      let dbase = temp[temp.length - 1];
      let temp2 = dbase.split('__');
      let temp3 = temp2[temp2.length-1];

      let executer = document.querySelector('vz-executer') as any;
      executer.selectedDatabase = this.databases.indexOf(temp3)
      let im = (document.querySelector("vz-image") as any),
          se = (document.querySelector("vz-selected") as any);
      if ((im) || (se)) {
        im.setPath(temp3);
        se.setPath(temp3);
      }
    }
  }

  selectedDatabaseChanged(): void{
    let executer = document.querySelector('vz-executer') as any;
    executer.selectedDatabase = this.selectedDatabase;
  }
  
  //Show the DEFAULT embedding.
  showVGG(): void {
    if(tf_tensorboard.sublogdir == ' '){
      this.projector.selectedRun = this.config["default"]["embedding_folder"];
    }else{
      this.projector.selectedRun = tf_tensorboard.sublogdir + '/' + this.config["default"]["embedding_folder"];
    }

    this.vggbutton.style["background-color"] = "#29b6f6";
    this.kirabutton.style["background-color"] = "transparent";
  }

  //Show the TRAINING embedding.
  showTrained(): void {
    this.projector.selectedRun = this.projector.completedalgorithm;
    this.vggbutton.style["background-color"] = "transparent";
    this.kirabutton.style["background-color"] = "#29b6f6";
  }

  //Run the training algorithm.
  runTraining(): void {

    if(this.selectedAlgorithm == -1){
      tf_tensorboard.handleAddNewNotification({title: "Select algorithm first!", icon: "close", plugin: "Modelmanager"}); 
      alert('Select algorithm first!');
      return;
    }

    //save the algorithm as a cookie
    window.localStorage.setItem(tf_tensorboard.sublogdir+'-algorithm', this.selectedAlgorithm.toString());

    //this.projectorMain.setInfo("Execution has started!");
    tf_tensorboard.notifyHelpChanged("Execution has started!","modelmanager");
    tf_tensorboard.handleAddNewNotification({title: "Execution has started!", icon: "hourglass-empty", plugin: "Modelmanager"}); 

    let dsToRun = this.selectedDatabase;
    const xhr = new XMLHttpRequest();
    xhr.open('GET', `${this.routePrefix}/execute?boardPath=${tf_tensorboard.boardPath}&imagePath=${this.config["default"]["image_folders"][dsToRun]}&num=${this.selectedAlgorithm}&subfolder=${tf_tensorboard.sublogdir}`);
    xhr.onerror = (err) => {
      //this.projectorMain.setInfo("Execution has failed! ERROR: Network error. Please contact administrator.");
      tf_tensorboard.notifyHelpChanged("Execution has failed! ERROR: Network error. Please contact administrator.","modelmanager");
      tf_tensorboard.handleAddNewNotification({title: "Execution has failed! ERROR: Network error. Please contact administrator.", icon: "close", plugin: "Modelmanager"}); 
    };
    xhr.onload = () => {
      if (xhr.status == 500) {
        let error_msg = xhr.responseText;
        //this.projectorMain.setInfo(`Execution has failed! ${error_msg}`);
        tf_tensorboard.notifyHelpChanged(`Execution has failed! ${error_msg}`,"modelmanager");
        tf_tensorboard.handleAddNewNotification({title: `Execution has failed! ${error_msg}`, icon: "close", plugin: "Modelmanager"}); 
        alert(`Execution has failed! ${error_msg}`);
      } else {
        //this.projectorMain.setInfo("Execution has finished! Reload the website to see the results.");
        tf_tensorboard.notifyHelpChanged("Execution has finished! Reload the website to see the results.","modelmanager");
        tf_tensorboard.handleAddNewNotification({title: "Execution has finished! Reload the website to see the results.", icon: "check", plugin: "Modelmanager"}); 
        if(confirm("Execution has finished! Would you like to reload the page now to see the results?")) {
          window.location.reload();
        }
      }
    }
    xhr.send();
  }
}

document.registerElement(Modelmanager.prototype.is, Modelmanager);

}  // namespace tf_modelmanager

