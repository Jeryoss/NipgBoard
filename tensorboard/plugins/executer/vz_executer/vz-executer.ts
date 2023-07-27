namespace vz_executer {

    export let ExecuterPolymer = PolymerElement({
        is: 'vz-executer',
        properties: {
            routePrefix: String,
            pageViewLogging: Boolean,
            eventLogging: Boolean,
            destination: String,
        }
    });

    /** This is the class responsible for the window that enables switching between projector views or running
     *  the training script from the front-end.
     */

    export class Executer extends ExecuterPolymer {
        public selectedDatabase: string;
        private selecteditem: number;
        private numnames: number;
        private projector: any;      //This refers to the invisible old Data panel which we manipulate.
        private projectorMain: any;  //This refers to the entire Projector itslef.
        private _selectedRunTest: string;
        private config: {};          //The configuraton data from logdir/cnf.json
        private names: any;          //The name of neural networks to the dropdown menu
        private runnable: string;

        /** Runs when the class is instantiated. */
        ready() {
            console.log(tf_tensorboard.sublogdir);

            if (tf_tensorboard.disabletraining) {
                this.runnable = 'true';
            }
            this.routePrefix = tf_backend.getRouter().pluginRoute('executer', '');

            let ret = this;
            var callHandler = window.setInterval(function () {
                //This plugin must not do anything substantial until the Data panel has been instantiated.
                ret.projector = (document.querySelector("vz-projector-data-panel") as any);
                if (ret.projector) {
                    clearInterval(callHandler);
                    ret.projectorMain = (document.querySelector("vz-projector") as any);
                    ret.getConfig();
                }

                ret.selecteditem = Number(window.localStorage.getItem(tf_tensorboard.sublogdir + '-algorithm'));
                if (window.localStorage.getItem(tf_tensorboard.sublogdir + '-algorithm') === null) {
                    ret.selecteditem = -1;
                }


            }, 2)
        };

        /** Fetch the configuration file from the server and check all the fields for correct values. */
        getConfig(): void {
            const xhr = new XMLHttpRequest();
            xhr.open('GET', `${this.routePrefix}/cnf?subfolder=${tf_tensorboard.sublogdir}`);
            xhr.onerror = (err) => {
                ////this.projectorMain.setInfo("Couldn't fetch configuration file, please try refreshing the website or contact an administrator!");
                tf_tensorboard.notifyHelpChanged("Couldn't fetch configuration file, please try refreshing the website or contact an administrator!", "executer");
                tf_tensorboard.handleAddNewNotification({
                    title: "Couldn't fetch configuration file, please try refreshing the website or contact an administrator!",
                    icon: "close",
                    plugin: "Executer"
                }); // waiting
            }
            //TODO:: újra az egészet
            xhr.onload = () => {
                if (xhr.status == 500) {
                    let error_msg = xhr.responseText;
                    ////this.projectorMain.setInfo(`${error_msg} Contact administrator!`);
                    tf_tensorboard.notifyHelpChanged(`${error_msg} Contact administrator!`, "executer");
                    tf_tensorboard.handleAddNewNotification({
                        title: `${error_msg} Contact administrator!`,
                        icon: "close",
                        plugin: "Executer"
                    });
                    alert(`${error_msg} Contact administrator!`);
                } else {
                    this.config = (JSON.parse(xhr.responseText));
                    if (!(this.config["default"])) {
                        ////this.projectorMain.setInfo("Incorrect config file, missing default! Contact administrator!");
                        tf_tensorboard.notifyHelpChanged("Incorrect config file, missing default! Contact administrator!", "executer");
                        tf_tensorboard.handleAddNewNotification({
                            title: "Incorrect config file, missing default! Contact administrator!",
                            icon: "close",
                            plugin: "Executer"
                        });
                        alert("Incorrect config file, missing default! Contact administrator!");
                    } else if (!(this.config["trainings"])) {
                        ////this.projectorMain.setInfo("Incorrect config file, missing trainings! Contact administrator!");
                        tf_tensorboard.notifyHelpChanged("Incorrect config file, missing trainings! Contact administrator!", "executer");
                        tf_tensorboard.handleAddNewNotification({
                            title: "Incorrect config file, missing trainings! Contact administrator!",
                            icon: "close",
                            plugin: "Executer"
                        });
                        alert("Incorrect config file, missing trainings! Contact administrator!");
                    } else if (this.config["default"]["embedding_folder"] === '') {
                        ////this.projectorMain.setInfo("Missing default embedding data in configuration file! Contact administrator!");
                        tf_tensorboard.notifyHelpChanged("Missing default embedding data in configuration file! Contact administrator!", "executer");
                        tf_tensorboard.handleAddNewNotification({
                            title: "Missing default embedding data in configuration file! Contact administrator!",
                            icon: "close",
                            plugin: "Executer"
                        });
                        alert("Missing data in configuration file! Contact administrator!");
                    }

                    let count = this.config["trainings"].length;
                    for (let i = 0; i < count; i++) {

                        if (this.config["trainings"][i]["embedding_folder"] === '') {
                            //this.projectorMain.setInfo("Missing training output in configuration file! Contact administrator!");
                            tf_tensorboard.notifyHelpChanged("Missing training output in configuration file! Contact administrator!", "executer");
                            tf_tensorboard.handleAddNewNotification({
                                title: "Missing training output in configuration file! Contact administrator!",
                                icon: "close",
                                plugin: "Executer"
                            }); // waiting
                            alert("Missing training output in configuration file! Contact administrator!");
                        } else if (this.config["trainings"][i]["algorithm_path"] === '') {
                            //this.projectorMain.setInfo("Missing path for executable module / algorithm! Contact administrator!");
                            tf_tensorboard.notifyHelpChanged("Missing path for executable module / algorithm! Contact administrator!", "executer");
                            tf_tensorboard.handleAddNewNotification({
                                title: "Missing path for executable module / algorithm! Contact administrator!",
                                icon: "close",
                                plugin: "Executer"
                            });
                            alert("Missing path for executable module / algorithm! Contact administrator!");
                        } else if (!(this.config["trainings"][i]["algorithm"])) {
                            //this.projectorMain.setInfo("Missing module / executable configuration! Contact administrator!");
                            tf_tensorboard.notifyHelpChanged("Missing module / executable configuration! Contact administrator!", "executer");
                            tf_tensorboard.handleAddNewNotification({
                                title: "Missing module / executable configuration! Contact administrator!",
                                icon: "close",
                                plugin: "Executer"
                            });
                            alert("Missing module / executable configuration! Contact administrator!");
                        } else if (this.config["trainings"][i]["algorithm"]["file"] === '') {
                            //this.projectorMain.setInfo("Missing executable module name! Contact administrator!");
                            tf_tensorboard.notifyHelpChanged("Missing executable module name! Contact administrator!", "executer");
                            tf_tensorboard.handleAddNewNotification({
                                title: "Missing executable module name! Contact administrator!",
                                icon: "close",
                                plugin: "Executer"
                            });
                            alert("Missing executable module name! Contact administrator!");
                        } else if (this.config["trainings"][i]["algorithm"]["callable"] === '') {
                            //this.projectorMain.setInfo("Missing callable name! Contact administrator!");
                            tf_tensorboard.notifyHelpChanged("Missing executable name! Contact administrator!", "executer");
                            tf_tensorboard.handleAddNewNotification({
                                title: "Missing callable name! Contact administrator!",
                                icon: "close",
                                plugin: "Executer"
                            });
                            alert("Missing callable name! Contact administrator!");
                        } else {
                            if (this.config["default"]["image_folders"]) {
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
                tf_tensorboard.notifyHelpChanged("Couldn't find or create sprite/metadata files! Contact administrator!", "executer");
                tf_tensorboard.handleAddNewNotification({
                    title: "Couldn't find or create sprite/metadata files! Contact administrator!",
                    icon: "close",
                    plugin: "Executer"
                });
                alert("Couldn't find or create sprite/metadata files! Contact administrator!");
            }
            xhr.onload = () => {
                let ret = this;
                var callHandler = window.setInterval(function () {
                    let im = (document.querySelector("vz-image") as any),
                        se = (document.querySelector("vz-selected") as any),
                        vd = (document.querySelector("vz-video") as any);
                    if ((im) || (se) || (vd)) {
                        //TODO: migrate this to the individual plugins. Each plugin should read whatever they need
                        //from the configuration file.
                        clearInterval(callHandler);

                        im.setPath(ret.config["default"]["image_folders"][0]);
                        se.setPath(ret.config["default"]["image_folders"][0]);
                        //vd.setPath(ret.config["default"]["video_folder"]);

                        let count = 0;
                        for (let i in ret.config["trainings"]) {
                            count++;
                        }

                        let namesd = [];
                        for (let j = 0; j < count; j++) {
                            if (ret.config["trainings"][j]["type"] == "train") {
                                namesd.push(ret.config["trainings"][j]["name"]);
                            }
                        }

                        ret.names = namesd;
                    }
                }, 2);

                //Pass the embedding names so the Projector can check their validity.
                let train_folders = []
                for (let i in this.config['trainings']) {
                    train_folders.push(this.config['trainings'][i]['embedding_folder']);
                }

                this.projector.sendNames(this.config["default"]["embedding_folder"], train_folders);

                //Wait for the projector to find them and set the correct default embedding to view.
                var callHandler2 = window.setInterval(function () {
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
            this._selectedRunTest = this.projector.selectedRun;
        }

        runTraining(): void {

            if (this.selecteditem == -1) {
                tf_tensorboard.handleAddNewNotification({
                    title: "Select algorithm first!",
                    icon: "close",
                    plugin: "Executer"
                });
                alert('Select algorithm first!');
                return;
            }

            let b = document.getElementById("ExecButton") as HTMLButtonElement;
            b.disabled = true;
            //b.disabled = false;
            //this.runnable = 'true';
            //this.runnable = '';
            //save the algorithm as a cookie
            window.localStorage.setItem(tf_tensorboard.sublogdir + '-algorithm', this.selecteditem.toString());

            //this.projectorMain.setInfo("Execution has started!");
            tf_tensorboard.notifyHelpChanged("Execution has started!", "executer");
            tf_tensorboard.handleAddNewNotification({
                title: "Execution has started!",
                icon: "hourglass-empty",
                plugin: "Executer"
            }); // waiting

            let dsToRun = this.selectedDatabase;
            const xhr = new XMLHttpRequest();
            console.log(this.selectedDatabase);
            console.log(this._selectedRunTest);
            if (!this._selectedRunTest) {
                //this.projectorMain.setInfo("Execution has failed! No algorithm has been selected!")
                //this.runnable = '';
                b.disabled = false;
                tf_tensorboard.notifyHelpChanged("Execution has failed! No algorithm has been selected!", "executer");
                tf_tensorboard.handleAddNewNotification({
                    title: "Execution has failed! No algorithm has been selected!",
                    icon: "close",
                    plugin: "Executer"
                }); // waiting
                alert(`Execution has failed! No algorithm has been selected!`);
                return;
            }
            console.log(tf_tensorboard.boardPath);
            console.log(this.config);
            console.log("----------");
            console.log(this.projector.selectedRun);
            +xhr.open('GET', `${this.routePrefix}/execute?boardPath=${tf_tensorboard.boardPath}&selectedRun=${this.projector.selectedRun.split('/')[1]}&imagePath=${this.config["default"]["image_folders"][dsToRun]}&num=${this.selecteditem}&subfolder=${tf_tensorboard.sublogdir}`);
            xhr.onerror = (err) => {
                //this.projectorMain.setInfo("Execution has failed! ERROR: Network error. Please contact administrator.");
                //this.runnable = '';
                b.disabled = false;
                tf_tensorboard.notifyHelpChanged("Execution has failed! ERROR: Network error. Please contact administrator.", "executer");
            }
            xhr.onerror = (err) => {
                //this.projectorMain.setInfo("Execution has failed! ERROR: Network error. Please contact administrator.");
                //this.runnable = '';
                b.disabled = false;
                tf_tensorboard.notifyHelpChanged("Execution has failed! ERROR: Network error. Please contact administrator.", "executer");
                tf_tensorboard.handleAddNewNotification({
                    title: "Execution has failed! ERROR: Network error. Please contact administrator.",
                    icon: "close",
                    plugin: "Executer"
                }); // waiting
            };
            xhr.onload = () => {
                b.disabled = false;
                //this.runnable = '';
                if (xhr.status == 500) {
                    //this.runnable = 'false';
                    let error_msg = xhr.responseText;
                    //this.projectorMain.setInfo(`Execution has failed! ${error_msg}`);
                    tf_tensorboard.notifyHelpChanged(`Execution has failed! ${error_msg}`, "executer");
                    tf_tensorboard.handleAddNewNotification({
                        title: `Execution has failed! ${error_msg}`,
                        icon: "close",
                        plugin: "Executer"
                    }); // waiting
                    alert(`Execution has failed! ${error_msg}`);
                } else {
                    //this.runnable = 'false';
                    tf_tensorboard.handleAddNewNotification({
                        title: "Execution has finished! Reload the website to see the results.",
                        icon: "check",
                        plugin: "Executer"
                    }); // waiting
                    //this.projectorMain.setInfo("Execution has finished! Reload the website to see the results.");
                    tf_tensorboard.notifyHelpChanged("Execution has finished! Reload the website to see the results.", "executer");
                    if (confirm("Execution has finished! Would you like to reload the page now to see the results?")) {
                        window.location.reload();
                    }
                }
                //b.disabled = false;
            }
            xhr.send();
        }


        showDetails(): void {
            const xhr = new XMLHttpRequest();
            //console.log(this.projector.selectedRun.split('/')[1]);
            //console.log(this.config["default"]["image_folders"][this.selectedDatabase]);


            if (!this._selectedRunTest) {
                //this.projectorMain.setInfo("Please select an embedding!");
                tf_tensorboard.notifyHelpChanged("Please select an embedding!", "executer");
                tf_tensorboard.handleAddNewNotification({
                    title: "Please select an embedding!",
                    icon: "close",
                    plugin: "Executer"
                }); // waiting
                alert(`Please select an embedding!`);
                return;
            }
            xhr.open('GET', `${this.routePrefix}/details?num=${this.selecteditem}&subfolder=${tf_tensorboard.sublogdir}`);
            xhr.onerror = (err) => {
                //this.projectorMain.setInfo("Execution has failed! ERROR: Network error. Please contact administrator.");
                tf_tensorboard.notifyHelpChanged("Execution has failed! ERROR: Network error. Please contact administrator.", "executer");
            }
            xhr.onload = () => {
                let data = JSON.parse(xhr.response);
                let training_name = data["name"];
                let training_algorithm = data["algorithm_path"] + "/" + data["file"] + ".py";
                let embedding_on = this.projector.selectedRun.split('/')[1];
                let embedding_to = data["embedding_folder"] + "__" + this.config["default"]["image_folders"][this.selectedDatabase];
                let db_on = this.config["default"]["image_folders"][this.selectedDatabase];
                let training_type = data["train"];
                let new_model = (data["train"] == "organized") ? "yes" : "no";

                let trainingname = document.getElementById("training-name") as HTMLDivElement;
                trainingname.innerHTML = "The name of the training algorithm: " + training_name;

                let embeddingon = document.getElementById("embedding-to") as HTMLDivElement;
                embeddingon.innerHTML = "Embedding that would be created: " + embedding_to;

                let trainingalgorithm = document.getElementById("algorithm-path") as HTMLDivElement;
                trainingalgorithm.innerHTML = "Path of the training algorithm: " + training_algorithm;

                let usedembedding = document.getElementById("used-embedding") as HTMLDivElement;
                usedembedding.innerHTML = "The embedding used to train on: " + embedding_on;

                let useddatabase = document.getElementById("used-database") as HTMLDivElement;
                useddatabase.innerHTML = "The database used to train on: " + db_on;

                let trainingtype = document.getElementById("training-type") as HTMLDivElement;
                trainingtype.innerHTML = "The type of the training algorithm is: " + training_type;

                let newmodel = document.getElementById("new-model") as HTMLDivElement;
                newmodel.innerHTML = "Will a new model be created: " + new_model;

                this.$.details.open();
            }
            xhr.send();
        }
    }


    document.registerElement(Executer.prototype.is, Executer);

}  // namespace tf_executer

