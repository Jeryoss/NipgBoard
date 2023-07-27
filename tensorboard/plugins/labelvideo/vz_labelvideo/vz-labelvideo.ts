namespace vz_labelvideo {

    export let labelvideoPolymer = PolymerElement({
        is: 'vz-labelvideo',
        properties: {
            labelvideosource: String,
            routePrefix: String,
            changePicture: Function,
            addToCache: Function,
            syncVids: Function,
            time: Number,
            prevFrame: Function,
            nextFrame: Function
        }
    });
    
    /** The instantiated class of the plotter */
    export class labelvideo extends labelvideoPolymer {
        private _port: string;
        private _anno: boolean;
        private _start: number;
        private _sync: boolean;
        private _delete: boolean;
        private _overlays: Array<any>;
        private _labels: Array<any>;
        private _labelnames: Array<any>;
        private _videos: Array<any>;
        private _video_player: Array<any>;
        private _fps: number;
        private _res: number;
        private _ratios: Array<number>;
        private _vidf: string;
        private _ovtoggles: Array<any>;
        private _ovnames: Array<any>;
        private _vidnum: number;
    
        /** Runs initially when plugin becomes ready */
        ready() {
            this._anno = false;
            this._sync = false;
            this._delete = false;
            this._ovtoggles = [];
            this._ovnames = [];
            this._labels = [];
            this._videos = [];
            this._video_player = [];
            this._overlays = [];
            this._ratios = [];
            this._res = 1080; //default is 1080p
            this._vidnum = 4;

            tf_tensorboard.registerSelectionChangedListener(
                (selection) => this.selectionUpdated(selection), "labelvideo");

            //In case plugin isn't loaded from Dashboard, manually get the routePrefix.
            if (!(this.routePrefix)) {
                this.routePrefix = tf_backend.getRouter().pluginRoute('labelvideo','');
            }

            //Point to this in case of callback faults.
            let _ret = this;

            //Fetch the names of the available videos.
            console.log(tf_tensorboard.sublogdir);
            const videoReq = new XMLHttpRequest();
            videoReq.open('GET', `${this.routePrefix}/videos?subfolder=${tf_tensorboard.sublogdir}`);
            videoReq.withCredentials = true;
            videoReq.onerror = (err) => {};
            videoReq.onload = () => {
                //console.log(videoReq.responseURL);
                let data = JSON.parse(videoReq.response);
                _ret._videos = data["videos"];
                //put an update here maybe
                /*for(let video of _ret._videos) {
                    this.updateMetadata(video);
                }*/
                console.log(_ret._videos);
                _ret._fps = parseInt(data["fps"]);
                _ret._vidf = data["video_folder"];
                _ret._labelnames = data["labels"];
                _ret._res = parseInt(data["video_res"]);

                //Fill label selection dropdown with data from config file.
                let labelselec = document.querySelector("#LVLabelselec");
                for(let lname of _ret._labelnames) {
                    let child = document.createElement("option");
                    child.innerHTML = lname;
                    child.setAttribute("value", lname);
                    labelselec.appendChild(child);
                }

                //Fetch the webservice port number; and instantiate the controls and set event handlers.
                const portReq = new XMLHttpRequest();
                console.log(_ret.routePrefix);
                portReq.open('GET', `${_ret.routePrefix}/port`);
                portReq.withCredentials = true;
                portReq.onerror = (err) => {}
                portReq.onload = () => {
                    //Fetch the port.
                    _ret._port = portReq.responseText;

                    //Instantiate the four video players.
                    for (let i = 1; i <= this._vidnum; i++) {
                        let player = _ret._video_player[i] = document.querySelector(`#LVvid${i}`) as HTMLVideoElement;
                        let prev = document.querySelector(`#LVprev${i}`);         
                        let next = document.querySelector(`#LVnext${i}`);         
                        let vidselec = document.querySelector(`#LVvidselec${i}`); 
                        let ovselec = document.querySelector(`#LVovselec${i}`);

                        //Frame by frame navigation.
                        prev.addEventListener('click', () => {
                            _ret.prevFrame(i);
                        })
                        next.addEventListener('click', () => {
                            _ret.nextFrame(i);
                        })

                        //We instantiate the base level of overlay toggle booleans.
                        _ret._ovtoggles[i] = [];

                        //Video selector dropdown content.
                        //Default video for video player #N is Nth video if it exists.
                        let c = (_ret._videos.length < i ? 1 : i);      
                        for(let v in _ret._videos) {
                            let child = document.createElement("option");
                            child.innerHTML = _ret._videos[v];
                            if (parseInt(v) + 1 == c) {
                                child.setAttribute("selected", "selected");
                                player.src = "http://" + window.location.hostname + ":" + _ret._port + "/" + tf_tensorboard.sublogdir + "/" + _ret._vidf + "/" + _ret._videos[v]+"?"+tf_tensorboard.username+"&"+tf_tensorboard.password;
                                //player.load();

                                //Load the corresponding starting overlays for this player.
                                _ret.loadOverlaysFromServer(i, _ret._videos[v]);
                            }
                            child.setAttribute("value", v.toString());
                            vidselec.appendChild(child);

                            //This is also a great time to instantiate the overlay toggle booleans.
                            _ret._ovtoggles[i][_ret._videos[v]] = [];
                        }

                        //Load the labels of the video that initially plays in the first player.
                        if (i==1) {
                            _ret.loadLabelsFromServer(_ret._videos[0]);
                        }


                        //Video selection event handler.
                        vidselec.addEventListener("change", function() {
                            let video = this.options[this.selectedIndex].text;
                            player.src = "http://" + window.location.hostname + ":" + _ret._port + "/" + tf_tensorboard.sublogdir + "/" + _ret._vidf + "/" + video+"?"+tf_tensorboard.username+"&"+tf_tensorboard.password;
                            player.load();
                            console.log(player.src);


                            //First time we select the video, load its overlays.
                            if(!_ret._overlays[video]) {
                                _ret.loadOverlaysFromServer(i, video);
                            } else {
                                _ret.loadOverlays(i, video);
                            }

                            //For the video #1 only, first time we select the video, load its labels from the server. Otherwise, load labels from cache.
                            if (i==1) {
                                if (!_ret._labels[video]) {
                                    _ret.loadLabelsFromServer(video);
                                } else {
                                    _ret.loadLabels(video);
                                }
                            }
                        })

                        //Overlay toggle event handler.
                        ovselec.addEventListener("change", function() {
                            let currentvideo = (vidselec as any).options[(vidselec as any).selectedIndex].text;

                            //TODO: optimize this
                            if (this.selectedIndex > 0) {
                                _ret._ovtoggles[i][currentvideo][this.selectedIndex - 1] = !_ret._ovtoggles[i][currentvideo][this.selectedIndex - 1];
                                _ret.updateOverlay(i, player.currentTime, currentvideo)

                                this.options.selectedIndex = 0;
                            }
                        })

                        let rect = player.getBoundingClientRect();
                        let canvas = document.querySelector("#LVoverlay"+i.toString()) as any;

                        canvas.style.height = Math.min(rect.height,rect.width*9.0/16.0) + "px";
                        canvas.height = Math.min(rect.height,rect.width*9.0/16.0);
                        this._ratios[i] = canvas.height / this._res;
                        //canvas.width = rect.height*16.0/9.0
                        //canvas.style.width = (rect.height*16.0/9.0) + "px";
                        canvas.width = Math.min(rect.height*16.0/9.0,rect.width)
                        canvas.style.width = Math.min(rect.height*16.0/9.0,rect.width) + "px";
                        canvas.style.paddingLeft = (rect.width - canvas.width)/2 + "px";
                        canvas.style.paddingTop = (rect.height - canvas.height)/2 + "px";

                        player.addEventListener("timeupdate", function() {
                            if (i == 1) {
                                if (_ret._sync) {
                                    let vid2 = document.querySelector("#LVvid2") as any,
                                        vid3 = document.querySelector("#LVvid3") as any,
                                        vid4 = document.querySelector("#LVvid4") as any;
                                    vid2.currentTime = vid3.currentTime = vid4.currentTime = player.currentTime;
                                }
                            }
                            _ret.updateOverlay(i, player.currentTime, (vidselec as any).options[(vidselec as any).selectedIndex].text);
                        })
                    }

                    let annoB = document.querySelector("#LVAnno") as any,
                        syncB = document.querySelector("#LVSync") as any,
                        deleB = document.querySelector("#LVDele") as any,
                        mul1 = document.querySelector("#LVmul1") as any,
                        mul2 = document.querySelector("#LVmul2") as any,
                        mul3 = document.querySelector("#LVmul3") as any,
                        mul4 = document.querySelector("#LVmul4") as any;
                    //let labelvid = document.getElementById("vz-labelvideo");
                    let labelvid = document.getElementsByTagName("vz-labelvideo-dashboard")[0];
                    
                    syncB.addEventListener('click', () => {
                        _ret.syncVids();
                    })
                    annoB.addEventListener('click', () => {
                        _ret.addToCache();
                        this._anno = !this._anno;
                    })
                    deleB.addEventListener('click', () => {
                        _ret.deleteLabel();
                    })
                    
                    mul1.addEventListener('click', () => {
                        _ret.changeLayout(1);
                    })
                    mul2.addEventListener('click', () => {
                        _ret.changeLayout(2);
                    })
                    mul3.addEventListener('click', () => {
                        _ret.changeLayout(3);
                    })
                    mul4.addEventListener('click', () => {
                        _ret.changeLayout(4);
                    })
                    labelvid.addEventListener('sizechanged',() =>{
                        this.refreshSize();
                    })
                    //const prReq = new XMLHttpRequest();
                    //prReq.open('GET', `${this.routePrefix}/embed?subfolder=${tf_tensorboard.sublogdir}`);
                    //rReq.onload = () => {
                        //console.log("OK");
                    //}
                    //prReq.send();
                }
                portReq.send();
            }
            videoReq.send();
        }

        selectionUpdated(selection: number[]){
            let _ret = this;
            let fname = null;
            if(selection.length>0){
                let projector = document.querySelector("vz-projector") as any;
                if ((projector) && (projector.dataSet) && (projector.dataSet.points) && (projector.dataSet.points.length > 0)) {
                    let input = projector.dataSet.points[selection[0]].metadata;
                    if(!("Videoname" in input)){
                        return;
                    }
                    let video = input["Videoname"];
                    fname = input["Filename"]
                    console.log(video);
                    let frame = input["Frame"];
                    if(!("Frame" in input)){
                        return;
                    }
                    let old_video = ""
                    let video_prev = -1; 
                    for(let i=0;i<_ret._vidnum;i++){
                        let selec2 = document.querySelector(`#LVvidselec${i+1}`) as any;
                        if(video === selec2.options[selec2.selectedIndex].text){
                            video_prev = i+1;
                        }
                    }
                    for(let i=0;i<_ret._videos.length;i++){
                        let selec = document.querySelector(`#LVvidselec1`) as any;
                        if(video === selec.options[i].text){
                            old_video = selec.options[selec.selectedIndex].text;
                            selec.selectedIndex = i;
                        }
                    }
                    let player = _ret._video_player[1] = document.querySelector(`#LVvid1`) as HTMLVideoElement;
                    player.src = "http://" + window.location.hostname + ":" + _ret._port + "/" + tf_tensorboard.sublogdir + "/" + _ret._vidf + "/" + video+"?"+tf_tensorboard.username+"&"+tf_tensorboard.password;
                    console.log(player.src);
                    player.load();

                    if(!_ret._overlays[video]) {
                        _ret.loadOverlaysFromServer(1, video);
                    } else {
                        _ret.loadOverlays(1, video);
                    }
                    
                    if (!_ret._labels[video]) {
                        _ret.loadLabelsFromServer(video);
                    } else {
                        _ret.loadLabels(video);
                    }

                    //make video jump to frame
                    player.currentTime = frame/_ret._fps;
                    console.log(player.currentTime);

                    if(video_prev<=_ret._vidnum&&video_prev!=-1){
                        for(let i=0;i<_ret._videos.length;i++){
                            let selec = document.querySelector(`#LVvidselec${video_prev}`) as any;
                            if(old_video === selec.options[i].text){
                                selec.selectedIndex = i;
                            }
                        }
                        let player_other = _ret._video_player[1] = document.querySelector(`#LVvid${video_prev}`) as HTMLVideoElement;
                        player_other.src = "http://" + window.location.hostname + ":" + _ret._port + "/" + tf_tensorboard.sublogdir + "/" + _ret._vidf + "/" + old_video+"?"+tf_tensorboard.username+"&"+tf_tensorboard.password;
                        player_other.load();

                        if(!_ret._overlays[old_video]) {
                            _ret.loadOverlaysFromServer(video_prev, old_video);
                        } else {
                            _ret.loadOverlays(video_prev, video);
                        }
                    }

                }
            }
            console.log(fname)
            console.log("from labelvideo, selection changed to: "+selection);
            _ret.resetCanvas();
            console.log("video number: "+_ret._vidnum)
        }

        // OVERLAY RELATED FUNCTIONS

        /**
         * Loads the overlays for the given video and instantiates the boolean variables for the given videoplayer.
         * @param i The videoplayer of which we'll instantiate the toggle booleans.
         * @param video The video of whose overlays we'll load from the server.
         */
        loadOverlaysFromServer(i: number, video: string) {
            const osxhr = new XMLHttpRequest();
            osxhr.open('GET', `${this.routePrefix}/file?subfolder=${tf_tensorboard.sublogdir}&folder=${this._vidf}&fname=${video.split(".")[0]}_overlays.json`);
            osxhr.onerror = () => {}
            osxhr.onload = () => {
                if (osxhr.status != 404) {
                    this._overlays[video] = JSON.parse(osxhr.responseText);

                    //Stash all of the overlay names for this given video.
                    this._ovnames[video] = [];
                    for (let overlay of this._overlays[video]) {
                        this._ovnames[video].push(overlay["Name"]);
                    }

                    //Load the overlays for the video.
                    this.loadOverlays(i, video);
                }
            }
            osxhr.send();
        }

        /**
         * Instantiates the toggle variables if they're not already set, and changes the dropdown to reflect the correct overlays.
         * @param i The video player whose dropdown is to be changed.
         * @param video The video whose overlays are to be set.
         */
        loadOverlays(i: number, video: string) {
            //Reset the dropdown selector to include the overlays for the current video.
            let ovselec = document.querySelector(`#LVovselec${i}`);

            //Set all of the toggle booleans if they're not already set.
            if (!this._ovtoggles[i]) this._ovtoggles[i] = [];
            this._ovtoggles[i][video] = [];
            let num_of_overlays = this._ovnames[video].length;
            for (let j = 1; j <= num_of_overlays; j++) {
                this._ovtoggles[i][video].push(true);
            }

            //Delete all of the current options.
            while (ovselec.firstChild) {
                ovselec.removeChild(ovselec.lastChild);
            }

            let temp = document.createElement("option");
            temp.innerHTML = "Select an overlay from the list to turn it ON or OFF."
            temp.setAttribute("selected", "selected");
            ovselec.appendChild(temp);

            //Fill the overlay dropdown with the proper names.
            for (let oname of this._ovnames[video]) {
                let child = document.createElement("option");
                child.innerHTML = oname;
                child.setAttribute("value", oname);
                ovselec.appendChild(child);
            }

            //Draw the overlays on the start of the video.
            this.updateOverlay(i, 0, video);
        }

        /**
         * Updates and draws the overlays on the canvas.
         * @param i The videoplayer's ID number.
         * @param time The timestamp.
         * @param video The video of whose overlay we're drawing.
         */
        updateOverlay(i: number, time: number, video: string) {
            let colors = ["yellow", "cyan", "magenta", "brown", "orange", "green", "blue", "pink"];
            let frame = Math.floor(time*this._fps);

            let canvas = document.querySelector("#LVoverlay"+i.toString()) as HTMLCanvasElement
            let context = canvas.getContext('2d');
            context.clearRect(0, 0, canvas.width, canvas.height);

            for (let o_ind in this._overlays[video]) if ((this._ovtoggles[i][video][o_ind]) && (this._overlays[video])) {
                let order = this._overlays[video][o_ind]["Edges"];
                let points = this._overlays[video][o_ind]["Points"][frame];
    
                if (points&&points.length>0) {
                    for (let p of points) {
                        context.beginPath();
                        context.fillStyle = colors[parseInt(o_ind) % 8];
                        context.arc(Math.floor(parseInt(p[0]) * this._ratios[i]),
                                    Math.floor(parseInt(p[1]) * this._ratios[i]), 3 , 0, 2*Math.PI);
                        context.fill();
                    }
                    for (let o of order) {
                        let i1 = o[0],
                            i2 = o[1];
                        let x1 = Math.floor(parseInt(points[i1][0]) * this._ratios[i]),
                            y1 = Math.floor(parseInt(points[i1][1]) * this._ratios[i]),
                            x2 = Math.floor(parseInt(points[i2][0]) * this._ratios[i]),
                            y2 = Math.floor(parseInt(points[i2][1]) * this._ratios[i]);

                        context.beginPath();
                        context.moveTo(x1, y1);
                        context.lineTo(x2, y2);
                        context.lineWidth = 2;
                        context.strokeStyle = colors[parseInt(o_ind) % 8];
                        context.stroke();
                    }
                }
            }
        }

        // LABEL RELATED FUNCTIONS

        /**
         * Loads the video's label file from the server.
         * @param video The video whose labels we're loading.
         */
        loadLabelsFromServer(video: string) {
            const lsxhr = new XMLHttpRequest();
            lsxhr.open('GET', `${this.routePrefix}/file?subfolder=${tf_tensorboard.sublogdir}&folder=${this._vidf}&fname=${video.split(".")[0]}_labels.json`);
            lsxhr.onload = () => {
                if (lsxhr.status != 404) {
                    this._labels[video] = (JSON.parse(lsxhr.responseText)).sort(function(a: any, b: any) {
                        return (parseFloat(a["from"]) < parseFloat(b["from"]) ? -1 : 1)
                    });
                    this.loadLabels(video);
                } else {
                    let lab = document.querySelector("#labels");
                    while (lab.firstChild) {
                        lab.removeChild(lab.lastChild);
                    }
                }
            }
            lsxhr.send();
        }

        /**
         * Changes the list of displayed label blocks on the screen using cached data.
         * @param video The video whose labels we need to display.
         */
        loadLabels(video: string) {
            let lab = document.querySelector("#labels");
            let sel = document.querySelector("#LVvidselec1") as any;
            
            //Delete the current displayed labels from the screen.
            while (lab.firstChild) {
                lab.removeChild(lab.lastChild);
            }
            console.log(this._labels[video]);
            //Add the cached labels onto the screen.
            for(let label of this._labels[video]) {
                this.addLabel(video, label["label"], label["id"], label["from"], label["to"]);
            }
        }

        /**
         * This function adds the corresponding label into the cache and makes it appear on screen.
         * @param video The video whose labels we are adding.
         * @param label The name of the label type.
         * @param id The id of the label type, used to quickly fetch the label color.
         * @param tsfrom The starting timestamp of the label window.
         * @param tsto The ending timestamp of the label window.
         */
        addLabel(video: string, label: string, id: number, tsfrom: number, tsto: number) {
            let colors = ["yellow", "cyan", "magenta", "brown", "orange", "green", "blue", "pink"];
            let _ret = this;

            let lab = document.querySelector("#labels");
            let vid = document.querySelector("#LVvid1") as any;

            let color = colors[id % 8];
            let anno = document.createElement("div");
            anno.setAttribute("class", "anno");
            anno.setAttribute("style", "width: 100px; float: left; background-color: "+color+"; padding-left: 20px; border-color: black; border-style: solid; border-width: 1px; border-radius: 10px;");
            anno.addEventListener("click", function(this, e) {
                if (_ret._delete) {
                    _ret.delFromCache(video, label, id, tsfrom, tsto);
                    this.remove();
                } else {
                    vid.currentTime = tsfrom;
                }
            });
            anno.innerHTML = parseFloat(tsfrom.toString()).toPrecision(3).toString() + "-" + parseFloat(tsto.toString()).toPrecision(3).toString() + "   " + label;
            lab.appendChild(anno);
        }

        /**
         * When the label annotating button is pressed this is invoked.
         */
        addToCache() {
            let vid = document.querySelector("#LVvid1") as any;
            let sel = document.querySelector("#LVvidselec1") as any;
            let cur = sel.options[sel.selectedIndex].text;

            if (!this._anno) {
                //If it is toggled on, simply record the timestamp.
                this._start = vid.currentTime;
            } else {
                //If it is toggled off after toggled on, save the label.
                let dropdown = document.querySelector("#LVLabelselec") as any,
                    label = dropdown.options[dropdown.selectedIndex].text,
                    id = dropdown.selectedIndex,
                    tsfrom = this._start,
                    tsto = vid.currentTime;

                //Pop up the label on the display screen.
                this.addLabel(cur, label, id, tsfrom, tsto);

                //Add the label into the cache.
                this._labels[cur].push({"label": label, "id": id, "from": tsfrom.toString(), "to": tsto.toString()});

                //Immediately save the changes to the server as well.
                this.saveLabels(cur);
                //this.updateMetadata(cur);

            }
        }

        /**
         * Deletes a given label from the cached storage, and immediately updates it on the server as well.
         * @param video The video whose label we're deleting.
         * @param label The label's label type.
         * @param id The label's label type ID.
         * @param tsfrom The label's starting timestamp.
         * @param tsto The label's ending timestamp.
         */

        delFromCache(video: string, label: string, id: number, tsfrom: number, tsto: number) {
            this._labels[video] = this._labels[video].filter(
                l => !( (l["label"] === label) && (l["id"] == id) && (l["from"] == tsfrom) && (l["to"] == tsto) )
            );
            this.saveLabels(video);
            //this.updateMetadata(video);
        }

        /**
         * Saves the cached label list onto the server.
         * @param video The video whose labels we're saving.
         */
        saveLabels(video: string) {
            const lxhr = new XMLHttpRequest();
            lxhr.open('POST', `${this.routePrefix}/save?subfolder=${tf_tensorboard.sublogdir}&folder=${this._vidf}&fname=${video.split(".")[0]}_labels.json`);
            lxhr.onload = () => {
                //Label finished saving onto the server.
            }
            lxhr.send(JSON.stringify(this._labels[video]));
        }

        //OTHER FUNCTIONS

        /**
         * Toggles the sync switch which synchronizes all the active videos.
         */
        syncVids() {
            let button = document.querySelector("#syncVid");
            this._sync = !this._sync;
        }

        /**
         * Toggles the delete switch which changes the function of clicking the label boxes.
         * When off, clicking a label box will navigate to the video timestamp.
         * When on, clicking a label box deletes it.
         */
        deleteLabel() {
            let button = document.querySelector("#deleteButton");
            this._delete = !this._delete;
        }

        /**
         * Move back one frame in a video player.
         * @param i Video player's ID.
         */
        prevFrame(i: number) {
            let vid = document.querySelector("#LVvid"+i) as any;
            let current = vid.currentTime;
            let next = current - 1.0/this._fps;
            vid.currentTime = next;
        }

        /**
         * Move forward one frame in a video player.
         * @param i Video player's ID.
         */
        nextFrame(i: number) {
            let vid = document.querySelector("#LVvid"+i) as any;
            let current = vid.currentTime;
            let next = current + 1.0/this._fps;
            vid.currentTime = next;
        }

        /**
         * Change the number of videos, and thus, the entire layout.
         * @param i The new number of videos: can be 1, 2, 3 or 4.
         */
        changeLayout(i: number) {
            switch(i){
                case 1: {
                    this._vidnum = 1;
                    let vidc = document.querySelector("#vidcontainer") as HTMLElement;
                    vidc.style.height = "80%";
                    vidc.style.width = "100%";
                    vidc.style.display = "block";

                    let labl = document.querySelector("#labels") as HTMLElement;
                    labl.style.height = "15%";

                    let vid1 = document.querySelector("#vid1") as HTMLElement;
                    vid1.style.height = "100%";
                    vid1.style.width = "100%";
                    vid1.style.visibility = "visible";

                    let vid2 = document.querySelector("#vid2") as HTMLElement;
                    vid2.style.height = "0%";
                    vid2.style.width = "0%";
                    vid2.style.visibility = "hidden";

                    let vid3 = document.querySelector("#vid3") as HTMLElement;
                    vid3.style.height = "0%";
                    vid3.style.width = "0%";
                    vid3.style.visibility = "hidden";

                    let vid4 = document.querySelector("#vid4") as HTMLElement;
                    vid4.style.height = "0%";
                    vid4.style.width = "0%";
                    vid4.style.visibility = "hidden";
                } break;
                case 2: {
                    this._vidnum = 2;
                    let vidc = document.querySelector("#vidcontainer") as HTMLElement;
                    vidc.style.height = "80%";
                    vidc.style.width = "100%";
                    vidc.style.display = "block";

                    let labl = document.querySelector("#labels") as HTMLElement;
                    labl.style.height = "15%";

                    let vid1 = document.querySelector("#vid1") as HTMLElement;
                    vid1.style.height = "100%";
                    vid1.style.width = "50%";
                    vid1.style.visibility = "visible";

                    let vid2 = document.querySelector("#vid2") as HTMLElement;
                    vid2.style.height = "100%";
                    vid2.style.width = "50%";
                    vid2.style.visibility = "visible";

                    let vid3 = document.querySelector("#vid3") as HTMLElement;
                    vid3.style.height = "0%";
                    vid3.style.width = "0%";
                    vid3.style.visibility = "hidden";

                    let vid4 = document.querySelector("#vid4") as HTMLElement;
                    vid4.style.height = "0%";
                    vid4.style.width = "0%";
                    vid4.style.visibility = "hidden";
                } break;
                case 3: {
                    this._vidnum = 3;
                    let vidc = document.querySelector("#vidcontainer") as HTMLElement;
                    vidc.style.height = "80%";
                    vidc.style.width = "100%";
                    vidc.style.display = "block";

                    let labl = document.querySelector("#labels") as HTMLElement;
                    labl.style.height = "15%";

                    let vid1 = document.querySelector("#vid1") as HTMLElement;
                    vid1.style.height = "50%";
                    vid1.style.width = "100%";
                    vid1.style.visibility = "visible";

                    let vid2 = document.querySelector("#vid2") as HTMLElement;
                    vid2.style.height = "50%";
                    vid2.style.width = "50%";
                    vid2.style.visibility = "visible";

                    let vid3 = document.querySelector("#vid3") as HTMLElement;
                    vid3.style.height = "50%";
                    vid3.style.width = "50%";
                    vid3.style.visibility = "visible";

                    let vid4 = document.querySelector("#vid4") as HTMLElement;
                    vid4.style.height = "0%";
                    vid4.style.width = "0%";
                    vid4.style.visibility = "hidden";
                } break;
                case 4: {
                    this._vidnum = 4;
                    let vidc = document.querySelector("#vidcontainer") as HTMLElement;
                    vidc.style.height = "80%";
                    vidc.style.width = "100%";
                    vidc.style.display = "block";

                    let labl = document.querySelector("#labels") as HTMLElement;
                    labl.style.height = "15%";

                    let vid1 = document.querySelector("#vid1") as HTMLElement;
                    vid1.style.height = "50%";
                    vid1.style.width = "50%";
                    vid1.style.visibility = "visible";

                    let vid2 = document.querySelector("#vid2") as HTMLElement;
                    vid2.style.height = "50%";
                    vid2.style.width = "50%";
                    vid2.style.visibility = "visible";

                    let vid3 = document.querySelector("#vid3") as HTMLElement;
                    vid3.style.height = "50%";
                    vid3.style.width = "50%";
                    vid3.style.visibility = "visible";

                    let vid4 = document.querySelector("#vid4") as HTMLElement;
                    vid4.style.height = "50%";
                    vid4.style.width = "50%";
                    vid4.style.visibility = "visible";
                } break;
            }
            this.resetCanvas();
        }

        /**
         * Resets each canvas to fit the corresponding new video sizes.
         */
        resetCanvas() {
            for (let i=1; i <= this._vidnum; i++) {
                console.log("Refresh video: "+i);
                let player = this._video_player[i] = document.querySelector(`#LVvid${i}`) as HTMLVideoElement;
                let rect = player.getBoundingClientRect();
                let canvas = document.querySelector("#LVoverlay"+i.toString()) as any;
                let timestamp = player.currentTime;
                let sel = document.querySelector("#LVvidselec"+i.toString()) as any;
                let cur = sel.options[sel.selectedIndex].text;

                canvas.style.height = Math.min(rect.height,rect.width*9.0/16.0) + "px";
                canvas.height = Math.min(rect.height,rect.width*9.0/16.0);
                this._ratios[i] = canvas.height / this._res;
                //canvas.width = rect.height*16.0/9.0
                //canvas.style.width = (rect.height*16.0/9.0) + "px";
                canvas.width = Math.min(rect.height*16.0/9.0,rect.width)
                canvas.style.width = Math.min(rect.height*16.0/9.0,rect.width) + "px";
                canvas.style.paddingLeft = (rect.width - canvas.width)/2 + "px";
                canvas.style.paddingTop = (rect.height - canvas.height)/2 + "px";

                this.updateOverlay(i, timestamp, cur);
            }
        }

        refreshSize(){
            console.log("Reset overlays.");
            this.resetCanvas();
        }
    };
    
    document.registerElement(labelvideo.prototype.is, labelvideo);

    } //namespace