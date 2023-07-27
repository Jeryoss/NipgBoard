/**
 * @namespace vz_selected
 */

namespace vz_selected {
    /**
     * Represents the Polymer element for the selected component.
     */
    export let SelectedPolymer = PolymerElement({
        is: 'vz-selected',
        properties: {
            selectedsource: String,
            routePrefix: String,
            changePicture: Function,
        },
    });

    /**
     * The instantiated class of the plotter.
     */
    export class selected extends SelectedPolymer {
        private _projector: any;
        private _port: string;
        private _path: string;

        /**
         * Runs initially when the plugin becomes ready.
         */
        ready() {
            // In case the plugin isn't loaded from Dashboard, manually get the routePrefix.
            if (!this.routePrefix) {
                this.routePrefix = tf_backend.getRouter().pluginRoute('selected', '');
            }
            this.selectedsource = "https://i.ibb.co/k3ZSjLK/noimage-board-3.png";

            // Register the callback responsible for catching changes within the selection.
            tf_tensorboard.registerSelectionChangedListener(
                (selection) => this.updateSelectedPoint(selection),
                "selected"
            );

            // Fetch the web service port number.
            const xhr = new XMLHttpRequest();
            xhr.open('GET', `${this.routePrefix}/port`);
            xhr.onerror = (err) => {
                // Handle error
            };
            xhr.onload = () => {
                this._port = xhr.responseText;
            };
            xhr.send();

            // Wait for the projector to be instantiated.
            let _ref = this;
            var callHandler = window.setInterval(function () {
                let projector = document.querySelector("vz-projector") as any;
                if (projector && projector.dataSet && projector.dataSet.points && projector.dataSet.points.length > 0) {
                    clearInterval(callHandler);
                    _ref._projector = projector;
                }
            });
        }

        /**
         * Called when the selection is changed, fetches the correct filename and changes the selection image.
         * @param selection - The selected point indices.
         */
        updateSelectedPoint(selection: number[]) {
            if (!tf_tensorboard.disablehighres) {
                if (selection.length > 0) {
                    let input = this._projector.dataSet.points[selection[0]].metadata["Filename"];
                    console.log(tf_tensorboard.sublogdir);
                    if (tf_tensorboard.sublogdir === " ") {
                        this.selectedsource = `http://${window.location.hostname}:${this._port}/${this._path}/${input}?${tf_tensorboard.username}&${tf_tensorboard.password}`;
                    } else {
                        this.selectedsource = `http://${window.location.hostname}:${this._port}/${tf_tensorboard.sublogdir}/${this._path}/${input}?${tf_tensorboard.username}&${tf_tensorboard.password}`;
                    }
                } else {
                    this.selectedsource = "https://i.ibb.co/k3ZSjLK/noimage-board-3.png";
                }
            }
        }

        /**
         * Changes the path within the web service where the images can be found.
         * @param path - The new path value.
         */
        setPath(path: string) {
            this._path = path;
            console.log(this._path);
        }
    }

    document.registerElement(selected.prototype.is, selected);
} // namespace vz_selected
