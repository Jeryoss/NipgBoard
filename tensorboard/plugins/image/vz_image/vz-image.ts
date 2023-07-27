/**
 * @namespace vz_image
 */

namespace vz_image {
    /**
     * Represents the Polymer element for the image component.
     */
    export let ImagePolymer = PolymerElement({
        is: 'vz-image',
        properties: {
            imagesource: String,
            routePrefix: String,
            changePicture: Function,
        },
    });

    /**
     * Represents the class responsible for the large display for the current hovered-on image.
     */
    export class image extends ImagePolymer {
        private _path: string;
        private _port: string;
        private _handler: any; // The interval handler that is considered active and to be terminated without result if a new handler is instantiated.
        private _img_buffer: string; // To encapsulate a request in the interval handler logic, we make sure to only consider a call valid if the most recent hover call was not on the same image.

        /**
         * Runs initially when the plugin becomes ready.
         */
        ready() {
            // In case the plugin isn't loaded from Dashboard, manually get the routePrefix.
            if (!this.routePrefix) {
                this.routePrefix = tf_backend.getRouter().pluginRoute('image', '');
            }

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
        }

        /**
         * Manually called from outside, changes the current visible on-hover picture after a 10 ms delay.
         * @param input - The input image URL.
         */
        changePicture(input: string) {
            let img_url =
                tf_tensorboard.sublogdir === " "
                    ? `http://${window.location.hostname}:${this._port}/${this._path}/${input}?${tf_tensorboard.username}&${tf_tensorboard.password}`
                    : `http://${window.location.hostname}:${this._port}/${tf_tensorboard.sublogdir}/${this._path}/${input}?${tf_tensorboard.username}&${tf_tensorboard.password}`;

            console.log(img_url);

            if (input === "") {
                // Display a stock placeholder for when hovering on the background and reset the buffer.
                this._img_buffer = "";
                this.imagesource = "https://i.ibb.co/CMx9Gkd/nohovering-board-3.png";

                clearInterval(this._handler);
            } else if (this._img_buffer !== input && !tf_tensorboard.disablehighres) {
                // Call is only valid if the most recent call wasn't identical.
                // TODO: should probably be migrated to the event caller level.
                this._img_buffer = input;

                // Interrupt the previous hover command as a new one has been given within the 10 ms timeout.
                clearInterval(this._handler);

                // The 10 ms timeout that sets the on-hover picture once finished.
                let counter = 0;
                const _ret = this;
                const callHandler = (this._handler = window.setInterval(function () {
                    counter += 1;
                    if (counter === 1) {
                        clearInterval(callHandler);
                        _ret._handler = null;
                        _ret.imagesource = img_url;
                    }
                }, 100)); // Timeout of 70 ms or lower is unstable and doesn't prevent server crash.
                // The lowest seemingly stable ms value is 85.
            }
        }

        /**
         * Changes the path within the web service where the images can be found.
         * @param path - The new path value.
         */
        setPath(path: string) {
            this._path = path;
        }
    }

    document.registerElement(image.prototype.is, image);
} // namespace vz_image
