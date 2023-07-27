# Developer Documentation for NIPGBoard for Tensorboard

This developer documentation details the folders, modules, logic and abstraction of the NIPGBoard code. For general instructions on building
or running any Tensorboard modification, we recommend [the original Development.md](https://github.com/tensorflow/tensorboard/blob/master/DEVELOPMENT.md).

## Network Architecture

NIPGBoard uses the following architecture:

* The entire application is hosted by a Werkzeug python server, by default on port 6006. This hosts the application and all of it's API services.
* In addition we host a web service hosting the LOGDIR, by default on port 6016. This enables easier image and video access instead of creating
  custom APIs.
* The front-end TypeScript code is instantiated via imported dashborads available with hash URLs, like #projector. Out of these right now we
  only use the projector plugin dashboard.

## Build files and importing

### Importing 3rd party packages

This is done through *.bzl* files, primarily in the *third_party* directory. It is one in the following fashion:

```
filegroup_external(
	name = "name" # The name of the package. In order to import a package later into the dependencies refer to it with @name.
	sha256_urls_extract = { "sha256 code": [ "download urls" ]} # Given correct sha256 and urls Bazel will download these upon building.
)
```

Please note that if you need a package to download for your TypeScript front-end code, import the package in *js.bzl* while also importing
the correspondin TypeScript typings in *typings.bzl*. Otherwise they won't build properly.

### Build files

For any piece of code, refer to the corresponding BUILD file found in the same level directory. A build package has many properties, most
notably the package name added in *name*, sources added in *srcs* and dependencies added in *deps*. These are necessary for the code to build.

### TypeScript namespacing instead of import directives

With the way any Tensorboard extension like NIPGBoard compiles the build files, you can't use the typical TypeScript import directives like
in something like Angular, as the Vulcanizer will compile everything into one long JS file. As such, instead, every TS file that belongs
to the same build package should be wrapper by simply a namespace referring to the same package name. And any external build package can
simply be used with the namespace prefix, even if a static analyzer / linter says that it isn't appropriate. 
Something along these lines should work perfectly:

```
plugin1_base.ts
namespace plugin1_sources {
	export imported = plugin2_sources.MyClass();
}
plugin1_extend.ts
namespace plugin1_sources {
	console.log(imported); #This will also work.
}
plugin2_base.ts
namespace plugin2_sources {
	export MyClass(): any;
}
```

## Backend

### Main server backend

The backend code for both the original Werkzeug server and our custom web service can be found in *tensorboard/program.py*.

The Werkzeug server needs to know which individual plugin backend modules (responsible for their personal respective API services) is going
to be imported. For this, refer to the code found in *tensorboard/default.py*

### Additonal backend sources

These can be find in the *components* folder, primarily *components/backend*.

### Plugin APIs

Every new plugin should have a *name_plugin.py* script that instantiates the plugin class from *base_plugin.py*. Refer to any of the other
plugins as an example. These codes also showcase how to name the API services, or to define them using the Werkzeug wrapper.

The front-end TypeScript knows its corresponding API route prefix via the *routePrefix* property that you can ask from the backend code in
the following fashion:

```
routePrefix = tf_backend.getRouter().pluginRoute('plugin_name', '');
```

## Front-end

### Main page and importing plugin dashboards

Plugins can be imported via their respective dashbaords, which by default in Tensorboard creates tabs in the header to switch between them,
in older deprecated versions of NIPGBoard, we could import them as dynamically movable windows instead. This is all done in
*tensorboard/components/tf_tensorboard*. Specifically:

* You can import the dashboards in *default-plugins.html*
* The main HTML file, like for example where the header or the dashboard drawing is defined, is *tf-tensorboard.html*
* Our specific synchrnoization file repsonsible for ID matching between plugins is found in *synchronizationEventContext.ts*

### Plugin structure

A new plugin should always follow the code base of the other plugins, in brief these important codefiles reside in *vz_plugin_name* and are:

* You can define the dashboard, with the necessary aspects such as loading the routePrefix, in *vz-plugin-dashboard.html*
* The plugin itself that resides in the dashboard is defined in *vz-plugin.html* and *vz-plugin.ts* files
* Be sure to mirror the nature of the other files, like utils, too.

The dashboard can technically be ommited, but in the case be sure to manually get the routePrefix at some point in time, like in the
ready() function when the plugin Polymer class instantiates.

## Specific NIPGBoard functionality locations

### Projector

This is a list of most relevant and import projector files and what their code entails:

* *projector_plugin.py* is the backend API source for the projector, with its important methods or services including aquiring the embedding and metadata/sprite files, and also is where we register any annotated pairs on the server.
* *data-provider.ts* is reponsible for calling the data services
* *projetorScatterPlotAdapter.ts* is the link between the renderred projection and the projector front-end inputs
* *scatterPlot.ts* and its different sister files are responsible for rendering the different entities on the projection, and handling controls that aren't from front-end buttons, but instead from moving or clicking on the canvas itself.
* *vz-projector-data-panel.ts* includes an original now deprecated window where the embedding or run selection is defined.
* *vz-projector.ts* is the main code for defining the front-end elements and maintaining top-level features such as defining changes in the dataset or selection.

### Executer

This is the custom plugin responsible for executing the training algorithm, or getting the configuration for the embeddings and algorithms:

* *executer.py* merely defines the nature of the class that parses the parameters and exeutes the algorithms.
* *executer_plugin.py* is the proper backend API that enables the services, and instantiates the executer.
* *vz_executer.ts* is the front-end code responsible for these features. It frequently communicates with the projector to swap information.

### Image and Selection

These are minor plugins responsible for seeing a bigger version of the sprite images on the projector. Right now only this uses the
web service as a failsafe in case sprite image splitting and rendering fails.
