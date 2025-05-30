package(default_visibility = ["//tensorboard:internal"])

load("//tensorboard/defs:web.bzl", "tf_web_library")
load("//tensorboard/defs:vulcanize.bzl", "tensorboard_html_binary")

licenses(["notice"])  # Apache 2.0

tf_web_library(
    name = "vz_projector",
    srcs = [
        "analyticsLogger.ts",
        "bundle.html",
        "data.ts",
        "data-provider.ts",
        "data-provider-demo.ts",
        "data-provider-proto.ts",
        "data-provider-server.ts",
        "external.d.ts",
        "knn.ts",
        "label.ts",
        "logging.ts",
        "projectorEventContext.ts",
        "projectorScatterPlotAdapter.ts",
        "renderContext.ts",
        "scatterPlot.ts",
        "scatterPlotRectangleSelector.ts",
        "scatterPlotVisualizer.ts",
        "scatterPlotVisualizer3DLabels.ts",
        "scatterPlotVisualizerCanvasLabels.ts",
        "scatterPlotVisualizerPolylines.ts",
        "scatterPlotVisualizerSprites.ts",
        "styles.html",
        "umap.d.ts",
        "util.ts",
        "vector.ts",
        "vz-projector.html",
        "vz-projector.ts",
        "vz-projector-app.html",
        "vz-projector-bookmark-panel.html",
        "vz-projector-bookmark-panel.ts",
        "vz-projector-dashboard.html",
        "vz-projector-data-panel.html",
        "vz-projector-data-panel.ts",
        "vz-projector-input.html",
        "vz-projector-input.ts",
        "vz-projector-inspector-panel.html",
        "vz-projector-inspector-panel.ts",
        "vz-projector-legend.html",
        "vz-projector-legend.ts",
        "vz-projector-metadata-card.html",
        "vz-projector-metadata-card.ts",
        "vz-projector-projections-panel.html",
        "vz-projector-projections-panel.ts",
        "vz-projector-util.ts",
    ],
    path = "/vz-projector",
    visibility = ["//visibility:public"],
    deps = [
        ":bh_tsne",
        ":heap",
        ":sptree",
        "//tensorboard/components/tf_backend",
        "//tensorboard/components/tf_dashboard_common",
        "//tensorboard/components/tf_imports:d3",
        "//tensorboard/components/tf_imports:numericjs",
        "//tensorboard/components/tf_imports:polymer",
        "//tensorboard/components/tf_imports:threejs",
        "//tensorboard/components/tf_imports:umap-js",
        "//tensorboard/components/tf_imports:weblas",
        "//tensorboard/components/tf_tensorboard:registry",
        "//tensorboard/components/tf_tensorboard:synchronization_event_context",
        "@org_polymer_iron_collapse",
        "@org_polymer_iron_icons",
        "@org_polymer_paper_button",
        "@org_polymer_paper_checkbox",
        "@org_polymer_paper_dialog",
        "@org_polymer_paper_dialog_scrollable",
        "@org_polymer_paper_dropdown_menu",
        "@org_polymer_paper_icon_button",
        "@org_polymer_paper_input",
        "@org_polymer_paper_item",
        "@org_polymer_paper_listbox",
        "@org_polymer_paper_slider",
        "@org_polymer_paper_spinner",
        "@org_polymer_paper_styles",
        "@org_polymer_paper_toast",
        "@org_polymer_paper_toggle_button",
        "@org_polymer_paper_tooltip",
    ],
)

tf_web_library(
    name = "heap",
    srcs = ["heap.ts"],
    path = "/vz-projector",
)

tf_web_library(
    name = "sptree",
    srcs = ["sptree.ts"],
    path = "/vz-projector",
)

tf_web_library(
    name = "bh_tsne",
    srcs = ["bh_tsne.ts"],
    path = "/vz-projector",
    deps = [":sptree"],
)

################# Standalone development #################

tf_web_library(
    name = "standalone_lib",
    srcs = [
      "standalone.html",
      "standalone_projector_config.json",
    ],
    path = "/",
    deps = [
        "@org_polymer_iron_icons",
        "@org_polymer_paper_icon_button",
        "@org_polymer_paper_tooltip",
        "@org_polymer_webcomponentsjs",
        ":vz_projector",
    ],
)

tensorboard_html_binary(
    name = "devserver",
    input_path = "/standalone.html",
    output_path = "/index.html",
    deps = [":standalone_lib"],
)
