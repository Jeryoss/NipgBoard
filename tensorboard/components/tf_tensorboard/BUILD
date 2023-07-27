package(default_visibility = ["//tensorboard:internal"])

load("//tensorboard/defs:web.bzl", "tf_web_library")
load("//tensorboard/defs:vulcanize.bzl", "tensorboard_html_binary")

licenses(["notice"])  # Apache 2.0

tf_web_library(
    name = "tf_tensorboard",
    srcs = [
        "autoReloadBehavior.ts",
        "style.html",
        "tf-tensorboard.html",
    ],
    path = "/tf-tensorboard",
    visibility = ["//visibility:public"],
    deps = [
        ":registry",
        "//tensorboard/components/tf_backend",
        "//tensorboard/components/tf_dashboard_common",
        "//tensorboard/components/tf_globals",
        "//tensorboard/components/tf_imports:lodash",
        "//tensorboard/components/tf_imports:polymer",
        "//tensorboard/components/tf_paginated_view",
        "//tensorboard/components/tf_storage",
        "@com_google_fonts_roboto",
        "@org_polymer_iron_icons",
        "@org_polymer_paper_button",
        "@org_polymer_paper_checkbox",
        "@org_polymer_paper_dialog",
        "@org_polymer_paper_header_panel",
        "@org_polymer_paper_icon_button",
        "@org_polymer_paper_tabs",
        "@org_polymer_paper_toolbar",
    ],
)

tf_web_library(
    name = "registry",
    srcs = [
        "registry.html",
        "registry.ts",
    ],
    path = "/tf-tensorboard",
    visibility = ["//visibility:public"],
)

tf_web_library(
    name = "synchronization_event_context",
    srcs = [
        "synchronizationEventContext.html",
        "synchronizationEventContext.ts",
    ],
    path = "/tf-tensorboard",
    visibility = ["//visibility:public"],
)

tf_web_library(
    name = "plugin_dialog",
    srcs = [
        "plugin-dialog.html",
    ],
    path = "/tf-tensorboard",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorboard/components/tf_imports:lodash",
        "//tensorboard/components/tf_imports:polymer",
    ],
)

tf_web_library(
    name = "default_plugins",
    srcs = ["default-plugins.html"],
    path = "/tf-tensorboard",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorboard/plugins/executer/vz_executer",
        "//tensorboard/plugins/projector/vz_projector",
        "//tensorboard/plugins/image/vz_image",
        "//tensorboard/plugins/selected/vz_selected",
        "//tensorboard/plugins/labelvideo/vz_labelvideo",
        "//tensorboard/plugins/graphcut/vz_graphcut",
        "//tensorboard/plugins/modelmanager/vz_modelmanager",
        "//tensorboard/plugins/multidash/vz_multidash",
    ],
)
