package(default_visibility = ["//tensorboard:internal"])

load("//tensorboard/defs:web.bzl", "tf_web_library")

licenses(["notice"])

tf_web_library(
    name = "vz_multidash",
    srcs = [
        "vz-multidash-style.html",
        "vz-multidash.html",
        "vz-multidash-dashboard.html",
    ],
    path = "/vz-multidash",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorboard/components/tf_backend",
        "//tensorboard/components/tf_dashboard_common",
        "//tensorboard/components/tf_imports:polymer",
        "//tensorboard/components/tf_tensorboard:synchronization_event_context",
        "//tensorboard/components/tf_tensorboard:registry",
        "//tensorboard/plugins/executer/vz_executer",
        "//tensorboard/plugins/projector/vz_projector",
        "//tensorboard/plugins/image/vz_image",
        "//tensorboard/plugins/selected/vz_selected",
        "@org_polymer_iron_icon",
        "@org_polymer_paper_button",
        "@org_polymer_paper_checkbox",
        "@org_polymer_paper_icon_button",
        "@org_polymer_paper_input",
        "@org_polymer_paper_item",
        "@org_polymer_paper_styles",
        "@org_polymer_paper_card",
    ]
)
