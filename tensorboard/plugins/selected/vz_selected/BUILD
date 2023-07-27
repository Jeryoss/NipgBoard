package(default_visibility = ["//tensorboard:internal"])

load("//tensorboard/defs:web.bzl", "tf_web_library")

licenses(["notice"])

tf_web_library(
    name = "vz_selected",
    srcs = [
        "vz-selected-dashboard.html",
        "vz-selected-util.ts",
        "vz-selected.html",
        "vz-selected.ts",
    ],
    path = "/vz-selected",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorboard/components/tf_backend",
        "//tensorboard/components/tf_dashboard_common",
        "//tensorboard/components/tf_imports:polymer",
        "//tensorboard/components/tf_tensorboard:synchronization_event_context",
        "//tensorboard/components/tf_tensorboard:registry",
    ]
)