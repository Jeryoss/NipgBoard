package(default_visibility = ["//tensorboard:internal"])

load("//tensorboard/defs:web.bzl", "tf_web_library")

licenses(["notice"])

tf_web_library(
    name = "vz_labelvideo",
    srcs = [
        "vz-labelvideo-dashboard.html",
        "vz-labelvideo-style.html",
        "vz-labelvideo-util.ts",
        "vz-labelvideo.html",
        "vz-labelvideo.ts",
    ],
    path = "/vz-labelvideo",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorboard/components/tf_backend",
        "//tensorboard/components/tf_dashboard_common",
        "//tensorboard/components/tf_imports:polymer",
        "//tensorboard/components/tf_tensorboard:synchronization_event_context",
        "//tensorboard/components/tf_tensorboard:registry",
        "@org_polymer_iron_icon",
        "@org_polymer_paper_button",
        "@org_polymer_paper_icon_button",
        "@org_polymer_paper_styles",
    ]
)