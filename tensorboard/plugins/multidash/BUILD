package(default_visibility = ["//tensorboard:internal"])

load("@org_tensorflow_tensorboard//tensorboard/defs:web.bzl", "tf_web_library")

licenses(["notice"])

py_library(
    name = "multidash_plugin",
    srcs = ["multidash_plugin.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorboard:expect_numpy_installed",
        "//tensorboard:expect_tensorflow_installed",
        "//tensorboard/backend:http_util",
        "//tensorboard/plugins:base_plugin",
        "//tensorboard/plugins:plugin_utils",
        "@org_pocoo_werkzeug"
    ]
)