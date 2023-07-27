package(default_visibility = ["//tensorboard:internal"])

licenses(["notice"])
exports_files(["LICENSE"])

py_library(
    name = "graphcut_plugin",
    srcs = ["graphcut_plugin.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorboard:expect_numpy_installed",
        "//tensorboard:expect_tensorflow_installed",
        "//tensorboard/backend:http_util",
        "//tensorboard/plugins:base_plugin",
        "@org_pocoo_werkzeug",
    ]
)