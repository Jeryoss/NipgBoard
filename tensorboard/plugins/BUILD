# Description:
# A plugin system for TensorBoard

package(default_visibility = ["//tensorboard:internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "base_plugin",
    srcs = ["base_plugin.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        "@org_pythonhosted_six",
    ],
)

py_library(
    name = "plugin_utils",
    srcs = ["plugin_utils.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        
    ],
)
