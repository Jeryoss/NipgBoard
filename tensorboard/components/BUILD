package(default_visibility = ["//tensorboard:internal"])

load("//tensorboard/defs:web.bzl", "tf_web_library")
load("//tensorboard/defs:vulcanize.bzl", "tensorboard_html_binary")

licenses(["notice"])  # Apache 2.0

tf_web_library(
    name = "tensorboard",
    srcs = [
        "analytics.html",
        "tensorboard.html",
    ],
    path = "/",
    deps = [
        "//tensorboard/components/tf_imports:webcomponentsjs",
        "//tensorboard/components/tf_tensorboard",
        "//tensorboard/components/tf_tensorboard:default_plugins",
    ],
)

tensorboard_html_binary(
    name = "index",
    compile = True,
    input_path = "/tensorboard.html",
    output_path = "/index.html",
    deps = [":tensorboard"],
)
