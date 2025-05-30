# Description:
# TensorBoard, a dashboard for investigating TensorFlow

package(default_visibility = [":internal"])

load("//tensorboard/defs:web.bzl", "tf_web_library")
load("//tensorboard/defs:zipper.bzl", "tensorboard_zip_file")

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])  # Needed for internal repo.

package_group(
    name = "internal",
    packages = ["//tensorboard/..."],
)

# The standard TensorBoard binary that serves the webapp.
py_binary(
    name = "tensorboard",
    srcs = ["main.py"],
    main = "main.py",
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":default",
        ":program",
        "//tensorboard:expect_tensorflow_installed",
        "//tensorboard/plugins:base_plugin",
        "//tensorboard/util:tb_logging",
    ],
)

# The public TensorBoard python library, bundled with the pip package and
# available via 'import tensorboard as tb' once installed.
py_library(
    name = "lib",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":lazy",
        ":notebook",
        ":program",
        "//tensorboard/summary",
        "//tensorboard/summary/writer",
    ],
)

py_test(
    name = "lib_test",
    size = "small",
    srcs = ["lib_test.py"],
    srcs_version = "PY2AND3",
    visibility = ["//tensorboard:internal"],
    tags = ["support_notf"],
    deps = [
        ":lib",
        "@org_pythonhosted_six",
    ],
)

py_library(
    name = "manager",
    srcs = ["manager.py"],
    srcs_version = "PY2AND3",
    visibility = ["//tensorboard:internal"],
    deps = [
        ":version",
        "//tensorboard/util:tb_logging",
        "@org_pythonhosted_six",
    ],
)

py_test(
    name = "manager_test",
    size = "small",
    srcs = ["manager_test.py"],
    srcs_version = "PY2AND3",
    visibility = ["//tensorboard:internal"],
    tags = ["support_notf"],
    deps = [
        ":manager",
        ":test",
        ":version",
        "//tensorboard/util:tb_logging",
        "@org_pythonhosted_mock",
        "@org_pythonhosted_six",
    ],
)

py_test(
    name = "manager_e2e_test",
    size = "large",  # spawns subprocesses, sleeps, makes requests to localhost
    timeout = "moderate",
    # On Python 2, this test fails about 0.5% of the time when run with
    # high parallelism; TensorBoard subprocess time out instead of
    # launching successfully.
    flaky = True,
    srcs = ["manager_e2e_test.py"],
    srcs_version = "PY2AND3",
    visibility = ["//tensorboard:internal"],
    deps = [
        ":manager",
        "//tensorboard:expect_tensorflow_installed",
        "@org_pythonhosted_mock",
        "@org_pythonhosted_six",
    ],
    data = [
        ":tensorboard",
    ],
)

py_library(
    name = "notebook",
    srcs = ["notebook.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":manager",
    ],
)

py_library(
    name = "program",
    srcs = ["program.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":manager",
        ":version",
        "//tensorboard:expect_absl_logging_installed",
        "//tensorboard/backend:application",
        "//tensorboard/backend/event_processing:event_file_inspector",
        "@org_pocoo_werkzeug",
        "@org_pythonhosted_six",
    ],
)

py_test(
    name = "program_test",
    size = "small",
    srcs = ["program_test.py"],
    srcs_version = "PY2AND3",
    tags = ["support_notf"],
    deps = [
        ":default",
        ":program",
        ":test",
        "//tensorboard/backend:application",
        "//tensorboard/plugins/core:core_plugin",
        "@org_pocoo_werkzeug",
    ],
)

py_library(
    name = "test",
    testonly = 1,
    srcs = ["test.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorboard:expect_absl_testing_absltest_installed",
        "//tensorboard/util:tb_logging",
        "@org_pythonhosted_six",
    ],
)

py_library(
    name = "default",
    srcs = ["default.py"],
    data = ["webfiles.zip"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorboard:expect_pkg_resources_installed",
        "//tensorboard/plugins:base_plugin",
        "//tensorboard/plugins/executer:executer_plugin",
        "//tensorboard/plugins/core:core_plugin",
        "//tensorboard/plugins/projector:projector_plugin",
        "//tensorboard/plugins/image:image_plugin",
        "//tensorboard/plugins/selected:selected_plugin",
        "//tensorboard/plugins/labelvideo:labelvideo_plugin",
        "//tensorboard/plugins/multidash:multidash_plugin",
        "//tensorboard/plugins/modelmanager:modelmanager_plugin",
        "//tensorboard/plugins/graphcut:graphcut_plugin",
    ],
)

py_test(
    name = "default_test",
    size = "small",
    srcs = ["default_test.py"],
    srcs_version = "PY2AND3",
    tags = ["support_notf"],
    deps = [
        ":default",
        ":test",
        "//tensorboard:expect_pkg_resources_installed",
        "//tensorboard/plugins:base_plugin",
        "@org_pythonhosted_mock",
    ],
)

py_library(
    name = "version",
    srcs = ["version.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
)

tensorboard_zip_file(
    name = "webfiles",
    deps = [":assets"],
)

tf_web_library(
    name = "assets",
    srcs = [
        "//tensorboard/components:index.html",
    ],
    path = "/",
    deps = [
        "@com_google_fonts_roboto",
    ],
)

py_library(
    name = "expect_numpy_installed",
    # This is a dummy rule used as a numpy dependency in open-source.
    # We expect numpy to already be installed on the system, e.g. via
    # `pip install numpy`
    visibility = ["//visibility:public"],
)

py_library(
    name = "expect_futures_installed",
    # This is a dummy rule used as a futures dependency in open-source.
    # We expect futures to already be installed on the system, e.g. via
    # `pip install futures`
    visibility = ["//visibility:public"],
)

py_library(
    name = "expect_sqlite3_installed",
    # This is a dummy rule used as a sqlite3 dependency in open-source.
    # We expect sqlite3 to already be present, as it is part of the standard
    # library.
    visibility = ["//visibility:public"],
)

py_library(
    name = "expect_tensorflow_installed",
    # This is a dummy rule used as a TensorFlow dependency in open-source.
    # We expect TensorFlow to already be installed on the system, e.g. via
    # `pip install tensorflow`
    visibility = ["//visibility:public"],
)

py_library(
    name = "expect_absl_app_installed",
    # This is a dummy rule used as a absl-py dependency in open-source.
    # We expect absl-py to already be installed on the system, e.g. via
    # `pip install absl-py`
    visibility = ["//visibility:public"],
)

py_library(
    name = "expect_absl_flags_installed",
    # This is a dummy rule used as a absl-py dependency in open-source.
    # We expect absl-py to already be installed on the system, e.g. via
    # `pip install absl-py`
    visibility = ["//visibility:public"],
)

py_library(
    name = "expect_absl_logging_installed",
    # This is a dummy rule used as a absl-py dependency in open-source.
    # We expect absl-py to already be installed on the system, e.g. via
    # `pip install absl-py`
    visibility = ["//visibility:public"],
)

py_library(
    name = "expect_absl_testing_absltest_installed",
    # This is a dummy rule used as a absl-py dependency in open-source.
    # We expect absl-py to already be installed on the system, e.g. via
    # `pip install absl-py`
    visibility = ["//visibility:public"],
)

py_library(
    name = "expect_pkg_resources_installed",
    # This is a dummy rule used as a pkg-resources dependency in open-source.
    # We expect pkg-resources to already be installed on the system, e.g., via
    # `pip install setuptools`.
    visibility = ["//visibility:public"],
)

py_library(
    name = "tf_contrib_ffmpeg",
    # This is a dummy rule for the open source world, which indicates
    # that srcs dereference tf.contrib.ffmpeg.
    visibility = ["//visibility:public"],
)

filegroup(
    name = "tf_web_library_default_typings",
    srcs = [
        # Ordering probably matters.
        "@com_microsoft_typescript//:lib.es6.d.ts",
        "@io_angular_clutz//:src/resources/closure.lib.d.ts",
        "//tensorboard/defs:clutz.d.ts",
    ],
    visibility = ["//visibility:public"],
)

py_library(
    name = "data_compat",
    srcs = ["data_compat.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorboard/compat:tensorflow",
        "//tensorboard/compat/proto:protos_all_py_pb2",
        "//tensorboard/util:tensor_util",
    ],
)

py_test(
    name = "data_compat_test",
    size = "small",
    srcs = ["data_compat_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":data_compat",
        "//tensorboard:expect_numpy_installed",
        "//tensorboard:expect_tensorflow_installed",
        "//tensorboard/compat/proto:protos_all_py_pb2",
        "//tensorboard/util:tensor_util",
        "//tensorboard/util:test_util",
    ],
)

py_binary(
    name = "encode_png_benchmark",
    srcs = ["encode_png_benchmark.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorboard:expect_absl_logging_installed",
        "//tensorboard:expect_numpy_installed",
        "//tensorboard:expect_tensorflow_installed",
        "//tensorboard/util:encoder",
        "//tensorboard/util:tb_logging",
        "@org_pythonhosted_six",
    ],
)

py_library(
    name = "plugin_util",
    srcs = ["plugin_util.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        "@org_mozilla_bleach",
        "@org_pythonhosted_markdown",
        "@org_pythonhosted_six",
    ],
)

py_test(
    name = "plugin_util_test",
    size = "small",
    srcs = ["plugin_util_test.py"],
    srcs_version = "PY2AND3",
    tags = ["support_notf"],
    deps = [
        ":plugin_util",
        ":test",
        "@org_pythonhosted_six",
    ],
)

alias(
    name = "summary",
    actual = "//tensorboard/summary",
    visibility = ["//tensorboard:internal"],
)

py_library(
    name = "lazy",
    srcs = ["lazy.py"],
    srcs_version = "PY2AND3",
)

py_test(
    name = "lazy_test",
    srcs = ["lazy_test.py"],
    srcs_version = "PY2AND3",
    size = "small",
    tags = ["support_notf"],
    deps = [
        ":lazy",
        "@org_pythonhosted_six",
    ],
)
