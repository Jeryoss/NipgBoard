package(default_visibility = ["//tensorboard:internal"])

load("//tensorboard/defs:web.bzl", "tf_web_library")

licenses(["notice"])  # Apache 2.0

tf_web_library(
    name = "vz_executer",
    srcs = [
        "vz-executer-style.html",
        "vz-executer-util.ts",
        "vz-executer.ts",
        "vz-executer.html",
        "vz-executer-dashboard.html",
    ],
    path = "/vz-executer",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorboard/components/tf_backend",
        "//tensorboard/components/tf_dashboard_common",
        "//tensorboard/components/tf_imports:polymer",
        "//tensorboard/components/tf_tensorboard:synchronization_event_context",
        "//tensorboard/components/tf_tensorboard:registry",
        "@org_polymer_iron_icon",
        "@org_polymer_paper_button",
        "@org_polymer_paper_dropdown_menu",
        "@org_polymer_paper_listbox",
        "@org_polymer_paper_icon_button",
        "@org_polymer_paper_input",
        "@org_polymer_paper_item",
        "@org_polymer_paper_menu",
        "@org_polymer_paper_styles",
	"@org_polymer_webcomponentsjs",
	"@org_polymer_neon_animation",
    ],
)
