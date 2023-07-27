package(default_visibility = ["//tensorboard:internal"])

load("//tensorboard/defs:web.bzl", "tf_web_library")

licenses(["notice"])  # Apache 2.0

tf_web_library(
    name = "vz_modelmanager",
    srcs = [
        "vz-modelmanager-style.html",
        "vz-modelmanager-util.ts",
        "vz-modelmanager.ts",
        "vz-modelmanager.html",
        "vz-modelmanager-dashboard.html",
    ],
    path = "/vz-modelmanager",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorboard/components/tf_backend",
        "//tensorboard/components/tf_dashboard_common",
        "//tensorboard/components/tf_imports:polymer",
        "//tensorboard/components/tf_tensorboard:synchronization_event_context",
        "//tensorboard/components/tf_tensorboard:registry",
        "@org_polymer_iron_icon",
        "@org_polymer_paper_button",
        "@org_polymer_paper_checkbox",
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
