# pyoxidizer.bzl - ready-to-build for Markterm

def make_exe():
    # Get default Python distribution
    dist = default_python_distribution()
    
    # Configure packaging policy
    policy = dist.make_python_packaging_policy()
    policy.bytecode_optimize_level_two = True  # optimize bytecode
    policy.extension_module_filter = "minimal"  # include only essential Python extensions
            
    # Configure the embedded Python interpreter
    python_config = dist.make_python_interpreter_config()
    python_config.run_filename = "test.py"  # <-- your app entry point
    
    # Create the executable
    exe = dist.to_python_executable(
        name="Markterm",
        packaging_policy=policy,
        config=python_config,
    )
    policy.resources_location = "in-memory"
    policy.resources_location_fallback =  "filesystem-relative:relative"
    policy.extension_module_filter = "no-libraries"
    
    # Embed dependencies
    exe.add_python_resources(
        exe.pip_install([
            "textual",
            "rich",
            "markdown-it-py",
            "textual-image",
        ])
    )

    # Embed your local utility.py and any modules
    exe.add_python_resources(
        exe.read_package_root(
            path=".",           # current folder
            packages=["utility"]  # include your local modules
        )
    )

    # Optional: include assets folder (CSS/images)


    return exe


def make_embedded_resources(exe):
    return exe.to_embedded_resources()


def make_install(exe):
    files = FileManifest()
    files.add_python_resource(".", exe)
    return files


def make_msi(exe):
    return exe.to_wix_msi_builder(
        "markterm",
        "Markterm",
        "1.0",
        "Smriti Khanal"
    )


def register_code_signers():
    if not VARS.get("ENABLE_CODE_SIGNING"):
        return
    # Optional: set up code signing if needed
    # signer.activate()


register_code_signers()

# Register targets
register_target("exe", make_exe)
register_target("resources", make_embedded_resources, depends=["exe"], default_build_script=True)
register_target("install", make_install, depends=["exe"], default=True)
register_target("msi_installer", make_msi, depends=["exe"])

resolve_targets()
