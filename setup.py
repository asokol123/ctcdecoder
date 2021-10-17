from setuptools import setup
from setuptools_rust import RustExtension


setup(
    name="setuptools-rust-starter",
    version="0.1.0",
    packages=["ctcdecoder"],
    install_requires=["numpy"],
    rust_extensions=[
        RustExtension(
            "ctcdecoder.ctcdecoder",
            "Cargo.toml",
            debug=False,
        ),
    ],
    include_package_data=True,
    zip_safe=False,
)
