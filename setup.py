import setuptools

with open("README.md", "r") as fh:
  long_description = fh.read()

setuptools.setup(
  name="pyqc",
  version="0.1.0",
  author="hongshun",
  author_email="shunzGIM@163.com",
  description="simulator for quantum circuit",
  long_description=long_description,
  long_description_content_type="text/markdown",
  url="https://github.com/shunzgim/PyQC",
  packages=setuptools.find_packages(),
  install_requires=['numpy', 'matplotlib', 'scipy', 'opt_einsum','psutil'],
  package_data = {'pyqc.backends.simulator.libs':['*.so']},
  classifiers=[
  "Programming Language :: Python :: 3",
  "License :: OSI Approved :: MIT License",
  "Operating System :: Linux",
  ],
)
