from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='my_lib',
      ext_modules=[cpp_extension.CppExtension('my_lib', ['my_lib.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})


Extension(
   name='my_lib',
   sources=['my_lib.cpp'],
   include_dirs=cpp_extension.include_paths(),
   language='c++')

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &lltm_forward, "LLTM forward");
  m.def("backward", &lltm_backward, "LLTM backward");
}