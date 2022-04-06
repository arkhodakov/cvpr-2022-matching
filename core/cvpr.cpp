#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>


namespace py = pybind11;

py::array_t<float> align_icp(py::array_t<float> gt_normalized, py::array_t<float> tg_normalized) {
  return arg1 + arg2;
}

PYBIND11_MODULE(cvpr, handle) {
  handle.doc() = "I'm a docstring hehe";
  handle.def("align_icp", &align_icp);
}