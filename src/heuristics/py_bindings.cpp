#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "gcp_solver.h"

namespace py = pybind11;

PYBIND11_MODULE(gcp_solver, m){
	py::class_<GCP_Solver>(m, "GCP_Solver")
	    .def(py::init<const std::vector<std::vector<int>>&, int, std::string>())
	    .def("solve", &GCP_Solver::solve);

}
