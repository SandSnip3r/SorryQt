#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <iostream>

using namespace std;
namespace py = pybind11;

int main() {
  py::scoped_interpreter guard{};

  size_t arraySize = 5;
  py::array_t<float> numpyArray(arraySize);
  auto uncheckedBuffer = numpyArray.unchecked<1>();
  cout << "Data: [";
  for (size_t i=0; i<uncheckedBuffer.size(); ++i) {
    cout << uncheckedBuffer(i) << ", ";
  }
  cout << "]" << endl;
  // py::object res = numpyArray.attr("nanargmax")();
  // nanargmax does not exist as a member function of a np array, instead call the global numpy nanargmax on our array
  py::object res = py::module::import("numpy").attr("nanargmax")(numpyArray);
  cout << "Argmax result: " << res.cast<size_t>() << endl;
  return 0;
}