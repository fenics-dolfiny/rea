#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <print>

#include "doctest.h"
#include "running_error.h"

// Write your tests
TEST_CASE("Addition") {
  running_error::running_error_t<double> a(0.1, 0.0);
  running_error::running_error_t<double> b(0.2, 0.0);

  SUBCASE("Addition") {
    running_error::running_error_t<double> c = a + b;
    CHECK(c.error <= c.eps);
  }
}