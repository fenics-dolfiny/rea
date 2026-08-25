#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <print>

#include "doctest.h"
#include "running_error.h"

// Write your tests
TEST_CASE_TEMPLATE("Addition", T, std::float16_t, std::float32_t,
                   std::float64_t) {
  running_error::running_error_t<T> a(static_cast<T>(0.1));
  running_error::running_error_t<T> b(static_cast<T>(0.2));

  SUBCASE("basic ops") {
    running_error::running_error_t<T> c = a + b;
    CHECK((c.val - 0.3) <= c.eps * c.val);
    CHECK(c.error <= c.eps);

    c = a * b;
    CHECK((c.val - 0.02) <= c.eps * c.val);
    CHECK(c.error <= c.eps);

    c = a / b;
    CHECK((c.val - 0.5) <= c.eps * c.val);
    CHECK(c.error <= c.eps);
  }
}