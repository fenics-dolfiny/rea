#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <algorithm>
#include <cmath>
#include <tuple>
#include <type_traits>

#include "doctest.h"
#include "running_error.h"

using running_error::re_exact_t;
using running_error::re_worst_t;

namespace {

// References in the widest type, so a test never measures its own reference.
using L = long double;

template <typename T>
constexpr L wide(T x) {
  return static_cast<L>(x);
}

// Via the library's epsilon: numeric_limits lies about _Float16 on clang.
template <typename T>
constexpr L eps_of() {
  return static_cast<L>(running_error::re_eps<T>);
}

// Check that the worst is an upper bound.
template <typename T>
void check_worst(const re_worst_t<T>& c, L exact) {
  CHECK(std::fabs(wide(c.val) - exact) <= wide(c.err));
}

// err is signed (computed - exact), so val - err recovers the exact result, up
// to second-order terms and the resolution of the reference type.
template <typename T>
void check_exact(const re_exact_t<T>& c, L exact) {
  using E = running_error::exact_type_t<T>;
  const L tol =
      10 * std::max(eps_of<T>() * eps_of<T>(), eps_of<E>()) * std::fabs(exact);
  CHECK(std::fabs(wide(c.val) - wide(c.err) - exact) <= tol);
}

// The same operation in both modes, against one reference: the bound has to
// cover it, the signed error has to recover it.
template <typename T>
void check(const re_worst_t<T>& w, const re_exact_t<T>& e, L exact) {
  check_worst(w, exact);
  check_exact(e, exact);
}

}  // namespace

// Write an operation once and check it in both modes..
#define CHECK_BOTH(expr, exact)            \
  {                                        \
    const auto& [a, b] = std::tie(aw, bw); \
    check_worst(expr, exact);              \
  }                                        \
  {                                        \
    const auto& [a, b] = std::tie(ae, be); \
    check_exact(expr, exact);              \
  }

TEST_CASE_TEMPLATE("Arithmetic operators", T, std::float16_t, std::float32_t,
                   std::float64_t) {
  // The mode defaults to WORST, and both modes expose the machine epsilon.
  static_assert(
      std::is_same_v<running_error::running_error_t<T>, re_worst_t<T>>);
  CHECK(wide(re_worst_t<T>::eps) == eps_of<T>());
  CHECK(wide(re_exact_t<T>::eps) == eps_of<T>());

  // A value already in T is its own exact representation.
  CHECK(wide(re_worst_t<T>(static_cast<T>(0.1)).err) == L(0));
  CHECK(wide(re_exact_t<T>(static_cast<T>(0.1)).err) == L(0));

  // The seeded error is relative to the double the object was built from, so
  // the reference is that same double widened, not the decimal literal.
  const double da = 0.1, db = 0.2;
  const L ra = wide(da), rb = wide(db);
  re_worst_t<T> aw(da), bw(db);
  re_exact_t<T> ae(da), be(db);

  CHECK_BOTH(a, ra);
  CHECK_BOTH(a + b, ra + rb);
  CHECK_BOTH(a - b, ra - rb);
  CHECK_BOTH(a * b, ra * rb);
  CHECK_BOTH(a / b, ra / rb);

  // Negation is exact: no rounding. The bound carries through unchanged, the
  // signed error flips along with the value.
  CHECK(wide((-aw).val) == -wide(aw.val));
  CHECK(wide((-aw).err) == wide(aw.err));
  CHECK(wide((-ae).val) == -wide(ae.val));
  CHECK(wide((-ae).err) == -wide(ae.err));

  SUBCASE("compound assignment") {
    re_worst_t<T> cw = aw;
    re_exact_t<T> ce = ae;
    cw += bw;
    ce += be;
    check(cw, ce, ra + rb);

    cw = aw;
    ce = ae;
    cw -= bw;
    ce -= be;
    check(cw, ce, ra - rb);

    cw = aw;
    ce = ae;
    cw *= bw;
    ce *= be;
    check(cw, ce, ra * rb);

    cw = aw;
    ce = ae;
    cw /= bw;
    ce /= be;
    check(cw, ce, ra / rb);

    // The scalar has no error of its own, so the bound only doubles, and the
    // signed error doubles exactly.
    cw = aw;
    ce = ae;
    cw *= static_cast<T>(2);
    ce *= static_cast<T>(2);
    check(cw, ce, 2 * ra);
    CHECK(wide(cw.err) >= 2 * wide(aw.err));
    CHECK(wide(ce.err) == 2 * wide(ae.err));
  }

  SUBCASE("scalar operands, both argument orders") {
    CHECK_BOTH(2.0 * a, 2 * ra);
    CHECK_BOTH(a * 2.0, 2 * ra);
    CHECK_BOTH(1.0 + a, 1 + ra);
    CHECK_BOTH(a + 1.0, ra + 1);
    CHECK_BOTH(1.0 - a, 1 - ra);
    CHECK_BOTH(a - 1.0, ra - 1);
    CHECK_BOTH(1.0 / a, 1 / ra);
    CHECK_BOTH(a / 2.0, ra / 2);
    CHECK_BOTH(a * 3, 3 * ra);  // integral scalar
  }
}

// Only EXACT mode has error-free transformations; WORST just adds an eps term.
TEST_CASE_TEMPLATE("Error-free transformations - EXACT", T, std::float16_t,
                   std::float32_t, std::float64_t) {
  const L eps = eps_of<T>();
  const T one = static_cast<T>(1);
  const T half_eps = static_cast<T>(running_error::re_eps<T> / 2);

  // 1 + eps/2 rounds to 1, losing the whole addend; eft_sum recovers it.
  const re_exact_t<T> s =
      re_exact_t<T>(one, T(0)) + re_exact_t<T>(half_eps, T(0));
  CHECK(wide(s.val) == L(1));
  CHECK(wide(s.val) - wide(s.err) == 1 + eps / 2);

  // 1 - eps/2 is representable: nothing lost, no error reported.
  const re_exact_t<T> d =
      re_exact_t<T>(one, T(0)) - re_exact_t<T>(half_eps, T(0));
  CHECK(wide(d.val) == 1 - eps / 2);
  CHECK(wide(d.err) == L(0));

  // (1 + eps)^2 rounds to 1 + 2eps; eft_prod returns the dropped eps^2.
  const T u = static_cast<T>(1 + eps);
  const re_exact_t<T> p = re_exact_t<T>(u, T(0)) * re_exact_t<T>(u, T(0));
  CHECK(wide(p.val) == 1 + 2 * eps);
  CHECK(wide(p.err) == -eps * eps);

  // 1/3 is representable in no binary format; eft_quot rounds once.
  check_exact(re_exact_t<T>(one, T(0)) / re_exact_t<T>(static_cast<T>(3), T(0)),
              L(1) / L(3));
}

TEST_CASE_TEMPLATE("Math functions", T, std::float16_t, std::float32_t,
                   std::float64_t) {
  const double da = 0.5, db = 0.1;
  const L ra = wide(da), rb = wide(db);
  re_worst_t<T> aw(da), bw(db);
  re_exact_t<T> ae(da), be(db);

  // Unqualified calls resolve by ADL, std:: ones through the header overloads.
  CHECK_BOTH(abs(-b), rb);
  CHECK_BOTH(sqrt(a), std::sqrt(ra));
  CHECK_BOTH(std::log(b), std::log(rb));
  CHECK_BOTH(sin(a), std::sin(ra));
  CHECK_BOTH(std::cos(a), std::cos(ra));
  CHECK_BOTH(tan(a), std::tan(ra));
  CHECK_BOTH(std::atan(a), std::atan(ra));
  CHECK_BOTH(acos(a), std::acos(ra));
  CHECK_BOTH(atan2(b, a), std::atan2(rb, ra));           // atan2(y, x)
  CHECK_BOTH(std::atan2(b, 0.5), std::atan2(rb, 0.5L));  // scalar overload
  CHECK_BOTH(pow(a, 2), ra * ra);                        // integral exponent
  CHECK_BOTH(std::pow(a, static_cast<T>(3)), ra * ra * ra);

  // abs is a sign flip: no rounding, and it undoes the negation in both modes.
  CHECK(wide(abs(-bw).err) == wide(bw.err));
  CHECK(wide(abs(-be).err) == wide(be.err));
}

TEST_CASE_TEMPLATE("Integer powers are products", T, std::float16_t,
                   std::float32_t, std::float64_t) {
  const double dx = 0.1;
  const L rx = wide(dx);
  re_worst_t<T> xw(dx);
  re_exact_t<T> xe(dx);

  // x^0 is exactly 1, with nothing accumulated.
  CHECK(wide(pow(xw, 0).val) == L(1));
  CHECK(wide(pow(xw, 0).err) == L(0));
  CHECK(wide(pow(xe, 0).val) == L(1));
  CHECK(wide(pow(xe, 0).err) == L(0));

  // x^1 is x untouched: no product, no rounding.
  CHECK(wide(pow(xw, 1).val) == wide(xw.val));
  CHECK(wide(pow(xw, 1).err) == wide(xw.err));
  CHECK(wide(pow(xe, 1).val) == wide(xe.val));
  CHECK(wide(pow(xe, 1).err) == wide(xe.err));

  // A left-associated product chain, bit for bit.
  CHECK(wide(pow(xw, 3).val) == wide((xw * xw * xw).val));
  CHECK(wide(pow(xw, 3).err) == wide((xw * xw * xw).err));
  CHECK(wide(pow(xe, 3).val) == wide((xe * xe * xe).val));
  CHECK(wide(pow(xe, 3).err) == wide((xe * xe * xe).err));

  check(pow(xw, 2), pow(xe, 2), rx * rx);
  check(pow(xw, -2), pow(xe, -2), 1 / (rx * rx));

  // Same product sequence in both modes, so the values agree.
  CHECK(wide(pow(xw, 7).val) == wide(pow(xe, 7).val));
}

TEST_CASE_TEMPLATE("fmax / fmin and comparisons", T, std::float16_t,
                   std::float32_t, std::float64_t) {
  re_worst_t<T> aw(0.1), bw(0.2);
  re_exact_t<T> ae(0.1), be(0.2);

  // One operand is selected whole, error and all.
  CHECK(wide(fmax(aw, bw).err) == wide(bw.err));
  CHECK(wide(fmax(ae, be).err) == wide(be.err));
  CHECK(wide(fmin(aw, bw).err) == wide(aw.err));
  CHECK(wide(fmin(ae, be).err) == wide(ae.err));

  // Ties keep the first operand.
  CHECK(wide(fmax(aw, aw).err) == wide(aw.err));
  CHECK(wide(fmax(ae, ae).err) == wide(ae.err));

  // Scalar overloads.
  CHECK(wide(std::fmax(aw, 1.0).val) == L(1));
  CHECK(wide(std::fmax(ae, 1.0).val) == L(1));
  CHECK(wide(std::fmin(1.0, aw).val) == wide(aw.val));
  CHECK(wide(std::fmin(1.0, ae).val) == wide(ae.val));

  // Comparisons look at the value only.
  const re_worst_t<T> dirty(aw.val, static_cast<T>(1));  // same val, huge err
  CHECK(aw == dirty);
  CHECK(aw != bw);
  CHECK(ae != be);
  CHECK(aw <= bw);
  CHECK(ae <= be);
  CHECK(bw >= aw);
  CHECK(be >= ae);
}

TEST_CASE_TEMPLATE("Complex-like interface", T, std::float16_t, std::float32_t,
                   std::float64_t) {
  re_worst_t<T> w(0.5);
  re_exact_t<T> e(0.5);

  CHECK(wide(std::real(w)) == wide(w.val));
  CHECK(wide(std::real(e)) == wide(e.val));
  CHECK(wide(std::imag(w)) == L(0));
  CHECK(wide(std::imag(e)) == L(0));
  CHECK(wide(std::norm(w)) == L(0.25));
  CHECK(wide(std::norm(e)) == L(0.25));
}

TEST_CASE_TEMPLATE("The WORST bound covers the EXACT error", T, std::float16_t,
                   std::float32_t, std::float64_t) {
  const double da = 0.1, db = 0.2;
  const L ra = wide(da), rb = wide(db);

  re_worst_t<T> aw(da), bw(db);
  re_exact_t<T> ae(da), be(db);

  // Values agree bit for bit; the signed error never exceeds the bound.
  const re_worst_t<T> w = aw * bw + aw / bw - aw;
  const re_exact_t<T> x = ae * be + ae / be - ae;
  const L exact = ra * rb + ra / rb - ra;

  CHECK(wide(w.val) == wide(x.val));
  CHECK(std::fabs(wide(x.err)) <= wide(w.err));
  check(w, x, exact);
}
