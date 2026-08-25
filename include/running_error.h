#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

#if defined(__clang__)
// Clang does not yet implement the C++23 <stdfloat> fixed-width
// floating-point types (std::float16_t/float32_t/float64_t).
namespace std {
using float16_t = _Float16;
using float32_t = float;
using float64_t = double;
}  // namespace std
#else
#include <stdfloat>
#endif

namespace running_error {

static_assert(std::numeric_limits<std::float64_t>::digits == 53);
static_assert(sizeof(std::float64_t) == 8);

static_assert(std::numeric_limits<std::float32_t>::digits == 24);
static_assert(sizeof(std::float32_t) == 4);

// No available mechanism to check the significant width for float16
static_assert(sizeof(std::float16_t) == 2);

enum class ErrorMode { WORST, EXACT };

// Exact type represents a wider type used in EXACT mode for reference value.
template <typename T>
struct exact_type {
  using type = T;
};

// Need to use long double, not std::float128_t: cling is C++20 and has neither
// that nor __float128. It is binary128 on aarch64, 80-bit (11 spare bits)
// elsewhere.
template <>
struct exact_type<std::float64_t> {
  using type = long double;

  // Mantissa bits, not sizeof: x86-64 pads the 80-bit format to 16 bytes.
  static_assert(std::numeric_limits<long double>::digits >= 64,
                "must support a format with at least 64-bit significand");
};

template <>
struct exact_type<std::float32_t> {
  using type = std::float64_t;
};

template <>
struct exact_type<std::float16_t> {
  using type = std::float32_t;
};

template <typename T>
using exact_type_t = typename exact_type<T>::type;

// ---------------------------------------------------------------------------
// Specializations for std::float16_t
// ---------------------------------------------------------------------------
template <typename V>
constexpr V re_abs(V x) noexcept {
  return x < V(0) ? -x : x;
}
template <typename V>
struct re_math_promote {
  using type = V;
};
template <>
struct re_math_promote<std::float16_t> {
  using type = std::float32_t;
};
template <typename V>
using re_math_promote_t = typename re_math_promote<V>::type;

template <typename V>
inline V re_sqrt(V x) {
  return static_cast<V>(std::sqrt(static_cast<re_math_promote_t<V>>(x)));
}
template <typename V>
inline V re_log(V x) {
  return static_cast<V>(std::log(static_cast<re_math_promote_t<V>>(x)));
}
template <typename V>
inline V re_sin(V x) {
  return static_cast<V>(std::sin(static_cast<re_math_promote_t<V>>(x)));
}
template <typename V>
inline V re_cos(V x) {
  return static_cast<V>(std::cos(static_cast<re_math_promote_t<V>>(x)));
}
template <typename V>
inline V re_acos(V x) {
  return static_cast<V>(std::acos(static_cast<re_math_promote_t<V>>(x)));
}
template <typename V>
inline V re_tan(V x) {
  return static_cast<V>(std::tan(static_cast<re_math_promote_t<V>>(x)));
}
template <typename V>
inline V re_atan(V x) {
  return static_cast<V>(std::atan(static_cast<re_math_promote_t<V>>(x)));
}
template <typename V>
inline V re_atan2(V y, V x) {
  using P = re_math_promote_t<V>;
  return static_cast<V>(std::atan2(static_cast<P>(y), static_cast<P>(x)));
}
template <typename V>
inline V re_pow(V x, V n) {
  using P = re_math_promote_t<V>;
  return static_cast<V>(std::pow(static_cast<P>(x), static_cast<P>(n)));
}
template <typename V>
inline V re_fma(V x, V y, V z) {
  using P = re_math_promote_t<V>;
  return static_cast<V>(
      std::fma(static_cast<P>(x), static_cast<P>(y), static_cast<P>(z)));
}
template <typename T>
inline constexpr T re_eps = std::numeric_limits<T>::epsilon();

// Error of representing `val` in precision `T`, signed as computed - true.
// Subtracting in S, the wider type, is exact before rounding into T.
template <typename T, typename S>
constexpr T representation_error(S val) noexcept {
  return static_cast<T>(static_cast<S>(static_cast<T>(val)) - val);
}

// Local rounding of one operation: computed - exact, differenced in the exact
// type E and rounded into the compute type T.
template <typename T, typename E>
constexpr T local_error(T computed, E exact) noexcept {
  return static_cast<T>(static_cast<E>(computed) - exact);
}

// ---------------------------------------------------------------------------
// Error-free Transformations
// ---------------------------------------------------------------------------

template <typename T>
inline T eft_sum(T a, T b, T s) noexcept {
  const T ap = s - b, bp = s - ap;  // Knuth TwoSum
  return (ap - a) + (bp - b);
}

template <typename T>
inline T eft_prod(T a, T b, T p) noexcept {
  return -re_fma(a, b, -p);
}

// fma gives the residual a - q*b exactly; dividing it by b rounds once.
template <typename T>
inline T eft_quot(T a, T b, T q) noexcept {
  return -re_fma(-q, b, a) / b;
}

// ---------------------------------------------------------------------------
// WORST mode: running upper bound on absolute rounding error
// ---------------------------------------------------------------------------

template <typename T, ErrorMode Mode = ErrorMode::WORST>
struct running_error_t {
  using re_t = running_error_t;

  T val;
  T error;

  constexpr running_error_t(T _val = T(0), T _err = T(0)) noexcept
      : val(_val), error(_err) {}

  // Construct from a plain arithmetic value.
  // The explicit cast covers narrowing conversions such as
  // std::float64_t -> std::float16_t that are not implicit in C++23. Disabled
  // for S == T so the (T, T) constructor above stays unambiguous. Seeded with
  // the representation error of the conversion.
  template <typename S,
            std::enable_if_t<std::is_arithmetic_v<S> && !std::is_same_v<S, T>,
                             int> = 0>
  constexpr running_error_t(S _val) noexcept
      : running_error_t(static_cast<T>(_val),
                        re_abs(representation_error<T>(_val))) {}

  //
  // Basic arithmetic operators, type homogeneous only
  //

  // f(a,b) = a + b:  ∂f/∂a = 1, ∂f/∂b = 1
  re_t operator+(const re_t& other) const {
    const T new_val = val + other.val;
    return re_t{new_val, error + other.error + re_eps<T> * re_abs(new_val)};
  }

  // f(a,b) = a - b:  ∂f/∂a = 1, ∂f/∂b = -1
  re_t operator-(const re_t& other) const {
    const T new_val = val - other.val;
    return re_t{new_val, error + other.error + re_eps<T> * re_abs(new_val)};
  }

  // Negation is exact in IEEE 754 (sign bit flip, no rounding).
  re_t operator-() const { return re_t{-val, error}; }

  // f(a,b) = a * b:  ∂f/∂a = b, ∂f/∂b = a
  re_t operator*(const re_t& other) const {
    const T new_val = val * other.val;
    return re_t{new_val, re_abs(other.val) * error + re_abs(val) * other.error +
                             re_eps<T> * re_abs(new_val)};
  }

  // f(a,b) = a / b:  ∂f/∂a = 1/b, ∂f/∂b = -a/b² = -new_val/b
  re_t operator/(const re_t& other) const {
    const T new_val = val / other.val;
    return re_t{new_val, error / re_abs(other.val) +
                             re_abs(new_val / other.val) * other.error +
                             re_eps<T> * re_abs(new_val)};
  }

  //
  // Compound assignment operators, type homogeneous only
  //

  re_t& operator+=(const re_t& other) { return *this = *this + other; }
  re_t& operator-=(const re_t& other) { return *this = *this - other; }
  re_t& operator*=(const re_t& other) { return *this = *this * other; }
  re_t& operator/=(const re_t& other) { return *this = *this / other; }

  //
  // Compound assignment with scalar
  //

  re_t& operator*=(T scalar) { return *this = re_t{scalar, T(0)} * *this; }

  //
  // Comparison operators
  //

  bool operator==(const re_t& other) const {
    return val == other.val && error == other.error;
  }
  bool operator!=(const re_t& other) const { return !(*this == other); }

  // Note: this is the machine epsilon (e.g. 2^-52 for std::float64_t), i.e.
  // twice the unit roundoff u = 2^-53 of round-to-nearest. The local rounding
  // term is therefore a factor ~2 looser than strictly necessary. This keeps
  // the bound a safe (conservative) upper bound.
  static constexpr T eps = re_eps<T>;
};

// ---------------------------------------------------------------------------
// EXACT mode: signed accumulated error via first-order derivative propagation.
//             Local rounding is exact for + - * / (error-free transformations)
//             and computed in the wider exact_type for the math functions.
// ---------------------------------------------------------------------------

template <typename T>
struct running_error_t<T, ErrorMode::EXACT> {
  using E = exact_type_t<T>;
  using re_t = running_error_t;

  // Without a strictly wider reference the local rounding is identically zero,
  // which would report a flawless computation rather than fail.
  static_assert(std::numeric_limits<E>::digits > std::numeric_limits<T>::digits,
                "ErrorMode::EXACT needs an exact_type wider than the compute "
                "type.");

  T val;
  T error;

  constexpr running_error_t(T _val = T(0), T _err = T(0)) noexcept
      : val(_val), error(_err) {}

  // Construct from a plain arithmetic value.
  // The explicit cast covers narrowing conversions such as
  // std::float64_t -> std::float16_t that are not implicit in C++23. Disabled
  // for S == T so the (T, T) constructor above stays unambiguous. Seeded with
  // the signed representation error of the conversion.
  template <typename S,
            std::enable_if_t<std::is_arithmetic_v<S> && !std::is_same_v<S, T>,
                             int> = 0>
  constexpr running_error_t(S _val) noexcept
      : running_error_t(static_cast<T>(_val), representation_error<T>(_val)) {}

  //
  // Basic arithmetic operators, type homogeneous only
  //

  // f(a,b) = a + b:  ∂f/∂a = 1, ∂f/∂b = 1
  re_t operator+(const re_t& other) const {
    const T new_val = val + other.val;
    return re_t{new_val,
                error + other.error + eft_sum(val, other.val, new_val)};
  }

  // f(a,b) = a - b:  ∂f/∂a = 1, ∂f/∂b = -1
  re_t operator-(const re_t& other) const {
    const T new_val = val - other.val;
    return re_t{new_val,
                error - other.error + eft_sum(val, -other.val, new_val)};
  }

  // Negation is exact in IEEE 754: ∂(-a)/∂a = -1, no local rounding.
  re_t operator-() const { return re_t{-val, -error}; }

  // f(a,b) = a * b:  ∂f/∂a = b, ∂f/∂b = a
  re_t operator*(const re_t& other) const {
    const T new_val = val * other.val;
    return re_t{new_val, other.val * error + val * other.error +
                             eft_prod(val, other.val, new_val)};
  }

  // f(a,b) = a / b:  ∂f/∂a = 1/b, ∂f/∂b = -a/b² = -new_val/b
  re_t operator/(const re_t& other) const {
    const T new_val = val / other.val;
    return re_t{new_val, error / other.val - new_val / other.val * other.error +
                             eft_quot(val, other.val, new_val)};
  }

  //
  // Compound assignment operators, type homogeneous only
  //

  re_t& operator+=(const re_t& other) { return *this = *this + other; }
  re_t& operator-=(const re_t& other) { return *this = *this - other; }
  re_t& operator*=(const re_t& other) { return *this = *this * other; }
  re_t& operator/=(const re_t& other) { return *this = *this / other; }

  //
  // Compound assignment with scalar
  //

  re_t& operator*=(T scalar) { return *this = re_t{scalar, T(0)} * *this; }

  //
  // Comparison operators
  //

  bool operator==(const re_t& other) const {
    return val == other.val && error == other.error;
  }
  bool operator!=(const re_t& other) const { return !(*this == other); }

  // Machine epsilon of the value type, provided for parity with WORST mode.
  static constexpr T eps = re_eps<T>;
};

// ---------------------------------------------------------------------------
// Convenience aliases
// ---------------------------------------------------------------------------

template <typename T>
using re_worst_t = running_error_t<T, ErrorMode::WORST>;
template <typename T>
using re_exact_t = running_error_t<T, ErrorMode::EXACT>;

// {T val; T err}: a homogeneous pair, 2*sizeof(T) with T's alignment.
static_assert(sizeof(re_worst_t<std::float64_t>) == 16 &&
              alignof(re_worst_t<std::float64_t>) == 8);
static_assert(sizeof(re_exact_t<std::float64_t>) == 16 &&
              alignof(re_exact_t<std::float64_t>) == 8);

static_assert(sizeof(re_worst_t<std::float32_t>) == 8 &&
              alignof(re_worst_t<std::float32_t>) == 4);
static_assert(sizeof(re_exact_t<std::float32_t>) == 8 &&
              alignof(re_exact_t<std::float32_t>) == 4);

// ---------------------------------------------------------------------------
// Free functions — WORST mode
//
// Free functions are used for inhomogeneous operations and to allow math
// function overloading. Missing standard scalar operations are delegated
// to the homogeneous operators which cover exactly the same math.
// ---------------------------------------------------------------------------

template <typename T, typename S,
          std::enable_if_t<std::is_arithmetic_v<S>, int> = 0>
re_worst_t<T> operator*(S lhs, const re_worst_t<T>& rhs) {
  return re_worst_t<T>(lhs) * rhs;
}

template <typename T, typename S,
          std::enable_if_t<std::is_arithmetic_v<S>, int> = 0>
re_worst_t<T> operator*(const re_worst_t<T>& lhs, S rhs) {
  return lhs * re_worst_t<T>(rhs);
}

template <typename T, typename S,
          std::enable_if_t<std::is_arithmetic_v<S>, int> = 0>
re_worst_t<T> operator+(S lhs, const re_worst_t<T>& rhs) {
  return re_worst_t<T>(lhs) + rhs;
}

template <typename T, typename S,
          std::enable_if_t<std::is_arithmetic_v<S>, int> = 0>
re_worst_t<T> operator+(const re_worst_t<T>& lhs, S rhs) {
  return lhs + re_worst_t<T>(rhs);
}

template <typename T, typename S,
          std::enable_if_t<std::is_arithmetic_v<S>, int> = 0>
re_worst_t<T> operator-(S lhs, const re_worst_t<T>& rhs) {
  return re_worst_t<T>(lhs) - rhs;
}

template <typename T, typename S,
          std::enable_if_t<std::is_arithmetic_v<S>, int> = 0>
re_worst_t<T> operator-(const re_worst_t<T>& lhs, S rhs) {
  return lhs - re_worst_t<T>(rhs);
}

template <typename T, typename S,
          std::enable_if_t<std::is_arithmetic_v<S>, int> = 0>
re_worst_t<T> operator/(S lhs, const re_worst_t<T>& rhs) {
  return re_worst_t<T>(lhs) / rhs;
}

template <typename T, typename S,
          std::enable_if_t<std::is_arithmetic_v<S>, int> = 0>
re_worst_t<T> operator/(const re_worst_t<T>& lhs, S rhs) {
  return lhs / re_worst_t<T>(rhs);
}

// Absolute value adds no error, just flips the sign bit if negative.
template <typename T>
re_worst_t<T> abs(const re_worst_t<T>& x) {
  return re_worst_t<T>{re_abs(x.val), x.error};
}

// sqrt: dy/dx = 1/(2*sqrt(x))
template <typename T>
re_worst_t<T> sqrt(const re_worst_t<T>& x) {
  const T new_val = re_sqrt(x.val);
  return re_worst_t<T>{new_val,
                       x.error / (2 * new_val) + re_eps<T> * re_abs(new_val)};
}

// log: dy/dx = 1/x
template <typename T>
re_worst_t<T> log(const re_worst_t<T>& x) {
  const T new_val = re_log(x.val);
  return re_worst_t<T>{new_val, x.error / x.val + re_eps<T> * re_abs(new_val)};
}

// sin: dy/dx = cos(x)
template <typename T>
re_worst_t<T> sin(const re_worst_t<T>& x) {
  const T new_val = re_sin(x.val);
  const T deriv = re_abs(re_cos(x.val));
  return re_worst_t<T>{new_val, deriv * x.error + re_eps<T> * re_abs(new_val)};
}

// cos: dy/dx = -sin(x)
template <typename T>
re_worst_t<T> cos(const re_worst_t<T>& x) {
  const T new_val = re_cos(x.val);
  const T deriv = re_abs(re_sin(x.val));
  return re_worst_t<T>{new_val, deriv * x.error + re_eps<T> * re_abs(new_val)};
}

// acos: dy/dx = -1/sqrt(1 - x²)
template <typename T>
re_worst_t<T> acos(const re_worst_t<T>& x) {
  const T new_val = re_acos(x.val);
  const T deriv = 1 / re_sqrt(1 - x.val * x.val);
  return re_worst_t<T>{new_val, deriv * x.error + re_eps<T> * re_abs(new_val)};
}

// tan: dy/dx = 1/cos²(x) = 1 + tan²(x), rewritten via new_val to avoid a
// second std::cos call.
template <typename T>
re_worst_t<T> tan(const re_worst_t<T>& x) {
  const T new_val = re_tan(x.val);
  const T deriv = 1 + new_val * new_val;
  return re_worst_t<T>{new_val, deriv * x.error + re_eps<T> * re_abs(new_val)};
}

// atan: dy/dx = 1/(1 + x²)
template <typename T>
re_worst_t<T> atan(const re_worst_t<T>& x) {
  const T new_val = re_atan(x.val);
  const T deriv = 1 / (1 + x.val * x.val);
  return re_worst_t<T>{new_val, deriv * x.error + re_eps<T> * re_abs(new_val)};
}

// atan2(y, x): ∂/∂y = x/(x²+y²), ∂/∂x = -y/(x²+y²)
template <typename T>
re_worst_t<T> atan2(const re_worst_t<T>& y, const re_worst_t<T>& x) {
  const T new_val = re_atan2(y.val, x.val);
  const T d = x.val * x.val + y.val * y.val;
  return re_worst_t<T>{new_val, re_abs(x.val / d) * y.error +
                                    re_abs(y.val / d) * x.error +
                                    re_eps<T> * re_abs(new_val)};
}

template <typename T, typename S,
          std::enable_if_t<std::is_arithmetic_v<S>, int> = 0>
re_worst_t<T> atan2(const re_worst_t<T>& y, S x) {
  return atan2(y, re_worst_t<T>(x));
}

template <typename T, typename S,
          std::enable_if_t<std::is_arithmetic_v<S>, int> = 0>
re_worst_t<T> atan2(S y, const re_worst_t<T>& x) {
  return atan2(re_worst_t<T>(y), x);
}

// fmax / fmin: exact selection of one operand; no rounding is introduced, and
// the chosen operand's error carries through unchanged.
template <typename T>
re_worst_t<T> fmax(const re_worst_t<T>& a, const re_worst_t<T>& b) {
  return (b.val > a.val) ? b : a;
}

template <typename T>
re_worst_t<T> fmin(const re_worst_t<T>& a, const re_worst_t<T>& b) {
  return (b.val < a.val) ? b : a;
}

template <typename T, typename S,
          std::enable_if_t<std::is_arithmetic_v<S>, int> = 0>
re_worst_t<T> fmax(const re_worst_t<T>& a, S b) {
  return fmax(a, re_worst_t<T>(b));
}

template <typename T, typename S,
          std::enable_if_t<std::is_arithmetic_v<S>, int> = 0>
re_worst_t<T> fmax(S a, const re_worst_t<T>& b) {
  return fmax(re_worst_t<T>(a), b);
}

template <typename T, typename S,
          std::enable_if_t<std::is_arithmetic_v<S>, int> = 0>
re_worst_t<T> fmin(const re_worst_t<T>& a, S b) {
  return fmin(a, re_worst_t<T>(b));
}

template <typename T, typename S,
          std::enable_if_t<std::is_arithmetic_v<S>, int> = 0>
re_worst_t<T> fmin(S a, const re_worst_t<T>& b) {
  return fmin(re_worst_t<T>(a), b);
}

// pow(x, n): dy/dx = n*x^(n-1)
template <typename T>
re_worst_t<T> pow(const re_worst_t<T>& x, T n) {
  const T new_val = re_pow(x.val, n);
  const T deriv = re_abs(n * re_pow(x.val, n - 1));
  return re_worst_t<T>{new_val, deriv * x.error + re_eps<T> * re_abs(new_val)};
}

template <typename T>
re_worst_t<T> pow(const re_worst_t<T>& x, int n) {
  return pow(x, static_cast<T>(n));
}

// ---------------------------------------------------------------------------
// Free functions — EXACT mode
//
// Free functions are used for inhomogeneous operations and to allow ADL-based
// math function overloading.
//
// Scalar has no accumulated error, so derivative propagation is one-sided.
// Local rounding is always correctly computed down in the delegated homogeneous
// operators by treating the scalar as an exact mathematical `val` with 0 error.
// ---------------------------------------------------------------------------

template <typename T, typename S,
          std::enable_if_t<std::is_arithmetic_v<S>, int> = 0>
re_exact_t<T> operator*(S lhs, const re_exact_t<T>& rhs) {
  return re_exact_t<T>(lhs) * rhs;
}

template <typename T, typename S,
          std::enable_if_t<std::is_arithmetic_v<S>, int> = 0>
re_exact_t<T> operator*(const re_exact_t<T>& lhs, S rhs) {
  return lhs * re_exact_t<T>(rhs);
}

template <typename T, typename S,
          std::enable_if_t<std::is_arithmetic_v<S>, int> = 0>
re_exact_t<T> operator+(S lhs, const re_exact_t<T>& rhs) {
  return re_exact_t<T>(lhs) + rhs;
}

template <typename T, typename S,
          std::enable_if_t<std::is_arithmetic_v<S>, int> = 0>
re_exact_t<T> operator+(const re_exact_t<T>& lhs, S rhs) {
  return lhs + re_exact_t<T>(rhs);
}

template <typename T, typename S,
          std::enable_if_t<std::is_arithmetic_v<S>, int> = 0>
re_exact_t<T> operator-(S lhs, const re_exact_t<T>& rhs) {
  return re_exact_t<T>(lhs) - rhs;
}

template <typename T, typename S,
          std::enable_if_t<std::is_arithmetic_v<S>, int> = 0>
re_exact_t<T> operator-(const re_exact_t<T>& lhs, S rhs) {
  return lhs - re_exact_t<T>(rhs);
}

template <typename T, typename S,
          std::enable_if_t<std::is_arithmetic_v<S>, int> = 0>
re_exact_t<T> operator/(S lhs, const re_exact_t<T>& rhs) {
  return re_exact_t<T>(lhs) / rhs;
}

template <typename T, typename S,
          std::enable_if_t<std::is_arithmetic_v<S>, int> = 0>
re_exact_t<T> operator/(const re_exact_t<T>& lhs, S rhs) {
  return lhs / re_exact_t<T>(rhs);
}

// abs is exact in IEEE 754, derivative = sign(val)
template <typename T>
re_exact_t<T> abs(const re_exact_t<T>& x) {
  const T sign = (x.val > T(0)) ? T(1) : (x.val < T(0)) ? T(-1) : T(0);
  return re_exact_t<T>{re_abs(x.val), sign * x.error};
}

// sqrt: ∂(√x)/∂x = 1/(2√x)
template <typename T>
re_exact_t<T> sqrt(const re_exact_t<T>& x) {
  using E = exact_type_t<T>;
  const T new_val = re_sqrt(x.val);
  const E exact_val = re_sqrt(static_cast<E>(x.val));
  const T deriv = 1 / (2 * new_val);
  return re_exact_t<T>{new_val,
                       deriv * x.error + local_error(new_val, exact_val)};
}

// log: ∂(ln x)/∂x = 1/x
template <typename T>
re_exact_t<T> log(const re_exact_t<T>& x) {
  using E = exact_type_t<T>;
  const T new_val = re_log(x.val);
  const E exact_val = re_log(static_cast<E>(x.val));
  const T deriv = 1 / x.val;
  return re_exact_t<T>{new_val,
                       deriv * x.error + local_error(new_val, exact_val)};
}

// sin: ∂(sin x)/∂x = cos(x)
template <typename T>
re_exact_t<T> sin(const re_exact_t<T>& x) {
  using E = exact_type_t<T>;
  const T new_val = re_sin(x.val);
  const E exact_val = re_sin(static_cast<E>(x.val));
  const T deriv = re_cos(x.val);
  return re_exact_t<T>{new_val,
                       deriv * x.error + local_error(new_val, exact_val)};
}

// cos: ∂(cos x)/∂x = -sin(x)
template <typename T>
re_exact_t<T> cos(const re_exact_t<T>& x) {
  using E = exact_type_t<T>;
  const T new_val = re_cos(x.val);
  const E exact_val = re_cos(static_cast<E>(x.val));
  const T deriv = -re_sin(x.val);
  return re_exact_t<T>{new_val,
                       deriv * x.error + local_error(new_val, exact_val)};
}

// acos: ∂(acos x)/∂x = -1/sqrt(1 - x²)
template <typename T>
re_exact_t<T> acos(const re_exact_t<T>& x) {
  using E = exact_type_t<T>;
  const T new_val = re_acos(x.val);
  const E exact_val = re_acos(static_cast<E>(x.val));
  const T deriv = -1 / re_sqrt(1 - x.val * x.val);
  return re_exact_t<T>{new_val,
                       deriv * x.error + local_error(new_val, exact_val)};
}

// tan: ∂(tan x)/∂x = 1/cos²(x) = 1 + tan²(x)
template <typename T>
re_exact_t<T> tan(const re_exact_t<T>& x) {
  using E = exact_type_t<T>;
  const T new_val = re_tan(x.val);
  const E exact_val = re_tan(static_cast<E>(x.val));
  const T deriv = 1 + new_val * new_val;
  return re_exact_t<T>{new_val,
                       deriv * x.error + local_error(new_val, exact_val)};
}

// atan: ∂(atan x)/∂x = 1/(1 + x²)
template <typename T>
re_exact_t<T> atan(const re_exact_t<T>& x) {
  using E = exact_type_t<T>;
  const T new_val = re_atan(x.val);
  const E exact_val = re_atan(static_cast<E>(x.val));
  const T deriv = 1 / (1 + x.val * x.val);
  return re_exact_t<T>{new_val,
                       deriv * x.error + local_error(new_val, exact_val)};
}

// atan2(y, x): ∂/∂y = x/(x²+y²), ∂/∂x = -y/(x²+y²)
template <typename T>
re_exact_t<T> atan2(const re_exact_t<T>& y, const re_exact_t<T>& x) {
  using E = exact_type_t<T>;
  const T new_val = re_atan2(y.val, x.val);
  const E exact_val = re_atan2(static_cast<E>(y.val), static_cast<E>(x.val));
  const T d = x.val * x.val + y.val * y.val;
  return re_exact_t<T>{new_val, (x.val / d) * y.error - (y.val / d) * x.error +
                                    local_error(new_val, exact_val)};
}

template <typename T, typename S,
          std::enable_if_t<std::is_arithmetic_v<S>, int> = 0>
re_exact_t<T> atan2(const re_exact_t<T>& y, S x) {
  return atan2(y, re_exact_t<T>(x));
}

template <typename T, typename S,
          std::enable_if_t<std::is_arithmetic_v<S>, int> = 0>
re_exact_t<T> atan2(S y, const re_exact_t<T>& x) {
  return atan2(re_exact_t<T>(y), x);
}

// fmax / fmin: exact selection of one operand; no rounding is introduced, and
// the chosen operand's accumulated error carries through unchanged.
template <typename T>
re_exact_t<T> fmax(const re_exact_t<T>& a, const re_exact_t<T>& b) {
  return (b.val > a.val) ? b : a;
}

template <typename T>
re_exact_t<T> fmin(const re_exact_t<T>& a, const re_exact_t<T>& b) {
  return (b.val < a.val) ? b : a;
}

template <typename T, typename S,
          std::enable_if_t<std::is_arithmetic_v<S>, int> = 0>
re_exact_t<T> fmax(const re_exact_t<T>& a, S b) {
  return fmax(a, re_exact_t<T>(b));
}

template <typename T, typename S,
          std::enable_if_t<std::is_arithmetic_v<S>, int> = 0>
re_exact_t<T> fmax(S a, const re_exact_t<T>& b) {
  return fmax(re_exact_t<T>(a), b);
}

template <typename T, typename S,
          std::enable_if_t<std::is_arithmetic_v<S>, int> = 0>
re_exact_t<T> fmin(const re_exact_t<T>& a, S b) {
  return fmin(a, re_exact_t<T>(b));
}

template <typename T, typename S,
          std::enable_if_t<std::is_arithmetic_v<S>, int> = 0>
re_exact_t<T> fmin(S a, const re_exact_t<T>& b) {
  return fmin(re_exact_t<T>(a), b);
}

// pow(x, n): ∂(x^n)/∂x = n*x^(n-1), signed
template <typename T>
re_exact_t<T> pow(const re_exact_t<T>& x, T n) {
  using E = exact_type_t<T>;
  const T new_val = re_pow(x.val, n);
  const E exact_val = re_pow(static_cast<E>(x.val), static_cast<E>(n));
  const T deriv = n * re_pow(x.val, n - 1);
  return re_exact_t<T>{new_val,
                       deriv * x.error + local_error(new_val, exact_val)};
}

}  // namespace running_error

// ---------------------------------------------------------------------------
// std:: overloads — WORST mode
// ---------------------------------------------------------------------------

namespace std {

template <typename T>
using re_worst_t = running_error::re_worst_t<T>;

template <typename T>
re_worst_t<T> abs(const re_worst_t<T>& x) {
  return running_error::abs(x);
}

template <typename T>
re_worst_t<T> sqrt(const re_worst_t<T>& x) {
  return running_error::sqrt(x);
}

template <typename T>
re_worst_t<T> log(const re_worst_t<T>& x) {
  return running_error::log(x);
}

template <typename T>
re_worst_t<T> sin(const re_worst_t<T>& x) {
  return running_error::sin(x);
}

template <typename T>
re_worst_t<T> cos(const re_worst_t<T>& x) {
  return running_error::cos(x);
}

template <typename T>
re_worst_t<T> acos(const re_worst_t<T>& x) {
  return running_error::acos(x);
}

template <typename T>
re_worst_t<T> fmax(const re_worst_t<T>& a, const re_worst_t<T>& b) {
  return running_error::fmax(a, b);
}

template <typename T>
re_worst_t<T> fmin(const re_worst_t<T>& a, const re_worst_t<T>& b) {
  return running_error::fmin(a, b);
}

template <typename T, typename S,
          std::enable_if_t<std::is_arithmetic_v<S>, int> = 0>
re_worst_t<T> fmax(const re_worst_t<T>& a, S b) {
  return running_error::fmax(a, b);
}

template <typename T, typename S,
          std::enable_if_t<std::is_arithmetic_v<S>, int> = 0>
re_worst_t<T> fmax(S a, const re_worst_t<T>& b) {
  return running_error::fmax(a, b);
}

template <typename T, typename S,
          std::enable_if_t<std::is_arithmetic_v<S>, int> = 0>
re_worst_t<T> fmin(const re_worst_t<T>& a, S b) {
  return running_error::fmin(a, b);
}

template <typename T, typename S,
          std::enable_if_t<std::is_arithmetic_v<S>, int> = 0>
re_worst_t<T> fmin(S a, const re_worst_t<T>& b) {
  return running_error::fmin(a, b);
}

template <typename T>
re_worst_t<T> tan(const re_worst_t<T>& x) {
  return running_error::tan(x);
}

template <typename T>
re_worst_t<T> atan(const re_worst_t<T>& x) {
  return running_error::atan(x);
}

template <typename T>
re_worst_t<T> atan2(const re_worst_t<T>& y, const re_worst_t<T>& x) {
  return running_error::atan2(y, x);
}

template <typename T, typename S,
          std::enable_if_t<std::is_arithmetic_v<S>, int> = 0>
re_worst_t<T> atan2(const re_worst_t<T>& y, S x) {
  return running_error::atan2(y, x);
}

template <typename T, typename S,
          std::enable_if_t<std::is_arithmetic_v<S>, int> = 0>
re_worst_t<T> atan2(S y, const re_worst_t<T>& x) {
  return running_error::atan2(y, x);
}

template <typename T>
re_worst_t<T> pow(const re_worst_t<T>& x, T n) {
  return running_error::pow(x, n);
}

template <typename T>
re_worst_t<T> pow(const re_worst_t<T>& x, int n) {
  return running_error::pow(x, n);
}

// Complex-like interface (val is "real", no imaginary component)
template <typename T>
T real(const re_worst_t<T>& x) {
  return x.val;
}

template <typename T>
T imag(const re_worst_t<T>&) {
  return T(0);
}

template <typename T>
T norm(const re_worst_t<T>& x) {
  return x.val * x.val;
}

// ---------------------------------------------------------------------------
// std:: overloads — EXACT mode
// ---------------------------------------------------------------------------

template <typename T>
using re_exact_t = running_error::re_exact_t<T>;

template <typename T>
re_exact_t<T> abs(const re_exact_t<T>& x) {
  return running_error::abs(x);
}

template <typename T>
re_exact_t<T> sqrt(const re_exact_t<T>& x) {
  return running_error::sqrt(x);
}

template <typename T>
re_exact_t<T> log(const re_exact_t<T>& x) {
  return running_error::log(x);
}

template <typename T>
re_exact_t<T> sin(const re_exact_t<T>& x) {
  return running_error::sin(x);
}

template <typename T>
re_exact_t<T> cos(const re_exact_t<T>& x) {
  return running_error::cos(x);
}

template <typename T>
re_exact_t<T> acos(const re_exact_t<T>& x) {
  return running_error::acos(x);
}

template <typename T>
re_exact_t<T> fmax(const re_exact_t<T>& a, const re_exact_t<T>& b) {
  return running_error::fmax(a, b);
}

template <typename T>
re_exact_t<T> fmin(const re_exact_t<T>& a, const re_exact_t<T>& b) {
  return running_error::fmin(a, b);
}

template <typename T, typename S,
          std::enable_if_t<std::is_arithmetic_v<S>, int> = 0>
re_exact_t<T> fmax(const re_exact_t<T>& a, S b) {
  return running_error::fmax(a, b);
}

template <typename T, typename S,
          std::enable_if_t<std::is_arithmetic_v<S>, int> = 0>
re_exact_t<T> fmax(S a, const re_exact_t<T>& b) {
  return running_error::fmax(a, b);
}

template <typename T, typename S,
          std::enable_if_t<std::is_arithmetic_v<S>, int> = 0>
re_exact_t<T> fmin(const re_exact_t<T>& a, S b) {
  return running_error::fmin(a, b);
}

template <typename T, typename S,
          std::enable_if_t<std::is_arithmetic_v<S>, int> = 0>
re_exact_t<T> fmin(S a, const re_exact_t<T>& b) {
  return running_error::fmin(a, b);
}

template <typename T>
re_exact_t<T> tan(const re_exact_t<T>& x) {
  return running_error::tan(x);
}

template <typename T>
re_exact_t<T> atan(const re_exact_t<T>& x) {
  return running_error::atan(x);
}

template <typename T>
re_exact_t<T> atan2(const re_exact_t<T>& y, const re_exact_t<T>& x) {
  return running_error::atan2(y, x);
}

template <typename T, typename S,
          std::enable_if_t<std::is_arithmetic_v<S>, int> = 0>
re_exact_t<T> atan2(const re_exact_t<T>& y, S x) {
  return running_error::atan2(y, x);
}

template <typename T, typename S,
          std::enable_if_t<std::is_arithmetic_v<S>, int> = 0>
re_exact_t<T> atan2(S y, const re_exact_t<T>& x) {
  return running_error::atan2(y, x);
}

template <typename T>
re_exact_t<T> pow(const re_exact_t<T>& x, T n) {
  return running_error::pow(x, n);
}

template <typename T>
re_exact_t<T> pow(const re_exact_t<T>& x, int n) {
  return running_error::pow(x, n);
}

template <typename T>
T real(const re_exact_t<T>& x) {
  return x.val;
}

template <typename T>
T imag(const re_exact_t<T>&) {
  return T(0);
}

template <typename T>
T norm(const re_exact_t<T>& x) {
  return x.val * x.val;
}

}  // namespace std
