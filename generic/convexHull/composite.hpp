
#ifndef COMPOSITE_HPP
#define COMPOSITE_HPP

#include "concepts.hpp"

template<Regular T0, Regular T1>
  struct pair {
    pair() = default;
    
    pair(T0 m0, T1 m1)
      : m0(m0), m1(m1) { }

    T0 m0;
    T1 m1;
  };

template<Regular T0, Regular T1, Regular T2>
  struct triple {
    triple() = default;
    
    triple(T0 m0, T1 m1, T2 m2)
      : m0(m0), m1(m1), m2(m2) { }

    T0 m0;
    T1 m1;
    T2 m2;
  };

#endif
