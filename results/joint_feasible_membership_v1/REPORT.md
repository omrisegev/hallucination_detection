# Information-feasible refinement: full matched evaluation

| Method | PB % | Within AUC |
|---|---:|---:|
| base__old | 38.7300 | 0.750078 |
| base__continuous | 38.0111 | 0.748206 |
| base__equal | 37.8196 | 0.749591 |
| duplicates__old | 38.7300 | 0.750078 |
| duplicates__continuous | 37.5348 | 0.745811 |
| duplicates__equal | 37.5137 | 0.746661 |
| noise__old | 38.7300 | 0.750078 |
| noise__continuous | 37.6288 | 0.750537 |
| noise__equal | 37.8900 | 0.748021 |
| innovation5 | 39.8314 | 0.760293 |
| historical_bocpd | 40.3676 | 0.763223 |
| near_copies__old | 37.7185 | 0.748723 |
| near_copies__continuous | 37.5544 | 0.745933 |
| near_copies__equal | 37.5929 | 0.746523 |
| structured_noise__old | 37.9235 | 0.748005 |
| structured_noise__continuous | 37.8974 | 0.749137 |
| structured_noise__equal | 36.8025 | 0.742034 |
| base__previous | 38.7300 | 0.750078 |
| duplicates__previous | 38.7300 | 0.750078 |
| noise__previous | 38.7300 | 0.750078 |
| near_copies__previous | 37.7185 | 0.748723 |
| structured_noise__previous | 38.7300 | 0.750078 |
| base__regroup_reference | 38.3374 | 0.750672 |
| duplicates__regroup_reference | 38.3374 | 0.750672 |
| noise__regroup_reference | 38.3374 | 0.750672 |
| near_copies__regroup_reference | 38.2991 | 0.750572 |
| structured_noise__regroup_reference | 38.3374 | 0.750672 |
| base__feasible | 38.3444 | 0.749885 |
| duplicates__feasible | 38.3444 | 0.749885 |
| noise__feasible | 38.3444 | 0.749885 |
| near_copies__feasible | 38.5417 | 0.750942 |
| structured_noise__feasible | 38.3444 | 0.749885 |

Native: {'base': 13769, 'duplicates': 13769, 'noise': 13769, 'near_copies': 13769, 'structured_noise': 13769}
Preservation: {'duplicates': True, 'noise': True, 'near_copies': True, 'structured_noise': True}

```json
{
  "near_copies__feasible minus near_copies__regroup_reference": {
    "pb": {
      "point": 0.0024264409236251505,
      "low": -0.00022508424006149053,
      "high": 0.005404175547920782,
      "confidence": 0.9958333333333333,
      "draws": 10000
    },
    "within": {
      "point": 0.000369583995567746,
      "low": -0.0004897906759664921,
      "high": 0.001299647789151613,
      "confidence": 0.9958333333333333,
      "draws": 10000
    }
  },
  "base__feasible minus base__previous": {
    "pb": {
      "point": -0.003855227006162476,
      "low": -0.011086479260000845,
      "high": 0.0032451091748724567,
      "confidence": 0.9958333333333333,
      "draws": 10000
    },
    "within": {
      "point": -0.0001922744566241752,
      "low": -0.0021424548508732445,
      "high": 0.0017168248738763168,
      "confidence": 0.9958333333333333,
      "draws": 10000
    }
  },
  "duplicates__feasible minus base__feasible": {
    "pb": {
      "point": 0.0,
      "low": 0.0,
      "high": 0.0,
      "confidence": 0.9958333333333333,
      "draws": 10000
    },
    "within": {
      "point": 0.0,
      "low": 0.0,
      "high": 0.0,
      "confidence": 0.9958333333333333,
      "draws": 10000
    }
  },
  "noise__feasible minus base__feasible": {
    "pb": {
      "point": 0.0,
      "low": 0.0,
      "high": 0.0,
      "confidence": 0.9958333333333333,
      "draws": 10000
    },
    "within": {
      "point": 0.0,
      "low": 0.0,
      "high": 0.0,
      "confidence": 0.9958333333333333,
      "draws": 10000
    }
  },
  "near_copies__feasible minus base__feasible": {
    "pb": {
      "point": 0.001972514237453149,
      "low": -0.0038003860833929148,
      "high": 0.008207805856936446,
      "confidence": 0.9958333333333333,
      "draws": 10000
    },
    "within": {
      "point": 0.0010562121560063753,
      "low": -0.00010511448999828837,
      "high": 0.0022833692549552725,
      "confidence": 0.9958333333333333,
      "draws": 10000
    }
  },
  "structured_noise__feasible minus base__feasible": {
    "pb": {
      "point": 0.0,
      "low": 0.0,
      "high": 0.0,
      "confidence": 0.9958333333333333,
      "draws": 10000
    },
    "within": {
      "point": 0.0,
      "low": 0.0,
      "high": 0.0,
      "confidence": 0.9958333333333333,
      "draws": 10000
    }
  }
}
```
