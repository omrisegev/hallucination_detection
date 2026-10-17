# Information-mass refinement: full matched evaluation

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
| base__feasible_reference | 38.3444 | 0.749885 |
| duplicates__feasible_reference | 38.3444 | 0.749885 |
| noise__feasible_reference | 38.3444 | 0.749885 |
| near_copies__feasible_reference | 38.5417 | 0.750942 |
| structured_noise__feasible_reference | 38.3444 | 0.749885 |
| base__minimax_reference | 38.0732 | 0.748555 |
| duplicates__minimax_reference | 38.0732 | 0.748555 |
| noise__minimax_reference | 38.0732 | 0.748555 |
| near_copies__minimax_reference | 37.4564 | 0.745483 |
| structured_noise__minimax_reference | 38.0732 | 0.748555 |
| base__mass | 38.9622 | 0.754189 |
| duplicates__mass | 38.9622 | 0.754189 |
| noise__mass | 37.1621 | 0.739068 |
| near_copies__mass | 37.9303 | 0.743852 |
| structured_noise__mass | 38.9622 | 0.754189 |

Native: {'base': 13769, 'duplicates': 13769, 'noise': 5531, 'near_copies': 8271, 'structured_noise': 13769}
Preservation: {'duplicates': True, 'noise': False, 'near_copies': False, 'structured_noise': True}

```json
{
  "near_copies__mass minus near_copies__feasible_reference": {
    "pb": {
      "point": -0.006113488934798239,
      "low": -0.0189682797420878,
      "high": 0.005921917750299072,
      "confidence": 0.9958333333333333,
      "draws": 10000
    },
    "within": {
      "point": -0.007089347190654016,
      "low": -0.010800135075974859,
      "high": -0.0036238730427396574,
      "confidence": 0.9958333333333333,
      "draws": 10000
    }
  },
  "base__mass minus base__previous": {
    "pb": {
      "point": 0.0023227986586973337,
      "low": -0.007963930294918794,
      "high": 0.01263145530675589,
      "confidence": 0.9958333333333333,
      "draws": 10000
    },
    "within": {
      "point": 0.004111718157372435,
      "low": 0.0015271013760446712,
      "high": 0.006789138176650347,
      "confidence": 0.9958333333333333,
      "draws": 10000
    }
  },
  "duplicates__mass minus base__mass": {
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
  "noise__mass minus base__mass": {
    "pb": {
      "point": -0.018001924214702092,
      "low": -0.031174705102583952,
      "high": -0.005173987056928158,
      "confidence": 0.9958333333333333,
      "draws": 10000
    },
    "within": {
      "point": -0.015121240779570333,
      "low": -0.018944817822890446,
      "high": -0.011304999865268957,
      "confidence": 0.9958333333333333,
      "draws": 10000
    }
  },
  "near_copies__mass minus base__mass": {
    "pb": {
      "point": -0.0103190003622049,
      "low": -0.022144282708367442,
      "high": 0.001430100649281496,
      "confidence": 0.9958333333333333,
      "draws": 10000
    },
    "within": {
      "point": -0.010337127648644251,
      "low": -0.0139227629906016,
      "high": -0.006991518599765436,
      "confidence": 0.9958333333333333,
      "draws": 10000
    }
  },
  "structured_noise__mass minus base__mass": {
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
