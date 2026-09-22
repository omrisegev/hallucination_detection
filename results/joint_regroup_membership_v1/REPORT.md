# Regrouped refinement: full matched evaluation

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
| base__regroup | 38.3374 | 0.750672 |
| duplicates__regroup | 38.3374 | 0.750672 |
| noise__regroup | 38.3374 | 0.750672 |
| near_copies__regroup | 38.2991 | 0.750572 |
| structured_noise__regroup | 38.3374 | 0.750672 |

Native: {'base': 13769, 'duplicates': 13769, 'noise': 13769, 'near_copies': 13769, 'structured_noise': 13769}
Preservation: {'duplicates': True, 'noise': True, 'near_copies': True, 'structured_noise': True}

```json
{
  "near_copies__regroup minus near_copies__previous": {
    "pb": {
      "point": 0.005805990817754458,
      "low": -0.0030917745486197736,
      "high": 0.01474501188950522,
      "confidence": 0.9958333333333333,
      "draws": 10000
    },
    "within": {
      "point": 0.0018492230187882042,
      "low": 2.069053978974723e-05,
      "high": 0.003724215838134731,
      "confidence": 0.9958333333333333,
      "draws": 10000
    }
  },
  "base__regroup minus base__previous": {
    "pb": {
      "point": -0.00392607859996863,
      "low": -0.012083647662821878,
      "high": 0.004105792762689876,
      "confidence": 0.9958333333333333,
      "draws": 10000
    },
    "within": {
      "point": 0.0005944386652199096,
      "low": -0.0012671086051920162,
      "high": 0.0024466491728561547,
      "confidence": 0.9958333333333333,
      "draws": 10000
    }
  },
  "duplicates__regroup minus base__regroup": {
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
  "noise__regroup minus base__regroup": {
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
  "near_copies__regroup minus base__regroup": {
    "pb": {
      "point": -0.0003830750923658477,
      "low": -0.0047798656585000135,
      "high": 0.0040154948310614415,
      "confidence": 0.9958333333333333,
      "draws": 10000
    },
    "within": {
      "point": -0.00010008496140545553,
      "low": -0.0010998419953216897,
      "high": 0.0008156101715867299,
      "confidence": 0.9958333333333333,
      "draws": 10000
    }
  },
  "structured_noise__regroup minus base__regroup": {
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
