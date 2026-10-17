# Global-signal membership: full matched evaluation

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
| base__signal | 38.7300 | 0.750078 |
| duplicates__signal | 38.7300 | 0.750078 |
| noise__signal | 38.3420 | 0.746950 |
| near_copies__signal | 37.7185 | 0.748723 |
| structured_noise__signal | 38.7300 | 0.750078 |

Native: {'base': 13769, 'duplicates': 13769, 'noise': 11039, 'near_copies': 13769, 'structured_noise': 13769}
Preservation: {'duplicates': True, 'noise': False, 'near_copies': False, 'structured_noise': True}

```json
{
  "structured_noise__signal minus structured_noise__old": {
    "pb": {
      "point": 0.008064937906284708,
      "low": -0.005108631192629038,
      "high": 0.020978613963117197,
      "confidence": 0.9958333333333333,
      "draws": 10000
    },
    "within": {
      "point": 0.002072328874888729,
      "low": -0.0005400542881759355,
      "high": 0.004692262575712922,
      "confidence": 0.9958333333333333,
      "draws": 10000
    }
  },
  "base__signal minus base__old": {
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
  "duplicates__signal minus base__signal": {
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
  "noise__signal minus base__signal": {
    "pb": {
      "point": -0.0038794984928818277,
      "low": -0.010185060841110717,
      "high": 0.002814688088566619,
      "confidence": 0.9958333333333333,
      "draws": 10000
    },
    "within": {
      "point": -0.003127265188863526,
      "low": -0.0048766176687546434,
      "high": -0.0016046501612181296,
      "confidence": 0.9958333333333333,
      "draws": 10000
    }
  },
  "near_copies__signal minus base__signal": {
    "pb": {
      "point": -0.010115144510088936,
      "low": -0.019304093006859816,
      "high": -0.0011177128112280772,
      "confidence": 0.9958333333333333,
      "draws": 10000
    },
    "within": {
      "point": -0.00135486931497375,
      "low": -0.0031794272236418245,
      "high": 0.0007317816997363841,
      "confidence": 0.9958333333333333,
      "draws": 10000
    }
  },
  "structured_noise__signal minus base__signal": {
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
