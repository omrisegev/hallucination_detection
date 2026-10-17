# Staged membership: full matched evaluation

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
| noise__previous | 38.3420 | 0.746950 |
| near_copies__previous | 37.7185 | 0.748723 |
| structured_noise__previous | 38.7300 | 0.750078 |
| base__staged | 38.7300 | 0.750078 |
| duplicates__staged | 38.7300 | 0.750078 |
| noise__staged | 38.7300 | 0.750078 |
| near_copies__staged | 37.7185 | 0.748723 |
| structured_noise__staged | 38.7300 | 0.750078 |

Native: {'base': 13769, 'duplicates': 13769, 'noise': 13769, 'near_copies': 13769, 'structured_noise': 13769}
Preservation: {'duplicates': True, 'noise': True, 'near_copies': False, 'structured_noise': True}

```json
{
  "noise__staged minus noise__previous": {
    "pb": {
      "point": 0.0038794984928818277,
      "low": -0.0028146880885665934,
      "high": 0.010185060841110727,
      "confidence": 0.9958333333333333,
      "draws": 10000
    },
    "within": {
      "point": 0.003127265188863526,
      "low": 0.0016046501612181365,
      "high": 0.004876617668754644,
      "confidence": 0.9958333333333333,
      "draws": 10000
    }
  },
  "base__staged minus base__previous": {
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
  "duplicates__staged minus base__staged": {
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
  "noise__staged minus base__staged": {
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
  "near_copies__staged minus base__staged": {
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
  "structured_noise__staged minus base__staged": {
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
