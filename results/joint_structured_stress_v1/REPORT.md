# Approximate-copy and structured nuisance verification

| Method | PB % | Within AUC | Native |
|---|---:|---:|---:|
| base__refined | 38.7300 | 0.750078 | frozen reference |
| base__continuous | 38.0111 | 0.748206 | frozen reference |
| base__equal_full | 37.8196 | 0.749591 | frozen reference |
| base__reliability_auto | 38.7300 | 0.750078 | frozen reference |
| innovation5 | 39.8314 | 0.760293 | frozen reference |
| historical_bocpd | 40.3676 | 0.763223 | frozen reference |
| near_copies__joint | 37.7185 | 0.748723 | 13769 |
| near_copies__continuous | 37.5544 | 0.745933 | 13769 |
| near_copies__equal | 37.5929 | 0.746523 | 13769 |
| structured_noise__joint | 37.9235 | 0.748005 | 13769 |
| structured_noise__continuous | 37.8974 | 0.749137 | 13769 |
| structured_noise__equal | 36.8025 | 0.742034 | 13769 |

Practical preservation: {'near_copies': False, 'structured_noise': False}
Added retained: {'near_copies': [0, 0, 0, 0, 0], 'structured_noise': [15, 15, 15, 15, 15]}
Original retained: {'near_copies': [23, 20, 22, 22, 23], 'structured_noise': [30, 29, 27, 27, 28]}

```json
{
  "near_copies__joint minus base__refined": {
    "pb": {
      "point": -0.010115144510088936,
      "low": -0.01798079544400916,
      "high": -0.0024155150536861146,
      "confidence": 0.9875,
      "draws": 10000
    },
    "within": {
      "point": -0.00135486931497375,
      "low": -0.0030025491048556777,
      "high": 0.0003611697213539189,
      "confidence": 0.9875,
      "draws": 10000
    }
  },
  "structured_noise__joint minus base__refined": {
    "pb": {
      "point": -0.008064937906284708,
      "low": -0.01939654085376227,
      "high": 0.003530117636192647,
      "confidence": 0.9875,
      "draws": 10000
    },
    "within": {
      "point": -0.002072328874888729,
      "low": -0.004335021174811958,
      "high": 0.0002463762966889627,
      "confidence": 0.9875,
      "draws": 10000
    }
  }
}
```
