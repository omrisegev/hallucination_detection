# Information-minimax refinement: full matched evaluation

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
| base__minimax | 38.0732 | 0.748555 |
| duplicates__minimax | 38.0732 | 0.748555 |
| noise__minimax | 38.0732 | 0.748555 |
| near_copies__minimax | 37.4564 | 0.745483 |
| structured_noise__minimax | 38.0732 | 0.748555 |

Native: {'base': 13769, 'duplicates': 13769, 'noise': 13769, 'near_copies': 13769, 'structured_noise': 13769}
Preservation: {'duplicates': True, 'noise': True, 'near_copies': False, 'structured_noise': True}

```json
{
  "near_copies__minimax minus near_copies__feasible_reference": {
    "pb": {
      "point": -0.010852936243763933,
      "low": -0.019692455935462167,
      "high": -0.0022351270048552954,
      "confidence": 0.9958333333333333,
      "draws": 10000
    },
    "within": {
      "point": -0.005458466889698865,
      "low": -0.007405846268224005,
      "high": -0.003485610041534307,
      "confidence": 0.9958333333333333,
      "draws": 10000
    }
  },
  "base__minimax minus base__previous": {
    "pb": {
      "point": -0.006567683774640232,
      "low": -0.01392982236326055,
      "high": 0.0005940541622801066,
      "confidence": 0.9958333333333333,
      "draws": 10000
    },
    "within": {
      "point": -0.0015231762775137492,
      "low": -0.003134007645515892,
      "high": 6.16187632774051e-05,
      "confidence": 0.9958333333333333,
      "draws": 10000
    }
  },
  "duplicates__minimax minus base__minimax": {
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
  "noise__minimax minus base__minimax": {
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
  "near_copies__minimax minus base__minimax": {
    "pb": {
      "point": -0.0061679652378330285,
      "low": -0.014944168786234299,
      "high": 0.002539192035642997,
      "confidence": 0.9958333333333333,
      "draws": 10000
    },
    "within": {
      "point": -0.003071352912802916,
      "low": -0.005014377490399179,
      "high": -0.0011072781421938205,
      "confidence": 0.9958333333333333,
      "draws": 10000
    }
  },
  "structured_noise__minimax minus base__minimax": {
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
