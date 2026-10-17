# Sparse Joint loading membership

| Method | PB % | Within AUC | Native |
|---|---:|---:|---:|
| base__joint_full | 38.3554 | 0.747238 | reference/control |
| base__joint_auto | 38.5998 | 0.749097 | reference/control |
| base__continuous | 38.0111 | 0.748206 | reference/control |
| base__equal_full | 37.8196 | 0.749591 | reference/control |
| base__equal_selected | 38.8563 | 0.750717 | reference/control |
| innovation5 | 39.8314 | 0.760293 | reference/control |
| historical_bocpd | 40.3676 | 0.763223 | reference/control |
| duplicates__joint_full | 37.6469 | 0.747697 | reference/control |
| duplicates__joint_auto | 37.0862 | 0.746547 | reference/control |
| duplicates__continuous | 37.5348 | 0.745811 | reference/control |
| duplicates__equal_full | 37.5137 | 0.746661 | reference/control |
| duplicates__equal_selected | 38.0042 | 0.748331 | reference/control |
| noise__joint_full | 36.1071 | 0.738825 | reference/control |
| noise__joint_auto | 36.9844 | 0.739382 | reference/control |
| noise__continuous | 37.6288 | 0.750537 | reference/control |
| noise__equal_full | 37.8900 | 0.748021 | reference/control |
| noise__equal_selected | 37.9903 | 0.746049 | reference/control |
| base__inverse_full | 37.5768 | 0.745942 | reference/control |
| base__inverse_auto | 37.3178 | 0.744936 | reference/control |
| duplicates__inverse_full | 36.7837 | 0.734612 | reference/control |
| duplicates__inverse_auto | 36.6840 | 0.739352 | reference/control |
| noise__inverse_full | 37.9043 | 0.745135 | reference/control |
| noise__inverse_auto | 37.7149 | 0.744893 | reference/control |
| base__reliability_full | 38.0948 | 0.748971 | reference/control |
| base__reliability_auto | 38.7300 | 0.750078 | reference/control |
| duplicates__reliability_full | 37.9485 | 0.749833 | reference/control |
| duplicates__reliability_auto | 37.6801 | 0.748921 | reference/control |
| noise__reliability_full | 37.1954 | 0.743744 | reference/control |
| noise__reliability_auto | 38.0338 | 0.745776 | reference/control |
| base__sparse | 38.0948 | 0.748971 | 13769 |
| base__alias_control | 38.0948 | 0.748971 | reference/control |
| duplicates__sparse | 38.0948 | 0.748971 | 13769 |
| duplicates__alias_control | 38.0948 | 0.748971 | reference/control |
| noise__sparse | 38.0948 | 0.748971 | 13769 |
| noise__alias_control | 37.1954 | 0.743744 | reference/control |

Practical preservation: {'duplicates': True, 'noise': True}
Selected counts: {'base': [51, 51, 51, 51, 51], 'duplicates': [51, 51, 51, 51, 51], 'noise': [51, 51, 51, 51, 51]}
Selected additions: {'base': [0, 0, 0, 0, 0], 'duplicates': [15, 15, 15, 15, 15], 'noise': [0, 0, 0, 0, 0]}

```json
{
  "base__sparse minus base__reliability_auto": {
    "pb": {
      "point": -0.006351857774123837,
      "low": -0.015190770008243162,
      "high": 0.002265489062456911,
      "confidence": 0.995,
      "draws": 10000
    },
    "within": {
      "point": -0.0011069595976338498,
      "low": -0.002885687626280138,
      "high": 0.0006534104085981136,
      "confidence": 0.995,
      "draws": 10000
    }
  },
  "duplicates__sparse minus duplicates__reliability_auto": {
    "pb": {
      "point": 0.004146815493087375,
      "low": -0.005435000089428318,
      "high": 0.014087679432305166,
      "confidence": 0.995,
      "draws": 10000
    },
    "within": {
      "point": 4.9727054233006385e-05,
      "low": -0.0014291494828857203,
      "high": 0.001482697154174417,
      "confidence": 0.995,
      "draws": 10000
    }
  },
  "noise__sparse minus noise__reliability_auto": {
    "pb": {
      "point": 0.0006093220798985,
      "low": -0.01182304867953213,
      "high": 0.013924575737270746,
      "confidence": 0.995,
      "draws": 10000
    },
    "within": {
      "point": 0.0031952245551518654,
      "low": 0.0002618154495305225,
      "high": 0.006122507846106956,
      "confidence": 0.995,
      "draws": 10000
    }
  },
  "duplicates__sparse minus base__sparse": {
    "pb": {
      "point": 0.0,
      "low": 0.0,
      "high": 0.0,
      "confidence": 0.995,
      "draws": 10000
    },
    "within": {
      "point": 0.0,
      "low": 0.0,
      "high": 0.0,
      "confidence": 0.995,
      "draws": 10000
    }
  },
  "noise__sparse minus base__sparse": {
    "pb": {
      "point": 0.0,
      "low": 0.0,
      "high": 0.0,
      "confidence": 0.995,
      "draws": 10000
    },
    "within": {
      "point": 0.0,
      "low": 0.0,
      "high": 0.0,
      "confidence": 0.995,
      "draws": 10000
    }
  }
}
```
