# Full-data Joint redundancy/noise stress

| Method | PB % | PRMB within AUC | Native answers |
|---|---:|---:|---:|
| base__joint_full | 38.3554 | 0.747238 | frozen reference |
| base__joint_auto | 38.5998 | 0.749097 | frozen reference |
| base__continuous | 38.0111 | 0.748206 | frozen reference |
| base__equal_full | 37.8196 | 0.749591 | frozen reference |
| base__equal_selected | 38.8563 | 0.750717 | frozen reference |
| innovation5 | 39.8314 | 0.760293 | frozen reference |
| historical_bocpd | 40.3676 | 0.763223 | frozen reference |
| duplicates__joint_full | 37.6469 | 0.747697 | 13769 |
| duplicates__joint_auto | 37.0862 | 0.746547 | 13769 |
| duplicates__continuous | 37.5348 | 0.745811 | 13769 |
| duplicates__equal_full | 37.5137 | 0.746661 | 13769 |
| duplicates__equal_selected | 38.0042 | 0.748331 | 13769 |
| noise__joint_full | 36.1071 | 0.738825 | 11039 |
| noise__joint_auto | 36.9844 | 0.739382 | 11039 |
| noise__continuous | 37.6288 | 0.750537 | 13769 |
| noise__equal_full | 37.8900 | 0.748021 | 13769 |
| noise__equal_selected | 37.9903 | 0.746049 | 11039 |

Practical preservation (-1pp PB / -.002 within, corrected intervals): {'duplicates__joint_auto': False, 'duplicates__joint_full': False, 'noise__joint_auto': False, 'noise__joint_full': False}

Exact-copy prediction invariance is separate; see stability and native failures in RESULTS.json.

```json
{
  "duplicates__joint_auto minus base__joint_auto": {
    "pb": {
      "point": -0.015136118710210533,
      "low": -0.02799758145932849,
      "high": -0.0023076343409193937,
      "confidence": 0.99375,
      "draws": 10000
    },
    "within": {
      "point": -0.002549499639161956,
      "low": -0.005401509744919236,
      "high": 0.00015863993648631942,
      "confidence": 0.99375,
      "draws": 10000
    }
  },
  "duplicates__joint_full minus base__joint_full": {
    "pb": {
      "point": -0.007085221565471811,
      "low": -0.019043506459095617,
      "high": 0.004566639845916192,
      "confidence": 0.99375,
      "draws": 10000
    },
    "within": {
      "point": 0.00045929563973767795,
      "low": -0.002008761704163646,
      "high": 0.0029706188438317405,
      "confidence": 0.99375,
      "draws": 10000
    }
  },
  "noise__joint_auto minus base__joint_auto": {
    "pb": {
      "point": -0.016153931178155112,
      "low": -0.03122861935672717,
      "high": 5.519335274367075e-05,
      "confidence": 0.99375,
      "draws": 10000
    },
    "within": {
      "point": -0.009714854321066557,
      "low": -0.013551656774572882,
      "high": -0.006167646350323333,
      "confidence": 0.99375,
      "draws": 10000
    }
  },
  "noise__joint_full minus base__joint_full": {
    "pb": {
      "point": -0.02248346971037951,
      "low": -0.036430403875466255,
      "high": -0.008124709784590315,
      "confidence": 0.99375,
      "draws": 10000
    },
    "within": {
      "point": -0.008412616739856382,
      "low": -0.01224761250210788,
      "high": -0.004646476802951004,
      "confidence": 0.99375,
      "draws": 10000
    }
  }
}
```
