# Fixed-bank Joint model-inverse readout

| Method | PB % | Within AUC | Native |
|---|---:|---:|---:|
| base__joint_full | 38.3554 | 0.747238 | frozen reference |
| base__joint_auto | 38.5998 | 0.749097 | frozen reference |
| base__continuous | 38.0111 | 0.748206 | frozen reference |
| base__equal_full | 37.8196 | 0.749591 | frozen reference |
| base__equal_selected | 38.8563 | 0.750717 | frozen reference |
| innovation5 | 39.8314 | 0.760293 | frozen reference |
| historical_bocpd | 40.3676 | 0.763223 | frozen reference |
| duplicates__joint_full | 37.6469 | 0.747697 | frozen reference |
| duplicates__joint_auto | 37.0862 | 0.746547 | frozen reference |
| duplicates__continuous | 37.5348 | 0.745811 | frozen reference |
| duplicates__equal_full | 37.5137 | 0.746661 | frozen reference |
| duplicates__equal_selected | 38.0042 | 0.748331 | frozen reference |
| noise__joint_full | 36.1071 | 0.738825 | frozen reference |
| noise__joint_auto | 36.9844 | 0.739382 | frozen reference |
| noise__continuous | 37.6288 | 0.750537 | frozen reference |
| noise__equal_full | 37.8900 | 0.748021 | frozen reference |
| noise__equal_selected | 37.9903 | 0.746049 | frozen reference |
| base__inverse_full | 37.5768 | 0.745942 | 13769 |
| base__inverse_auto | 37.3178 | 0.744936 | 13769 |
| duplicates__inverse_full | 36.7837 | 0.734612 | 13769 |
| duplicates__inverse_auto | 36.6840 | 0.739352 | 13769 |
| noise__inverse_full | 37.9043 | 0.745135 | 11039 |
| noise__inverse_auto | 37.7149 | 0.744893 | 11039 |

Noise inverse heads preserve the original2730-answer H1 fallback.
Only the readout changes. Inverse is the historical lambda0 model map, fixed condition1000.
Practical preservation: {'duplicates': False, 'noise': False}

```json
{
  "base__inverse_auto minus base__joint_auto": {
    "pb": {
      "point": -0.012820479616690972,
      "low": -0.026872107020560757,
      "high": 0.001268423661463796,
      "confidence": 0.995,
      "draws": 10000
    },
    "within": {
      "point": -0.004160915484512806,
      "low": -0.006795436817133387,
      "high": -0.0013306085196149464,
      "confidence": 0.995,
      "draws": 10000
    }
  },
  "duplicates__inverse_auto minus duplicates__joint_auto": {
    "pb": {
      "point": -0.004021599523042263,
      "low": -0.022163989559305323,
      "high": 0.015449132225475271,
      "confidence": 0.995,
      "draws": 10000
    },
    "within": {
      "point": -0.007195123179319585,
      "low": -0.010838347388671192,
      "high": -0.0032995390180781477,
      "confidence": 0.995,
      "draws": 10000
    }
  },
  "noise__inverse_auto minus noise__joint_auto": {
    "pb": {
      "point": 0.007305279440307122,
      "low": -0.004027486602633841,
      "high": 0.01916943859814462,
      "confidence": 0.995,
      "draws": 10000
    },
    "within": {
      "point": 0.005511126833816049,
      "low": 0.002412174759018905,
      "high": 0.008942541198815681,
      "confidence": 0.995,
      "draws": 10000
    }
  },
  "duplicates__inverse_auto minus base__inverse_auto": {
    "pb": {
      "point": -0.006337238616561824,
      "low": -0.016706955535832294,
      "high": 0.004205160131588683,
      "confidence": 0.995,
      "draws": 10000
    },
    "within": {
      "point": -0.005583707333968735,
      "low": -0.007689418018756815,
      "high": -0.0034417102834616875,
      "confidence": 0.995,
      "draws": 10000
    }
  },
  "noise__inverse_auto minus base__inverse_auto": {
    "pb": {
      "point": 0.003971827878842982,
      "low": -0.003315285624782225,
      "high": 0.011900659973434721,
      "confidence": 0.995,
      "draws": 10000
    },
    "within": {
      "point": -4.281200273770214e-05,
      "low": -0.0018505947481666993,
      "high": 0.0017971662595627866,
      "confidence": 0.995,
      "draws": 10000
    }
  }
}
```
