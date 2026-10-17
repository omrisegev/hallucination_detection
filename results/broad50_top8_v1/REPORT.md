# Matched Top8 follow-up

| Method | PB % | PRMB within AUC |
|---|---:|---:|
| B50__joint_full | 37.6203 | 0.747899 |
| B50__joint_auto | 37.4147 | 0.747475 |
| B50__equal_full | 37.7080 | 0.751224 |
| B50__equal_selected | 38.0435 | 0.751094 |
| B50__continuous | 37.7705 | 0.750229 |
| B51_bocpd__joint_full | 37.2996 | 0.751047 |
| B51_bocpd__joint_auto | 36.8625 | 0.752522 |
| B51_bocpd__equal_full | 37.8606 | 0.752265 |
| B51_bocpd__equal_selected | 38.8756 | 0.754703 |
| B51_bocpd__continuous | 37.8422 | 0.750856 |
| B50__top10_auto | 38.2737 | 0.748011 |
| B51_bocpd__top10_auto | 38.5998 | 0.749097 |
| innovation5 | 39.8314 | 0.760293 |
| historical_bocpd | 40.3676 | 0.763223 |

Primary readout contrasts,98.75% source-group intervals:
```json
{
  "B50__joint_auto minus B50__top10_auto": {
    "pb": {
      "point": -0.008590565298044195,
      "low": -0.022153506389898266,
      "high": 0.004565084501825819,
      "confidence": 0.9875,
      "draws": 10000
    },
    "within": {
      "point": -0.0005367163294521804,
      "low": -0.0036753765666154006,
      "high": 0.0024915842380465207,
      "confidence": 0.9875,
      "draws": 10000
    }
  },
  "B51_bocpd__joint_auto minus B51_bocpd__top10_auto": {
    "pb": {
      "point": -0.017373333259050827,
      "low": -0.029211640098858314,
      "high": -0.0052017172334711615,
      "confidence": 0.9875,
      "draws": 10000
    },
    "within": {
      "point": 0.0034254674302173216,
      "low": 0.0008044560210649316,
      "high": 0.006156532084832326,
      "confidence": 0.9875,
      "draws": 10000
    }
  }
}
```
