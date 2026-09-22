# Alternating Joint feature selection plus BOCPD

Full development population; 95% information retention rule; frozen non-digit PB gate.

| Method | PB macro % | PRMB within AUC |
|---|---:|---:|
| B50__joint_full | 38.0485 | 0.746070 |
| B50__joint_auto | 38.2737 | 0.748011 |
| B50__equal_full | 37.8963 | 0.748902 |
| B50__equal_selected | 38.4216 | 0.749734 |
| B50__continuous | 37.7632 | 0.747926 |
| B50__joint_keep40 | 38.1074 | 0.747092 |
| B50__joint_keep30 | 38.2193 | 0.748062 |
| B50__joint_keep20 | 38.1079 | 0.746771 |
| B50__joint_keep12 | 37.9015 | 0.743534 |
| B50__joint_keep8 | 38.0165 | 0.741941 |
| B51_bocpd__joint_full | 38.3554 | 0.747238 |
| B51_bocpd__joint_auto | 38.5998 | 0.749097 |
| B51_bocpd__equal_full | 37.8196 | 0.749591 |
| B51_bocpd__equal_selected | 38.8563 | 0.750717 |
| B51_bocpd__continuous | 38.0111 | 0.748206 |
| B51_noreset__joint_full | 37.8891 | 0.747294 |
| B51_noreset__joint_auto | 38.2131 | 0.749471 |
| B51_noreset__equal_full | 38.0308 | 0.749630 |
| B51_noreset__equal_selected | 38.7678 | 0.750907 |
| B51_noreset__continuous | 37.8603 | 0.748775 |
| original4 | 37.4749 | 0.753436 |
| innovation5 | 39.8314 | 0.760293 |
| historical_bocpd | 40.3676 | 0.763223 |
| historical_noreset | 39.8608 | 0.762839 |

Automatic retained counts: {'B50': [33, 38, 34, 35, 38], 'B51_bocpd': [33, 35, 36, 36, 35], 'B51_noreset': [36, 36, 35, 36, 35]}

Fixed-count paths are descriptive; none was chosen using outer labels as the automatic rule.
BOCPD means the pure cached signed residual channel; historical_bocpd includes its old innovation5 correction.

Primary paired contrasts (8 endpoints corrected):
```json
{
  "B50__joint_auto minus B50__joint_full": {
    "pb": {
      "point": 0.0022528817716523375,
      "low": -0.005999599002490765,
      "high": 0.009667810630268177,
      "confidence": 0.99375,
      "draws": 10000
    },
    "within": {
      "point": 0.0019416042318822768,
      "low": 0.0003161969517007445,
      "high": 0.003635308347434129,
      "confidence": 0.99375,
      "draws": 10000
    }
  },
  "B51_bocpd__joint_auto minus B51_bocpd__joint_full": {
    "pb": {
      "point": 0.002444060154242811,
      "low": -0.00625340017710075,
      "high": 0.010412199220473917,
      "confidence": 0.99375,
      "draws": 10000
    },
    "within": {
      "point": 0.001858798921210325,
      "low": 0.00024289337510339837,
      "high": 0.003408424933509311,
      "confidence": 0.99375,
      "draws": 10000
    }
  },
  "B51_bocpd__joint_auto minus B50__joint_auto": {
    "pb": {
      "point": 0.0032606747054952367,
      "low": -0.0065946065333957575,
      "high": 0.01324976240351149,
      "confidence": 0.99375,
      "draws": 10000
    },
    "within": {
      "point": 0.0010852082364471993,
      "low": -0.0011910159073315295,
      "high": 0.0033205353689479506,
      "confidence": 0.99375,
      "draws": 10000
    }
  },
  "B51_bocpd__joint_auto minus B51_noreset__joint_auto": {
    "pb": {
      "point": 0.0038674638818597673,
      "low": -0.007419798415906708,
      "high": 0.015573041887183463,
      "confidence": 0.99375,
      "draws": 10000
    },
    "within": {
      "point": -0.0003741394027944045,
      "low": -0.003062612575091311,
      "high": 0.0021711483177490713,
      "confidence": 0.99375,
      "draws": 10000
    }
  }
}
```
