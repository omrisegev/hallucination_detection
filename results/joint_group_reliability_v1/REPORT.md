# Model-derived outer group reliability

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
| base__inverse_full | 37.5768 | 0.745942 | frozen reference |
| base__inverse_auto | 37.3178 | 0.744936 | frozen reference |
| duplicates__inverse_full | 36.7837 | 0.734612 | frozen reference |
| duplicates__inverse_auto | 36.6840 | 0.739352 | frozen reference |
| noise__inverse_full | 37.9043 | 0.745135 | frozen reference |
| noise__inverse_auto | 37.7149 | 0.744893 | frozen reference |
| base__reliability_full | 38.0948 | 0.748971 | 13769 |
| base__reliability_auto | 38.7300 | 0.750078 | 13769 |
| duplicates__reliability_full | 37.9485 | 0.749833 | 13769 |
| duplicates__reliability_auto | 37.6801 | 0.748921 | 13769 |
| noise__reliability_full | 37.1954 | 0.743744 | 11039 |
| noise__reliability_auto | 38.0338 | 0.745776 | 11039 |

Noise candidates include2730 explicit H1 fallback answers.
Only outer group weighting changed; all fits, selected supports and within-group v weights frozen.
Practical preservation: {'duplicates': False, 'noise': False}

```json
{
  "base__reliability_auto minus base__joint_auto": {
    "pb": {
      "point": 0.0013015526983947145,
      "low": -0.00658350924929671,
      "high": 0.009240505038353651,
      "confidence": 0.995,
      "draws": 10000
    },
    "within": {
      "point": 0.000981155257240296,
      "low": -0.0008103452807432759,
      "high": 0.0026830520235753257,
      "confidence": 0.995,
      "draws": 10000
    }
  },
  "duplicates__reliability_auto minus duplicates__joint_auto": {
    "pb": {
      "point": 0.005938998141394036,
      "low": -0.0032825822682284226,
      "high": 0.015366759234851443,
      "confidence": 0.995,
      "draws": 10000
    },
    "within": {
      "point": 0.002373968244535396,
      "low": 0.0009041620770274453,
      "high": 0.003879416954452897,
      "confidence": 0.995,
      "draws": 10000
    }
  },
  "noise__reliability_auto minus noise__joint_auto": {
    "pb": {
      "point": 0.01049430402252749,
      "low": 0.0002319629118909153,
      "high": 0.020138337371579152,
      "confidence": 0.995,
      "draws": 10000
    },
    "within": {
      "point": 0.006393825425521138,
      "low": 0.0038104061233914825,
      "high": 0.009204874156591037,
      "confidence": 0.995,
      "draws": 10000
    }
  },
  "duplicates__reliability_auto minus base__reliability_auto": {
    "pb": {
      "point": -0.010498673267211212,
      "low": -0.01951037910111644,
      "high": -0.0017539400074458307,
      "confidence": 0.995,
      "draws": 10000
    },
    "within": {
      "point": -0.0011566866518668562,
      "low": -0.0029830579157603355,
      "high": 0.0008242290775227651,
      "confidence": 0.995,
      "draws": 10000
    }
  },
  "noise__reliability_auto minus base__reliability_auto": {
    "pb": {
      "point": -0.006961179854022337,
      "low": -0.01946825849164206,
      "high": 0.005103126317568436,
      "confidence": 0.995,
      "draws": 10000
    },
    "within": {
      "point": -0.004302184152785715,
      "low": -0.006994470581840233,
      "high": -0.0015751100405860376,
      "confidence": 0.995,
      "draws": 10000
    }
  }
}
```
