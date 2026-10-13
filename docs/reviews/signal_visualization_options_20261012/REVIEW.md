# Validation receipt

Independent read-only reviewer verified 2,208 numerical checks against frozen
source arrays and CSV, maximum discrepancy 8.88e-16. Checks included family
means and intervals, paired deltas, source hashes, 707 source groups, 56/16
channel membership, frozen orientations, eight error-type mappings, and absence
of undefined MS AUC. Source population: 6,030 raw / 4,691 position-controlled
eligible answers. Forest ranges contain all displayed intervals.

The reviewer identified the last-to-first seam omitted by a single band polygon.
The final template uses eight explicit quadrilateral band sectors with modulo
axis indexing. Full error-code legend is visible beside shared controls.

Browser checks: all six views render; N16 selects five family radars and 16
individual channel points; position control updates eligibility counts; feature
delta mode renders; rank paths and scatter selectors render; SVG download works.
No browser error/warning messages were captured. Responsive layout checked at
1200 and 375 pixels; wide graphs use horizontal scrolling on narrow screens.

No model inference, feature fitting, bank selection, or original advisor-page
replacement occurred during the original A–F construction review. Git delivery
followed separately in f54aa811; the current reading delivery adds G/H.

## G/H independent review

Separate reviewer `/root/gallery_extensions_review` accepted the extension data
and UI mappings. Every donor Q80 threshold and source-OOF flag was independently
replayed; 282 conditional phi/MI pairs matched SciPy/sklearn within 1e-12, and
40 PRMScore/MS F1 calculations matched sklearn. Family means/memberships and
source/output hashes passed. Binary fixtures covered independence, perfectly
opposed errors, constants and missing classes. UI review verified same-peer
intersection across labels, self exclusion, max of class-mean absolute phi,
undefined handling and family-summary versus fusion-score labels.

G is descriptive pairwise conditional dependence, with error rates/peer counts;
it is not an independence test. H shows raw member-channel performance, with MS
valid-step F1 separated from the other eight macro-F1 curves. Browser validation
checked 56/16 G points, outside-family and single-peer options, absent MS error
class, raw/position AUC, 45 N16-family PRM points, 27 selected-feature PRM points,
unchanged PRM values under position control, and no console errors.

The A–F DATA.json is byte-identical to the published base. The A–H reading
delivery contains this review and the extension summaries; raw NPZ arrays stay
local with hashes. The original frozen v11 and extension prediction-seal files
were preserved. Publication is verified separately through the remote commit.
